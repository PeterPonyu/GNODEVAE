import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm

class SC3:
    """
    Python implementation of the SC3 (Single-Cell Consensus Clustering) algorithm.

    This class provides a streamlined workflow to perform consensus clustering on
    single-cell RNA-seq data, taking an AnnData object as input and producing
    cluster assignments.

    The workflow consists of:
    1. Gene filtering.
    2. Calculating multiple distance matrices (Euclidean, Pearson, Spearman).
    3. Transforming these matrices (PCA, Laplacian).
    4. Performing k-means clustering on all transformed matrices for a range of k.
    5. Building a consensus matrix from all clustering results for each k.
    6. Hierarchical clustering on the consensus matrix to get final labels.
    """

    def __init__(self, adata: AnnData, use_highly_variable: bool = True, layer: str = None):
        """
        Initializes the SC3 object.

        Args:
            adata: An AnnData object containing the single-cell data.
            use_highly_variable: If True, uses highly variable genes. 
                                 If False, uses all genes.
            layer: The layer in adata to use for calculations. If None, uses adata.X.
        """
        self.adata = adata.copy()
        self.use_highly_variable = use_highly_variable
        self.layer = layer
        self._prepare_data()

    def _prepare_data(self):
        """Selects the data layer and subsets to highly variable genes if specified."""
        if self.layer and self.layer in self.adata.layers:
            self.adata.X = self.adata.layers[self.layer]
        
        if self.use_highly_variable:
            if 'highly_variable' not in self.adata.var.keys():
                print("Highly variable genes not found. Calculating...")
                sc.pp.highly_variable_genes(self.adata)
            self.adata = self.adata[:, self.adata.var['highly_variable']].copy()

    def fit(self, ks, d_region_min=0.02, d_region_max=0.05):
        """
        Executes the complete SC3 workflow.

        Args:
            ks (list or int): A list or a single integer for the number of clusters k.
            d_region_min: Minimum number of eigenvectors (as a fraction of cells).
            d_region_max: Maximum number of eigenvectors (as a fraction of cells).

        Returns:
            The fitted SC3 object.
        """
        if isinstance(ks, int):
            ks = [ks]
        
        print("1. Calculating distance matrices...")
        distance_matrices = self._calculate_distances()
        
        print("2. Calculating transformations...")
        n_cells = self.adata.n_obs
        n_dim_min = int(np.floor(d_region_min * n_cells))
        n_dim_max = int(np.ceil(d_region_max * n_cells))
        dims_range = list(range(max(2, n_dim_min), min(n_cells - 1, n_dim_max + 1)))
        transformed_matrices = self._calculate_transformations(distance_matrices, max(dims_range))

        print("3. Running k-means clustering...")
        kmeans_results = self._run_kmeans(transformed_matrices, ks, dims_range)
        
        print("4. Calculating consensus matrices...")
        self._calculate_consensus(kmeans_results, ks)
        
        print("SC3 analysis complete.")
        return self

    def get_embedding(self, k: int) -> np.ndarray:
        """
        Returns the consensus matrix for a given k, which can be viewed as a 
        cell-by-cell similarity embedding.
        """
        if 'sc3_consensus_matrices' in self.adata.uns and k in self.adata.uns['sc3_consensus_matrices']:
            return self.adata.uns['sc3_consensus_matrices'][k]
        else:
            raise ValueError(f"Consensus matrix for k={k} not found. Please run .fit() for this k first.")

    def _calculate_distances(self):
        # self.adata.X is (n_cells, n_genes)
        if hasattr(self.adata.X, 'toarray'):
            data = self.adata.X.toarray()
        else:
            data = self.adata.X

        # For correlation between cells, cells should be columns.
        # So we need (n_genes, n_cells)
        data_t = data.T

        # Euclidean is between rows of the input. So input should be (n_cells, n_genes)
        euclidean_dist = euclidean_distances(data)

        # Pearson correlation. Input is (n_genes, n_cells)
        # We need to check for cells (columns) with zero variance.
        col_vars = np.var(data_t, axis=0)
        valid_cell_mask = col_vars > 0
        
        pearson_dist = np.ones((data.shape[0], data.shape[0]))
        
        if np.sum(valid_cell_mask) > 1:
            # subset the data to only include cells with non-zero variance
            valid_data = data_t[:, valid_cell_mask]
            # Calculate correlation on the valid cells. The result will be smaller.
            corr_matrix = np.corrcoef(valid_data, rowvar=False)
            # Place the results back into the full-size matrix
            ix = np.ix_(valid_cell_mask, valid_cell_mask)
            pearson_dist[ix] = 1 - corr_matrix
        
        # Spearman correlation. Input is (n_genes, n_cells)
        spearman_corr, _ = spearmanr(data_t, axis=0)
        # Handle case where spearmanr returns a single value
        if spearman_corr.ndim == 0:
            spearman_corr = np.array([[1.0]])
        # Replace NaNs that can occur from zero variance columns
        spearman_dist = 1 - np.nan_to_num(spearman_corr)

        return {'euclidean': euclidean_dist, 'pearson': pearson_dist, 'spearman': spearman_dist}

    def _calculate_transformations(self, dists, max_dim):
        transformed = {}
        for name, d in dists.items():
            pca = PCA(n_components=max_dim)
            transformed[f'{name}_pca'] = pca.fit_transform(d)
            
            A = np.exp(-d / d.max())
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1)))
            L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
            eigvals, eigvecs = eigh(L)
            sorted_indices = np.argsort(eigvals)
            transformed[f'{name}_laplacian'] = eigvecs[:, sorted_indices][:, :max_dim]
        return transformed

    def _kmeans_worker(self, task):
        k = task['k']
        dim = task['dim']
        name = task['name']
        data = task['data']
        
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(data)
        return {'key': f'{name}_{k}_{dim}', 'labels': labels}

    def _run_kmeans(self, transfs, ks, dims_range):
        tasks = [{'k': k, 'dim': dim, 'name': name, 'data': data[:, :dim]} 
                 for k in ks for dim in dims_range for name, data in transfs.items()]
        
        results = []
        for task in tqdm(tasks, desc="Running K-Means"):
            results.append(self._kmeans_worker(task))

        return {res['key']: res['labels'] for res in results}

    

    def _calculate_consensus(self, kmeans_results, ks):
        self.adata.uns['sc3_consensus_matrices'] = {}
        for k in ks:
            relevant_keys = [key for key in kmeans_results.keys() if f'_{k}_' in key]
            if not relevant_keys:
                continue
            
            clusterings = np.array([kmeans_results[key] for key in relevant_keys]).T
            n_cells, n_runs = clusterings.shape
            consensus_matrix = np.zeros((n_cells, n_cells))
            
            for i in range(n_cells):
                for j in range(i, n_cells):
                    similarity = np.sum(clusterings[i, :] == clusterings[j, :])
                    consensus_matrix[i, j] = consensus_matrix[j, i] = similarity
            
            consensus_matrix /= n_runs
            distance_from_consensus = 1 - consensus_matrix
            linked = linkage(distance_from_consensus[np.triu_indices(n_cells, k=1)], method='complete')
            labels = fcluster(linked, k, criterion='maxclust')
            
            self.adata.uns['sc3_consensus_matrices'][k] = consensus_matrix
            self.adata.obs[f'sc3_{k}_clusters'] = pd.Categorical(labels)
