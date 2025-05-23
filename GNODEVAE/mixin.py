import torch  
import numpy as np  
from sklearn.cluster import KMeans  
from sklearn.metrics import (  
    normalized_mutual_info_score,  
    adjusted_mutual_info_score,  
    silhouette_score,  
    davies_bouldin_score,  
    calinski_harabasz_score,  
)  
from sklearn.decomposition import (
    PCA, 
    NMF, 
    FastICA, 
    TruncatedSVD,
    FactorAnalysis, 
    LatentDirichletAllocation
)
from typing import Optional, Tuple  
from anndata import AnnData
import scanpy as sc

class scviMixin:  
    def _normal_kl(  
        self,   
        mu1: torch.Tensor,   
        lv1: torch.Tensor,   
        mu2: torch.Tensor,   
        lv2: torch.Tensor  
    ) -> torch.Tensor:  
        """  
        Compute the KL divergence between two normal distributions.  

        Parameters  
        ----------  
        mu1 : torch.Tensor  
            Mean of the first distribution.  
        lv1 : torch.Tensor  
            Log variance of the first distribution.  
        mu2 : torch.Tensor  
            Mean of the second distribution.  
        lv2 : torch.Tensor  
            Log variance of the second distribution.  

        Returns  
        -------  
        torch.Tensor  
            KL divergence between the two distributions.  
        """  
        v1 = torch.exp(lv1)  
        v2 = torch.exp(lv2)  
        lstd1 = lv1 / 2.0  
        lstd2 = lv2 / 2.0  
        kl = lstd2 - lstd1 + (v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2) - 0.5  
        return kl  

    def _log_nb(  
        self,   
        x: torch.Tensor,   
        mu: torch.Tensor,   
        theta: torch.Tensor,   
        eps: float = 1e-8  
    ) -> torch.Tensor:  
        """  
        Compute the log likelihood of the negative binomial distribution.  

        Parameters  
        ----------  
        x : torch.Tensor  
            Observed data.  
        mu : torch.Tensor  
            Mean of the distribution.  
        theta : torch.Tensor  
            Dispersion parameter of the distribution.  
        eps : float, optional  
            Small constant to avoid numerical instability, by default 1e-8.  

        Returns  
        -------  
        torch.Tensor  
            Log likelihood of the negative binomial distribution.  
        """  
        log_theta_mu_eps = torch.log(theta + mu + eps)  
        res = (  
            theta * (torch.log(theta + eps) - log_theta_mu_eps)  
            + x * (torch.log(mu + eps) - log_theta_mu_eps)  
            + torch.lgamma(x + theta)  
            - torch.lgamma(theta)  
            - torch.lgamma(x + 1)  
        )  
        return res  


class adjMixin:  
    def _build_adj(  
        self,   
        edge_index: torch.Tensor,   
        num_nodes: int,   
        edge_weight: Optional[torch.Tensor] = None  
    ) -> torch.Tensor:  
        """  
        Build a sparse adjacency matrix.  

        Parameters  
        ----------  
        edge_index : torch.Tensor  
            Edge indices of the graph, shape (2, num_edges).  
        num_nodes : int  
            Number of nodes in the graph.  
        edge_weight : torch.Tensor, optional  
            Weights of the edges, by default None. If None, all edges are assigned a weight of 1.  

        Returns  
        -------  
        torch.Tensor  
            Sparse adjacency matrix in COO format.  
        """  
        if edge_weight is None:  
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)  
        adj = torch.sparse_coo_tensor(  
            edge_index,  
            edge_weight,  
            size=(num_nodes, num_nodes),  
            device=edge_index.device,  
        )  
        return adj  


class envMixin:  
    def _calc_score(  
        self,   
        latent: np.ndarray  
    ) -> Tuple[float, float, float, float, float, float]:  
        """  
        Calculate clustering and correlation scores for the latent space.  

        Parameters  
        ----------  
        latent : np.ndarray  
            Latent space representation of the data, shape (num_samples, num_features).  

        Returns  
        -------  
        Tuple[float, float, float, float, float, float]  
            A tuple containing the following scores:  
            - ARI: Adjusted Rand Index.  
            - NMI: Normalized Mutual Information.  
            - ASW: Average Silhouette Width.  
            - C_H: Calinski-Harabasz Index.  
            - D_B: Davies-Bouldin Index.  
            - P_C: Average pairwise correlation.  
        """  
        n = latent.shape[1]  
        labels = self._calc_label(latent)  
        scores = self._metrics(latent, labels)  
        return scores  

    def _calc_label(  
        self,   
        latent: np.ndarray  
    ) -> np.ndarray:  
        """  
        Perform KMeans clustering on the latent space.  

        Parameters  
        ----------  
        latent : np.ndarray  
            Latent space representation of the data, shape (num_samples, num_features).  

        Returns  
        -------  
        np.ndarray  
            Cluster labels for each sample.  
        """  
        labels = KMeans(latent.shape[1]).fit_predict(latent)  
        return labels  

    def _calc_corr(  
        self,   
        latent: np.ndarray  
    ) -> float:  
        """  
        Calculate the average absolute pairwise correlation in the latent space.  

        Parameters  
        ----------  
        latent : np.ndarray  
            Latent space representation of the data, shape (num_samples, num_features).  

        Returns  
        -------  
        float  
            Average absolute pairwise correlation.  
        """  
        acorr = abs(np.corrcoef(latent.T))  
        return acorr.sum(axis=1).mean().item() - 1  

    def _metrics(  
        self,   
        latent: np.ndarray,   
        labels: np.ndarray  
    ) -> Tuple[float, float, float, float, float, float]:  
        """  
        Compute clustering and correlation metrics.  

        Parameters  
        ----------  
        latent : np.ndarray  
            Latent space representation of the data, shape (num_samples, num_features).  
        labels : np.ndarray  
            Cluster labels for each sample.  

        Returns  
        -------  
        Tuple[float, float, float, float, float, float]  
            A tuple containing the following scores:  
            - ARI: Adjusted Rand Index.  
            - NMI: Normalized Mutual Information.  
            - ASW: Average Silhouette Width.  
            - C_H: Calinski-Harabasz Index.  
            - D_B: Davies-Bouldin Index.  
            - P_C: Average pairwise correlation.  
        """  
        ARI = adjusted_mutual_info_score(self.labels[self.idx], labels)  
        NMI = normalized_mutual_info_score(self.labels[self.idx], labels)  
        ASW = silhouette_score(latent, labels)  
        C_H = calinski_harabasz_score(latent, labels)  
        D_B = davies_bouldin_score(latent, labels)  
        P_C = self._calc_corr(latent)  
        return ARI, NMI, ASW, C_H, D_B, P_C

class scMixin:  
    def _preprocess(  
        self,  
        adata: AnnData,  
        layer: str,  
        n_var: int  
    ) -> None:  
        """Preprocess the AnnData object by normalizing, log-transforming, and selecting highly variable genes.  

        Parameters  
        ----------  
        adata : AnnData  
            Annotated data matrix (e.g., single-cell data).  
        layer : str  
            Layer of the AnnData object to store original features.  
        n_var : int  
            Number of highly variable genes to select.  
        """  
        try:  
            # Store original data in specified layer  
            if layer not in adata.layers.keys():  
                adata.layers[layer] = adata.X.copy()   
                print(f'Creating layer: {layer}.')  
            
            # Normalize and log-transform the data  
            if 'log1p' not in adata.uns.keys():   
                sc.pp.normalize_total(adata, target_sum=1e4)   
                sc.pp.log1p(adata)  
                print('Performing normalization.')  
            
            # Select highly variable genes  
            if 'highly_variable' not in adata.var.keys(): 
                if n_var:
                    sc.pp.highly_variable_genes(adata, n_top_genes=n_var)  
                else:
                    sc.pp.highly_variable_genes(adata)
                print('Selecting highly variable genes.')  
        
        except Exception as e:  
            print(f"Error during preprocessing: {e}")  

    def _decomposition(  
        self,  
        adata: AnnData,   
        tech: str,  
        latent_dim: int  
    ) -> None:  
        """Perform dimensionality reduction based on the selected method.  

        Parameters  
        ----------  
        adata : AnnData  
            Annotated data matrix (e.g., single-cell data).  
        tech : str  
            Decomposition method to use (PCA, NMF, FastICA, TruncatedSVD, FactorAnalysis, LatentDirichletAllocation).  
        latent_dim : int  
            Latent space dimension.  
        """  
        try:  
            # Perform dimensionality reduction based on the selected method  
            if tech == 'PCA':  
                latent = PCA(n_components=latent_dim).fit_transform(adata[:, adata.var['highly_variable']].X.toarray())  
            elif tech == 'NMF':  
                latent = NMF(n_components=latent_dim).fit_transform(adata[:, adata.var['highly_variable']].X.toarray())  
            elif tech == 'FastICA':  
                latent = FastICA(n_components=latent_dim).fit_transform(adata[:, adata.var['highly_variable']].X.toarray())  
            elif tech == 'TruncatedSVD':  
                latent = TruncatedSVD(n_components=latent_dim).fit_transform(adata[:, adata.var['highly_variable']].X.toarray())  
            elif tech == 'FactorAnalysis':  
                latent = FactorAnalysis(n_components=latent_dim).fit_transform(adata[:, adata.var['highly_variable']].X.toarray())  
            elif tech == 'LatentDirichletAllocation':  
                latent = LatentDirichletAllocation(n_components=latent_dim).fit_transform(adata[:, adata.var['highly_variable']].X.toarray())  
            else:  
                raise ValueError(f"Unsupported decomposition method: {tech}. Choose from PCA, NMF, FastICA, TruncatedSVD, FactorAnalysis, LatentDirichletAllocation.")  
            
            # Store the latent representation  
            adata.obsm[f'X_{tech}'] = latent  
            print(f'Stored latent representation in adata.obsm["X_{tech}"].')  

        except Exception as e:  
            print(f"Error during decomposition: {e}")  

    def _batchcorrect(  
        self,  
        adata: AnnData,  
        batch_tech: str,  
        tech: str,  
        layer: str  
    ) -> None:  
        """Correct batch effects using specified methods.  

        Parameters  
        ----------  
        adata : AnnData  
            Annotated data matrix (e.g., single-cell data).  
        batch_tech : str  
            Method to correct batch effects ('harmony' or 'scvi').  
        tech : str  
            Decomposition method used for batch correction.  
        layer : str  
            Layer of the AnnData object to use for scVI.  
        """  
        try:  
            # Batch effect correction if specified  
            if batch_tech == 'harmony':  
                import scanpy.external as sce  
                # Use Harmony integration  
                sce.pp.harmony_integrate(adata, key='batch', basis=f'X_{tech}', adjusted_basis=f'X_harmony_{tech}')  
                print('Applied Harmony integration for batch correction.')  
            
            elif batch_tech == 'scvi':  
                import scvi  
                # Use original X for scVI  
                scvi.model.SCVI.setup_anndata(adata, layer=layer)  
                model = scvi.model.SCVI(adata)
                model.train()  
                latent = model.get_latent_representation()  
                adata.obsm['X_scvi'] = latent   
                print('Applied scVI for batch correction.')  

        except Exception as e:  
            print(f"Error during batch correction: {e}")  

            