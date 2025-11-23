"""
Mixin Classes for GNODEVAE

This module provides mixin classes that add specific functionality to the
main agent and environment classes. Mixins follow the composition pattern
to separate concerns and promote code reuse.

Key Components:
- scviMixin: Single-cell variational inference utilities
- adjMixin: Adjacency matrix construction utilities  
- envMixin: Environment management utilities
- scMixin: Single-cell data preprocessing utilities
"""

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
    """
    Mixin providing variational inference utilities for single-cell data.
    
    This mixin implements common probability distributions and loss functions
    used in variational autoencoders for single-cell RNA-seq data, including:
    - KL divergence between normal distributions
    - Negative binomial log-likelihood for count data
    
    These methods are used to compute the VAE loss components.
    """
    
    def _normal_kl(  
        self,   
        mu1: torch.Tensor,   
        lv1: torch.Tensor,   
        mu2: torch.Tensor,   
        lv2: torch.Tensor  
    ) -> torch.Tensor:  
        """
        Compute KL divergence between two normal distributions.
        
        Calculates KL(N(mu1, sigma1^2) || N(mu2, sigma2^2)) analytically.
        This is used as a regularization term in the VAE loss to encourage
        the learned latent distribution to match a prior distribution
        (typically N(0, I)).
        
        Parameters  
        ----------  
        mu1 : torch.Tensor  
            Mean of the first distribution, shape (batch_size, latent_dim).
        lv1 : torch.Tensor  
            Log variance of the first distribution, shape (batch_size, latent_dim).
        mu2 : torch.Tensor  
            Mean of the second distribution (prior), shape (batch_size, latent_dim).
        lv2 : torch.Tensor  
            Log variance of the second distribution (prior).
            
        Returns  
        -------  
        torch.Tensor  
            KL divergence for each dimension, shape (batch_size, latent_dim).
            Sum over the latent dimension to get total KL per sample.
            
        Notes
        -----
        The KL divergence is computed as:
        KL = 0.5 * (log(σ2^2/σ1^2) + (σ1^2 + (μ1-μ2)^2)/σ2^2 - 1)
        
        For standard VAE with N(0,1) prior, this simplifies to:
        KL = 0.5 * (μ^2 + σ^2 - log(σ^2) - 1)
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
        Compute log-likelihood of negative binomial distribution.
        
        The negative binomial (NB) distribution is commonly used to model
        UMI count data in single-cell RNA-seq, as it can capture both the
        mean-variance relationship and overdispersion in count data.
        
        Parameters  
        ----------  
        x : torch.Tensor  
            Observed counts, shape (batch_size, num_genes).
        mu : torch.Tensor  
            Mean parameter of NB distribution, shape (batch_size, num_genes).
        theta : torch.Tensor  
            Dispersion parameter of NB distribution. Lower values indicate
            higher overdispersion. Shape (num_genes,) or (batch_size, num_genes).
        eps : float, default=1e-8
            Small constant for numerical stability in logarithms.
            
        Returns  
        -------  
        torch.Tensor  
            Log-likelihood for each gene count, shape (batch_size, num_genes).
            
        Notes
        -----
        The negative binomial distribution is parameterized as:
        NB(x | μ, θ) with mean μ and variance μ + μ^2/θ
        
        The log-likelihood is:
        log p(x|μ,θ) = log Γ(x+θ) - log Γ(θ) - log Γ(x+1) 
                       + x*log(μ) + θ*log(θ) - (x+θ)*log(θ+μ)
        
        This is used as the reconstruction loss for count data.
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
    """
    Mixin providing adjacency matrix construction utilities.
    
    This mixin provides methods for building adjacency matrices from edge
    representations, which is useful for loss computation and graph manipulation.
    """
    
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
        true_labels = self.labels[self.idx]
        ARI = adjusted_mutual_info_score(true_labels, labels)  
        NMI = normalized_mutual_info_score(true_labels, labels)  
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

            