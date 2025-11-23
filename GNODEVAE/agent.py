"""
Training Agents for GNODEVAE

This module implements agent classes that manage the training process for
GNODEVAE models. Agents combine environment setup, model training, and
evaluation metrics to provide a high-level interface for users.

Key Classes:
- BaseAgent: Base class with common training functionality
- agent: Standard GraphVAE agent
- agent_r: GNODEVAE agent with ODE component
"""

from .env import Env, Env_r
from anndata import AnnData
import numpy as np
import torch
import tqdm
from typing import Optional, Self
import time
import psutil

class BaseAgent:
    """
    Base class for training agents.
    
    This class provides common functionality for training, including:
    - Progress tracking with tqdm
    - Resource monitoring (CPU/GPU memory)
    - Score computation and storage
    - Latent representation extraction
    
    The BaseAgent uses multiple inheritance to combine environment setup
    (Env/Env_r) with training logic.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the base agent.
        
        This __init__ will be called by agent and agent_r, which will then
        call their respective Env/Env_r initialization methods.
        """
        super().__init__(*args, **kwargs)

    def fit(
        self,
        epochs: int = 300,
        update_steps: int = 10,
        silent: bool = False,
    ) -> Self:
        """
        Train the model for a specified number of epochs.
        
        This method iterates through epochs, performs training steps,
        monitors resources, and tracks evaluation metrics. Progress is
        displayed using tqdm with periodic updates of loss and scores.
        
        Parameters
        ----------
        epochs : int, default=300
            Number of training epochs to run.
        update_steps : int, default=10
            Frequency of progress bar updates (every N epochs).
        silent : bool, default=False
            If True, suppress progress bar output.
            
        Returns
        -------
        Self
            Returns self for method chaining.
            
        Attributes Set
        --------------
        resource : list
            List of tuples containing (step_time, cpu_memory, gpu_memory)
            for each epoch.
        time_all : float
            Total training time in seconds.
            
        Notes
        -----
        The method computes and displays the following metrics:
        - Loss: Total reconstruction and KL divergence loss
        - ARI: Adjusted Rand Index (clustering agreement)
        - NMI: Normalized Mutual Information
        - ASW: Average Silhouette Width (cluster separation)
        - C_H: Calinski-Harabasz Index (cluster quality)
        - D_B: Davies-Bouldin Index (lower is better)
        - P_C: Pearson Correlation
        - Step Time: Time per training step
        """
        self.resource = []
        start_time = time.time()

        try:
            with tqdm.tqdm(total=epochs, desc='Fitting', ncols=150, disable=silent, miniters=update_steps) as pbar:
                for i in range(epochs):
                    step_start_time = time.time()

                    # Perform one training step (forward + backward + optimize)
                    self.step()

                    step_end_time = time.time()
                    step_time = step_end_time - step_start_time

                    # Monitor resource usage
                    process = psutil.Process()
                    cpu_memory = process.memory_info().rss / (1024 ** 2)  # MB

                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
                    else:
                        gpu_memory = 0.0

                    self.resource.append((step_time, cpu_memory, gpu_memory))

                    # Update progress bar periodically
                    if (i + 1) % update_steps == 0:
                        # Compute average metrics over recent steps
                        recent_losses = self.loss[-update_steps:] if len(self.loss) >= update_steps else self.loss
                        recent_scores = self.score[-update_steps:] if len(self.score) >= update_steps else self.score
                        recent_resources = self.resource[-update_steps:] if len(self.resource) >= update_steps else self.resource

                        loss = np.mean([sum(loss_step) for loss_step in recent_losses])
                        ari, nmi, asw, ch, db, pc = np.mean(recent_scores, axis=0)
                        st, cm, gm = np.mean(recent_resources, axis=0)

                        pbar.set_postfix({
                            'Loss': f'{loss:.2f}',
                            'ARI': f'{ari:.2f}',
                            'NMI': f'{nmi:.2f}',
                            'ASW': f'{asw:.2f}',
                            'C_H': f'{ch:.2f}',
                            'D_B': f'{db:.2f}',
                            'P_C': f'{pc:.2f}',
                            'Step Time': f'{st:.2f}s',
                        }, refresh=False)
                    pbar.update(1)

        except Exception as e:
            print(f"{e}")
            raise e

        end_time = time.time()
        self.time_all = end_time - start_time
        return self

    def _get_latent_representation(self) -> np.ndarray:
        """
        Extract latent representations from all data partitions.
        
        This internal method collects latent embeddings from each graph
        partition and concatenates them into a single array.
        
        Returns
        -------
        np.ndarray
            Concatenated latent representations, shape (num_cells, latent_dim).
        """
        ls_l = []
        for cd in self.cdata:
            latent = self.take_latent(cd)
            ls_l.append(latent)
        latent = np.vstack(ls_l)
        return latent

    def get_latent(self) -> np.ndarray:
        """
        Get latent representations in original cell order.
        
        Since cells may be reordered during graph partitioning, this method
        restores the original ordering using stored indices.
        
        Returns
        -------
        np.ndarray
            Latent representations in original cell order,
            shape (num_cells, latent_dim).
            
        Examples
        --------
        >>> agent = agent_r(adata=adata, ...)
        >>> agent.fit(epochs=100)
        >>> latent = agent.get_latent()
        >>> adata.obsm['X_gnodevae'] = latent
        """
        latent = self._get_latent_representation()
        # Restore original cell ordering
        lut = dict(zip(self.idx, latent))
        latent_ordered = np.vstack([lut[i] for i in range(self.n_obs)])
        return latent_ordered

    def score_final(self) -> None:
        """
        Compute final clustering scores after training.
        
        This method evaluates the quality of the learned latent representation
        by computing various clustering metrics. Results are stored in the
        `final_score` attribute.
        
        Sets
        ----
        final_score : tuple
            Tuple of (ARI, NMI, ASW, CH, DB, PC) scores.
        """
        latent = self._get_latent_representation()
        score = self._calc_score(latent)
        self.final_score = score


class agent_r(BaseAgent, Env_r):
    """
    GNODEVAE agent with Neural ODE component.
    
    This agent class provides a high-level interface for training GNODEVAE
    models that include the Neural ODE component for trajectory inference.
    It combines environment setup (data preprocessing, graph construction)
    with the ODE-augmented VAE model.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing single-cell gene expression.
    layer : str, default='counts'
        Layer of AnnData to use for input features.
    n_var : Optional[int], default=None
        Number of highly variable genes to select. If None, uses all genes.
    tech : str, default='PCA'
        Dimensionality reduction technique ('PCA', 'NMF', 'FastICA', etc.).
    n_neighbors : int, default=15
        Number of neighbors for k-NN graph construction.
    batch_tech : Optional[str], default=None
        Batch correction method ('harmony' or 'scvi').
    all_feat : bool, default=True
        Whether to use all features or only highly variable genes.
    hidden_dim : int, default=128
        Hidden layer dimension for neural networks.
    latent_dim : int, default=10
        Latent space dimension for cell embeddings.
    encoder_type : str, default='graph'
        Type of encoder ('graph' or 'linear').
    graph_type : str, default='GAT'
        Graph convolution type ('GAT', 'GCN', 'SAGE', etc.).
    structure_decoder_type : str, default='mlp'
        Structure decoder type ('mlp', 'bilinear', 'inner_product').
    feature_decoder_type : str, default='linear'
        Feature decoder type ('linear' or 'graph').
    hidden_layers : int, default=2
        Number of hidden layers.
    decoder_hidden_dim : int, default=128
        Hidden dimension for structure decoder.
    dropout : float, default=0.05
        Dropout rate for regularization.
    use_residual : bool, default=True
        Whether to use residual connections.
    Cheb_k : int, default=1
        Order of Chebyshev polynomials.
    alpha : float, default=0.5
        Teleport probability for SSGConv.
    threshold : float, default=0
        Threshold for edge probability.
    sparse_threshold : Optional[int], default=None
        Maximum number of edges per node.
    lr : float, default=1e-4
        Learning rate for optimizer.
    beta : float, default=1.0
        Weight for KL divergence loss.
    graph : float, default=1.0
        Weight for graph reconstruction loss.
    device : torch.device, optional
        Computing device (auto-detected if not specified).
    num_parts : int, default=10
        Number of graph partitions for mini-batch training.
    *args, **kwargs
        Additional arguments passed to parent classes.
        
    Examples
    --------
    >>> import scanpy as sc
    >>> from GNODEVAE import agent_r
    >>> adata = sc.read_h5ad('data.h5ad')
    >>> model = agent_r(adata=adata, latent_dim=10)
    >>> model.fit(epochs=300)
    >>> latent = model.get_latent()
    """
    
    def __init__(
        self,
        adata: AnnData,
        layer: str = 'counts',
        n_var: Optional[int] = None,
        tech: str = 'PCA',
        n_neighbors: int = 15,
        batch_tech: Optional[str] = None,
        all_feat: bool = True,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        encoder_type: str = 'graph',
        graph_type: str = 'GAT',
        structure_decoder_type: str = 'mlp',
        feature_decoder_type: str = 'linear',
        hidden_layers: int = 2,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.05,
        use_residual: bool = True,
        Cheb_k: int = 1,
        alpha: float = 0.5,
        threshold: float = 0,
        sparse_threshold: Optional[int] = None,
        lr: float = 1e-4,
        beta: float = 1.0,
        graph: float = 1.0,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        num_parts: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(
            adata=adata,
            layer=layer,
            n_var=n_var,
            tech=tech,
            n_neighbors=n_neighbors,
            batch_tech=batch_tech,
            all_feat=all_feat,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            encoder_type=encoder_type,
            graph_type=graph_type,
            structure_decoder_type=structure_decoder_type,
            feature_decoder_type=feature_decoder_type,
            hidden_layers=hidden_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
            use_residual=use_residual,
            Cheb_k=Cheb_k,
            alpha=alpha,
            threshold=threshold,
            sparse_threshold=sparse_threshold,
            lr=lr,
            beta=beta,
            graph=graph,
            device=device,
            num_parts=num_parts,
            *args,
            **kwargs,
        )


class agent(BaseAgent, Env):
    """
    Standard GraphVAE agent (without ODE component).
    
    This agent class provides a high-level interface for training standard
    GraphVAE models without the Neural ODE component. It's suitable for
    static clustering tasks where temporal dynamics are not needed.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (e.g., single-cell data).
    layer : str, optional
        Layer of the AnnData object to use for input features, by default 'counts'.
    n_var : int
        Number of highly variable genes to select.
    tech : str
        Decomposition method to use (PCA, NMF, FastICA, TruncatedSVD, FactorAnalysis, LatentDirichletAllocation).
    n_neighbors : int
        Number of neighbors for graph construction.
    latent_dim : int
        Latent space dimension for clustering.
    batch_tech : Optional[str]
        Method to correct batch effects ('harmony' or 'scvi').
    all_feat : bool
        Whether to use all features or only highly variable ones
    hidden_dim : int, optional
        Hidden layer dimension for the encoder, by default 128.
    latent_dim : int, optional
        Latent space dimension, by default 10.
    encoder_type : str, optional
        Type of graph convolutional layer ('GCN', 'Cheb', 'SAGE', 'Graph', 'TAG', 'ARMA', 'GAT', 'Transformer', 'SG', 'SSG'), by default 'GAT'.
    encoder_hidden_layers : int, optional
        Number of hidden layers in the graph encoder, by default 2.
    decoder_type : str, optional
        Type of graph decoder ('Bilinear', 'InnerProduct', 'MLP'), by default 'MLP'.
    decoder_hidden_dim : int, optional
        Hidden dimension for the MLPDecoder (if used), by default 128.
    feature_decoder_hidden_layers : int, optional
        Number of hidden layers in the feature decoder, by default 2.
    dropout : float, optional
        Dropout rate, by default 5e-3.
    use_residual : bool, optional
        Whether to use residual connections, by default True.
    Cheb_k : int, optional
        The order of Chebyshev polynomials for ChebConv, by default None.
    alpha : float, optional
        Teleport probability, by default 0.5.
    lr : float, optional
        Learning rate for the optimizer, by default 1e-4.
    beta : float, optional
        Weight for the KL divergence term in the loss function, by default 1.0.
    graph : float, optional
        Weight for the graph reconstruction loss, by default 1.0.
    device : torch.device, optional
        Device to run the model on (e.g., 'cpu' or 'cuda'), by default uses GPU if available.
    num_parts : int, optional
        Number of partitions for clustering the graph data, by default 10.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    def __init__(
        self,
        adata: AnnData,
        layer: str = 'counts',
        n_var: Optional[int] = None,
        tech: str = 'PCA',
        n_neighbors: int = 15,
        batch_tech: Optional[str] = None,
        all_feat: bool = True,
        interpretable: bool = False,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        idim: Optional[int] = 2,
        encoder_type: str = 'GAT',
        encoder_hidden_layers: int = 2,
        decoder_type: str = 'MLP',
        decoder_hidden_dim: int = 128,
        feature_decoder_hidden_layers: int = 2,
        dropout: float = 5e-3,
        use_residual: bool = True,
        Cheb_k: Optional[int] = 1,
        alpha: Optional[float] = .5,
        lr: float = 1e-4,
        beta: float = 1.0,
        graph: float = 1.0,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        num_parts: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(
            adata=adata,
            layer=layer,
            n_var=n_var,
            tech=tech,
            n_neighbors=n_neighbors,
            batch_tech=batch_tech,
            all_feat=all_feat,
            interpretable=interpretable,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            idim=idim,
            encoder_type=encoder_type,
            encoder_hidden_layers=encoder_hidden_layers,
            decoder_type=decoder_type,
            decoder_hidden_dim=decoder_hidden_dim,
            feature_decoder_hidden_layers=feature_decoder_hidden_layers,
            dropout=dropout,
            use_residual=use_residual,
            Cheb_k=Cheb_k,
            alpha=alpha,
            lr=lr,
            beta=beta,
            graph=graph,
            device=device,
            num_parts=num_parts,
            *args,
            **kwargs,
        )