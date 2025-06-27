from .env import Env, Env_r
from anndata import AnnData
import numpy as np
import torch
import tqdm
from typing import Optional, Self
import time
import psutil

class BaseAgent:
    def __init__(self, *args, **kwargs):
        # This __init__ will be called by agent and agent_r, which will then call their respective Env/Env_r inits
        super().__init__(*args, **kwargs)

    def fit(
        self,
        epochs: int = 300,
        update_steps: int = 10,
        silent: bool = False,
    ) -> Self:
        self.resource = []
        start_time = time.time()

        try:
            with tqdm.tqdm(total=epochs, desc='Fitting', ncols=200, disable=silent) as pbar:
                for i in range(epochs):
                    step_start_time = time.time()

                    self.step()

                    step_end_time = time.time()
                    step_time = step_end_time - step_start_time

                    process = psutil.Process()
                    cpu_memory = process.memory_info().rss / (1024 ** 2)

                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    else:
                        gpu_memory = 0.0

                    self.resource.append((step_time, cpu_memory, gpu_memory))

                    if (i + 1) % update_steps == 0:
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
                            'CPU Mem': f'{cm:.0f}MB',
                            'GPU Mem': f'{gm:.0f}MB',
                        }, refresh=False)
                    pbar.update(1)

        except Exception as e:
            print(f"{e}")
            raise e

        end_time = time.time()
        self.time_all = end_time - start_time
        return self

    def _get_latent_representation(self) -> np.ndarray:
        ls_l = []
        for cd in self.cdata:
            latent = self.take_latent(cd)
            ls_l.append(latent)
        latent = np.vstack(ls_l)
        # Assuming self.idx and self.n_obs are available from Env/Env_r
        lut = dict(zip(self.idx, latent))
        latent_ordered = np.vstack([lut[i] for i in range(self.n_obs)])
        return latent_ordered

    def get_latent(self) -> np.ndarray:
        return self._get_latent_representation()

    def score_final(self) -> None:
        latent = self._get_latent_representation()
        score = self._calc_score(latent)
        self.final_score = score


class agent_r(BaseAgent, Env_r):
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
    Agent class for training and evaluating the GraphVAE model.

    This class extends the `Env` class and provides additional functionality
    for fitting the model and extracting latent representations.

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