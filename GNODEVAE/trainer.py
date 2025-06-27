from .mixin import scviMixin, adjMixin  
from .module import GraphVAE, iGraphVAE
from .module_refined import GraphVAE_r
import torch  
import torch.nn.functional as F  
from torch_geometric.data import Data  
import numpy as np
from typing import List, Tuple, Optional

class _BaseTrainer(scviMixin, adjMixin):
    """Base trainer class with shared logic."""
    def __init__(self, model, lr, beta, graph, device):
        self.gvae = model.to(device)
        self.opt = torch.optim.Adam(self.gvae.parameters(), lr=lr)
        self.beta = beta
        self.graph = graph
        self.loss = []
        self.device = device

    def take_latent(self, cd: Data) -> np.ndarray:
        """
        Extract the latent representation from the GraphVAE model.

        Parameters
        ----------
        cd : torch_geometric.data.Data
            Input graph data containing node features (`x`), edge indices (`edge_index`),
            and edge attributes (`edge_attr`).

        Returns
        -------
        np.ndarray
            Latent representation of the input data, shape (num_nodes, latent_dim).
        """
        states = cd.x
        edge_index = cd.edge_index
        edge_weight = cd.edge_attr
        res = self.gvae(states, edge_index, edge_weight)
        q_z = res[0]
        return q_z.detach().cpu().numpy()

    def _kl_loss(self, q_m, q_s):
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        return self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()

    def _adj_loss(self, adj, pred_a):
        return self.graph * F.binary_cross_entropy_with_logits(pred_a, adj)

    def _optimize(self, loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

class Trainer_r(_BaseTrainer):  
    """  
    Trainer class for training the GraphVAE_r model.  

    This class combines the functionality of `scviMixin` and `adjMixin` to train  
    a Graph Variational Autoencoder (GraphVAE) model. It handles the forward pass,  
    loss computation, and optimization steps.  

    Parameters  
    ----------  
    input_dim : int  
        Input feature dimension.  
    hidden_dim : int  
        Hidden layer dimension.  
    latent_dim : int  
        Latent space dimension.  
    encoder_type : str  
        Type of encoder ('linear' or 'graph')  
    graph_type : str, optional  
        Type of graph conv layer  
        ('GCN', 'Cheb', 'SAGE', 'TAG', 'ARMA', 'GAT', 'Transformer'), by default 'GAT'  
    structure_decoder_type : str, optional  
        Type of structure decoder ('bilinear', 'inner_product', 'mlp'), by default 'mlp'  
    feature_decoder_type : str, optional  
        Type of feature decoder ('linear' or 'graph'), by default 'linear'  
    hidden_layers : int, optional  
        Number of hidden layers, by default 2  
    decoder_hidden_dim : int, optional  
        Hidden dimension for structure MLP decoder, by default 128  
    dropout : float, optional  
        Dropout rate, by default 0.05  
    use_residual : bool, optional  
        Whether to use residual connections, by default True  
    Cheb_k : int, optional  
        Order of Chebyshev polynomials, by default 1  
    alpha : float, optional  
        Teleport probability, by default 0.5  
    sparse_threshold : int, optional  
        Maximum edges per node for structure decoder, by default None
    lr : float  
        Learning rate for the optimizer.  
    beta : float  
        Weight for the KL divergence term in the loss function.  
    graph : float  
        Weight for the graph reconstruction loss.  
    device : torch.device  
        Device to run the model on (e.g., 'cpu' or 'cuda').  

    Attributes  
    ----------  
    gvae : GraphVAE_r
        The GraphVAE_r model instance.  
    opt : torch.optim.Adam  
        Optimizer for training the model.  
    beta : float  
        Weight for the KL divergence term in the loss function.  
    graph : float  
        Weight for the graph reconstruction loss.  
    loss : List[Tuple[float, float, float]]  
        List of loss values (reconstruction loss, KL divergence, BCE loss) for each training step.  
    device : torch.device  
        Device to run the model on.  
    """  

    def __init__(  
        self,  
        input_dim: int,  
        hidden_dim: int,  
        latent_dim: int,  
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
        beta: float = 1,  
        graph: float = 1,  
        device: torch.device = torch.device('cuda'),  
    ):  
        
        model = GraphVAE_r(  
            input_dim,  
            hidden_dim,  
            latent_dim,  
            encoder_type,  
            graph_type,  
            structure_decoder_type,  
            feature_decoder_type,  
            hidden_layers,  
            decoder_hidden_dim,
            dropout,
            use_residual,
            Cheb_k,
            alpha,
            threshold,
            sparse_threshold
        )
        super().__init__(model, lr, beta, graph, device)

    def update(  
        self,   
        cd: Data
    ) -> None:  
        """  
        Perform a single training step for the GraphVAE model.  

        This method computes the reconstruction loss, KL divergence, and graph  
        reconstruction loss, and updates the model parameters using backpropagation.  

        Parameters  
        ----------  
        cd : torch_geometric.data.Data  
            Input graph data containing node features (`x`), edge indices (`edge_index`),  
            and edge attributes (`edge_attr`).  

        Returns  
        -------  
        None  
        """  
        states = cd.x  
        edge_index = cd.edge_index  
        edge_weight = cd.edge_attr  
        
        q_z, q_m, q_s, pred_a, pred_x = self.gvae(states, edge_index, edge_weight)  
        
        l = states.sum(-1).view(-1, 1)
        recon_loss = self._recon_loss(l, states, pred_x)  
        kl_loss = self._kl_loss(q_m, q_s)
        
        num_nodes = states.size(0)  
        adj = self._build_adj(edge_index, num_nodes, edge_weight).to_dense()
        adj_loss = self._adj_loss(adj, pred_a)

        loss = recon_loss + kl_loss + adj_loss
        self.loss.append((recon_loss.item(), kl_loss.item(), adj_loss.item()))
        self._optimize(loss)

    def _recon_loss(
        self,
        l,
        states,
        pred_x
    ):
        pred_x = pred_x * l  
        disp = torch.exp(self.gvae.feature_decoder.disp)  
        recon_loss = -self._log_nb(states, pred_x, disp).sum(-1).mean()
        return recon_loss

class Trainer(_BaseTrainer):  
    """  
    Trainer class for training the GraphVAE model.  

    This class combines the functionality of `scviMixin` and `adjMixin` to train  
    a Graph Variational Autoencoder (GraphVAE) model. It handles the forward pass,  
    loss computation, and optimization steps.  

    Parameters  
    ----------  
    interpretable : bool 
        Use the iVGAE model.
    input_dim : int  
        Dimensionality of the input features.  
    hidden_dim : int  
        Dimensionality of the hidden layers in the encoder and decoder.  
    latent_dim : int  
        Dimensionality of the latent space.
    idim : int, optional
        Dimensionality of the interpreted space.
    encoder_type : str  
        Type of encoder to use.  
    encoder_hidden_layers : int  
        Number of hidden layers in the encoder.  
    decoder_type : str  
        Type of decoder to use.  
    decoder_hidden_dim : int  
        Dimensionality of the hidden layers in the decoder.  
    feature_decoder_hidden_layers : int  
        Number of hidden layers in the feature decoder.  
    dropout : float  
        Dropout rate for the model. 
    use_residual : bool, optional  
        Whether to use residual connections, by default True.
    Cheb_k : int, optional  
        The order of Chebyshev polynomials for ChebConv, by default None.
    alpha : float, optional
        Teleport probability, by default 0.5.
    lr : float  
        Learning rate for the optimizer.  
    beta : float  
        Weight for the KL divergence term in the loss function.  
    graph : float  
        Weight for the graph reconstruction loss.  
    device : torch.device  
        Device to run the model on (e.g., 'cpu' or 'cuda').  

    Attributes  
    ----------  
    gvae : GraphVAE  
        The GraphVAE model instance.  
    opt : torch.optim.Adam  
        Optimizer for training the model.  
    beta : float  
        Weight for the KL divergence term in the loss function.  
    graph : float  
        Weight for the graph reconstruction loss.  
    loss : List[Tuple[float, float, float]]  
        List of loss values (reconstruction loss, KL divergence, BCE loss) for each training step.  
    device : torch.device  
        Device to run the model on.  
    """  

    def __init__(  
        self,  
        interpretable: bool,
        input_dim: int,  
        hidden_dim: int,  
        latent_dim: int,  
        idim: Optional[int],
        encoder_type: str,  
        encoder_hidden_layers: int,  
        decoder_type: str,  
        decoder_hidden_dim: int,  
        feature_decoder_hidden_layers: int,  
        dropout: float,
        use_residual: bool,
        Cheb_k: Optional[int],
        alpha: Optional[float],
        lr: float,  
        beta: float,  
        graph: float,  
        device: torch.device,  
    ):  
        self.interpretable = interpretable
        if interpretable:
            model = iGraphVAE(
                input_dim,  
                hidden_dim,  
                latent_dim,  
                idim,
                encoder_type,  
                encoder_hidden_layers,  
                decoder_type,  
                decoder_hidden_dim,  
                feature_decoder_hidden_layers,  
                dropout,
                use_residual,
                Cheb_k,
                alpha
            )
        else:
            model = GraphVAE(  
                input_dim,  
                hidden_dim,  
                latent_dim,  
                encoder_type,  
                encoder_hidden_layers,  
                decoder_type,  
                decoder_hidden_dim,  
                feature_decoder_hidden_layers,  
                dropout,
                use_residual,
                Cheb_k,
                alpha
            )
        super().__init__(model, lr, beta, graph, device)

    def update(  
        self,   
        cd: Data
    ) -> None:  
        """  
        Perform a single training step for the GraphVAE model.  

        This method computes the reconstruction loss, KL divergence, and graph  
        reconstruction loss, and updates the model parameters using backpropagation.  

        Parameters  
        ----------  
        cd : torch_geometric.data.Data  
            Input graph data containing node features (`x`), edge indices (`edge_index`),  
            and edge attributes (`edge_attr`).  

        Returns  
        -------  
        None  
        """  
        states = cd.x  
        edge_index = cd.edge_index  
        edge_weight = cd.edge_attr  
        
        if self.interpretable:
            q_z, q_m, q_s, pred_a, pred_x, i_q_z, pred_ia, pred_ix = self.gvae(states, edge_index, edge_weight)
            l = states.sum(-1).view(-1, 1)
            recon_loss = self._recon_loss(l, states, pred_x)
            irecon_loss = self._recon_loss(l, states, pred_ix)
            
            kl_loss = self._kl_loss(q_m, q_s)

            num_nodes = states.size(0)  
            adj = self._build_adj(edge_index, num_nodes, edge_weight).to_dense()
            adj_loss = self._adj_loss(adj, pred_a)
            iadj_loss = self._adj_loss(adj, pred_ia)

            loss = recon_loss + kl_loss + adj_loss + irecon_loss  + iadj_loss
            self.loss.append((recon_loss.item(),
                              kl_loss.item(),
                              adj_loss.item(),
                              irecon_loss.item(),
                              iadj_loss.item()
                             ))
            
        else:
            q_z, q_m, q_s, pred_a, pred_x = self.gvae(states, edge_index, edge_weight)  
            l = states.sum(-1).view(-1, 1)
            recon_loss = self._recon_loss(l, states, pred_x)  
            kl_loss = self._kl_loss(q_m, q_s)
            
            num_nodes = states.size(0)  
            adj = self._build_adj(edge_index, num_nodes, edge_weight).to_dense()
            adj_loss = self._adj_loss(adj, pred_a)

            loss = recon_loss + kl_loss + adj_loss
            self.loss.append((recon_loss.item(), kl_loss.item(), adj_loss.item()))
        
        self._optimize(loss)

    def _recon_loss(
        self,
        l,
        states,
        pred_x
    ):
        pred_x = pred_x * l  
        disp = torch.exp(self.gvae.x_decoder.disp)  
        recon_loss = -self._log_nb(states, pred_x, disp).sum(-1).mean()
        return recon_loss