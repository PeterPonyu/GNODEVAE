from .mixin import scviMixin, adjMixin  
from .GODEVAE import GODEVAE_r  
from .utils_ODE import get_step_size  
import torch  
import torch.nn.functional as F  
from torch_geometric.data import Data  
import numpy as np  
from typing import List, Tuple, Optional  
from torchdiffeq import odeint  


class GODEVAE_Trainer_r(scviMixin, adjMixin):  
    """  
    Trainer class for training the GODEVAE_r model.  

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
    ode_hidden_dim : int  
        Hidden dimension for the ODE function.  
    encoder_type : str  
        Type of encoder ('linear' or 'graph').  
    graph_type : str, optional  
        Type of graph conv layer ('GCN', 'Cheb', 'SAGE', 'TAG', 'ARMA', 'GAT', 'Transformer'), by default 'GAT'.  
    structure_decoder_type : str, optional  
        Type of structure decoder ('bilinear', 'inner_product', 'mlp'), by default 'mlp'.  
    feature_decoder_type : str, optional  
        Type of feature decoder ('linear' or 'graph'), by default 'linear'.  
    hidden_layers : int, optional  
        Number of hidden layers, by default 2.  
    decoder_hidden_dim : int, optional  
        Hidden dimension for structure MLP decoder, by default 128.  
    dropout : float, optional  
        Dropout rate, by default 0.05.  
    use_residual : bool, optional  
        Whether to use residual connections, by default True.  
    Cheb_k : int, optional  
        Order of Chebyshev polynomials, by default 1.  
    alpha : float, optional  
        Teleport probability, by default 0.5.  
    threshold : float, optional  
        Threshold for structure decoder, by default 0.  
    sparse_threshold : int, optional  
        Maximum edges per node for structure decoder, by default None.  
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
    godevae : GODEVAE_r  
        The GODEVAE_r model instance.  
    opt : torch.optim.Adam  
        Optimizer for training the model.  
    beta : float  
        Weight for the KL divergence term in the loss function.  
    graph : float  
        Weight for the graph reconstruction loss.  
    loss : List[Tuple[float, float, float, float, float]]  
        List of loss values (reconstruction loss, KL divergence, BCE loss, ODE reconstruction loss, latent divergence) for each training step.  
    device : torch.device  
        Device to run the model on.  
    """  

    def __init__(  
        self,  
        input_dim: int,  
        hidden_dim: int,  
        latent_dim: int,  
        ode_hidden_dim: int,  
        encoder_type: str = "graph",  
        graph_type: str = "GAT",  
        structure_decoder_type: str = "mlp",  
        feature_decoder_type: str = "linear",  
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
        w_recon: float = 1.0,
        w_kl: float = 1.0,
        w_adj: float = 1.0,
        w_recon_ode: float = 1.0,
        w_z_div: float = 1.0,
        device: torch.device = torch.device("cuda"),  
        latent_type: str = 'q_m',
    ):  
        self.godevae = GODEVAE_r(  
            input_dim,  
            hidden_dim,  
            latent_dim,  
            ode_hidden_dim,  
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
            sparse_threshold,  
        ).to(device)  

        self.opt = torch.optim.Adam(self.godevae.parameters(), lr=lr)  
        self.beta = beta  
        self.graph = graph  
        self.w_recon = w_recon
        self.w_kl = w_kl
        self.w_adj = w_adj
        self.w_recon_ode = w_recon_ode
        self.w_z_div = w_z_div
        self.loss: List[Tuple[float, float, float, float, float]] = []  
        self.device = device  
        self.latent_type = latent_type

    @torch.no_grad()  
    def take_latent(self, cd: Data) -> np.ndarray:  
        """  
        Extract latent variables from the encoder.  

        Parameters
        ----------
        cd : Data
            Input data.
        """  
        states = cd.x  
        edge_index = cd.edge_index  
        edge_weight = cd.edge_attr  
        if self.godevae.encoder_type == 'linear':  
            q_z, q_m, _, t = self.godevae.encoder(states)
        else:    
            q_z, q_m, _, t = self.godevae.encoder(states, edge_index, edge_weight)
        if self.latent_type == 'q_m':
            return q_m.cpu().numpy()
        elif self.latent_type == 'q_z':
            return q_z.cpu().numpy()
        else:
            raise ValueError("latent_type must be 'q_m' or 'q_z'")

    @torch.no_grad()  
    def take_odelatent(  
        self,  
        cd: Data,  
        step_size: Optional[int] = None,  
        step_wise: bool = False,  
        batch_size: Optional[int] = None,
    ) -> np.ndarray:  
        """  
        Extract ODE latent variables by solving the latent ODE.  

        Parameters
        ----------
        cd : Data
            Input data.
        step_size : int, optional
            Step size for the ODE solver, by default None.
        step_wise : bool, optional
            Whether to solve the ODE step-wise, by default False.
        batch_size : int, optional
            Batch size for processing, by default None.
        """  
        states = cd.x  
        edge_index = cd.edge_index  
        edge_weight = cd.edge_attr  
        
        if self.godevae.encoder_type == 'linear':  
            q_z, q_m, _, t = self.godevae.encoder(states)
        else:    
            q_z, q_m, _, t = self.godevae.encoder(states, edge_index, edge_weight) 

        if self.latent_type == 'q_m':
            latent = q_m
        elif self.latent_type == 'q_z':
            latent = q_z
        else:
            raise ValueError("latent_type must be 'q_m' or 'q_z'")
                    
        sort_t, sort_idx, sort_ridx = np.unique(t.cpu(), return_index=True, return_inverse=True)  
        sort_t = torch.tensor(sort_t).to(self.device)  
        q_z_sort = latent[sort_idx]  

        q_z_ode = []  
        if batch_size is None:  
            batch_size = len(sort_t)  

        for i in range(0, len(sort_t), batch_size):  
            t_batch = sort_t[i : i + batch_size]  
            z_batch = q_z_sort[i : i + batch_size]  
            z0 = z_batch[0]  

            if not step_wise:  
                options = get_step_size(step_size, t_batch[0], t_batch[-1], len(t_batch))  
                pred_z = odeint(  
                    self.godevae.lode_func, z0, t_batch, method="euler", options=options  
                )  
            else:  
                pred_z = torch.empty((len(t_batch), z_batch.size(1)), device=self.device)  
                pred_z[0] = z0  
                for j in range(len(t_batch) - 1):  
                    t_segment = t_batch[j : j + 2]  
                    options = get_step_size(step_size, t_segment[0], t_segment[-1], len(t_segment))  
                    pred_z[j + 1] = odeint(  
                        self.godevae.lode_func, z_batch[j], t_segment, method="euler", options=options  
                    )[1]  

            q_z_ode.append(pred_z)  

        q_z_ode = torch.cat(q_z_ode)  
        q_z_ode = q_z_ode[sort_ridx.ravel()]  
        return q_z_ode.cpu().numpy()  

    @torch.no_grad()  
    def take_time(self, cd: Data) -> np.ndarray:  
        """  
        Extract time variables from the encoder.  
        """  
        states = cd.x  
        edge_index = cd.edge_index  
        edge_weight = cd.edge_attr  
        if self.godevae.encoder_type == 'linear':  
            ts = self.godevae.encoder(states)[-1]
        else:    
            ts = self.godevae.encoder(states, edge_index, edge_weight)[-1]  
        return ts.cpu().numpy().ravel()  

    def update(self, cd: Data) -> None:  
        """  
        Perform a single training step for the GODEVAE model.  
        """  
        states = cd.x  
        edge_index = cd.edge_index  
        edge_weight = cd.edge_attr  

        # Forward pass  
        q_z, q_m, q_s, pred_a, pred_x, q_z_ode, pred_x_ode = self.godevae(states, edge_index, edge_weight)  

        # Compute losses  
        l = states.sum(-1).view(-1, 1)  
        recon_loss = self._recon_loss(l, states, pred_x)  
        kl_loss = self._kl_loss(q_m, q_s)  

        num_nodes = states.size(0)  
        adj = self._build_adj(edge_index, num_nodes, edge_weight).to_dense()  
        adj_loss = self._adj_loss(adj, pred_a)  
        
        idx1, idx2 = self.godevae.idx1, self.godevae.idx2  
        
        recon_loss_ode = self._recon_loss(l[idx1][idx2], states[idx1][idx2], pred_x_ode)  
        z_div = F.mse_loss(q_m[idx1][idx2], q_z_ode, reduction="none").sum(-1).mean()  

        # Total loss  
        loss = (self.w_recon * recon_loss + 
                self.w_kl * kl_loss + 
                self.w_adj * adj_loss + 
                self.w_recon_ode * recon_loss_ode + 
                self.w_z_div * z_div)  

        # Store loss values  
        self.loss.append((recon_loss.item(), kl_loss.item(), adj_loss.item(), recon_loss_ode.item(), z_div.item()))  

        # Backpropagation  
        self.opt.zero_grad()  
        loss.backward()  
        self.opt.step()  

    def _recon_loss(self, l, states, pred_x):  
        """  
        Compute reconstruction loss for node features.  
        """  
        pred_x = pred_x * l  
        disp = torch.exp(self.godevae.feature_decoder.disp)  
        return -self._log_nb(states, pred_x, disp).sum(-1).mean()  

    def _kl_loss(self, q_m, q_s):  
        """  
        Compute KL divergence for the latent space.  
        """  
        p_m = torch.zeros_like(q_m)  
        p_s = torch.zeros_like(q_s)  
        return self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()  

    def _adj_loss(self, adj, pred_a):  
        """  
        Compute graph reconstruction loss.  
        """  
        return self.graph * F.binary_cross_entropy_with_logits(pred_a, adj)