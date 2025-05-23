from .module_refined import BaseGraphNetwork, BaseLinearModel, GraphStructureDecoder, LinearDecoder  
from .utils_ODE import get_step_size, LatentODEfunc  
from torchdiffeq import odeint  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.distributions import Normal  
from typing import Optional, Tuple  


class GraphEncoder_t(BaseGraphNetwork):  
    def __init__(  
        self,  
        input_dim: int,  
        hidden_dim: int,  
        latent_dim: int,  
        conv_layer_type: str = 'GAT',  
        hidden_layers: int = 2,  
        dropout: float = 0.05,  
        Cheb_k: int = 1,  
        alpha: float = 0.5,  
    ):  
        """  
        Graph-based encoder with an additional layer to generate time variable `t`.  
        """  
        super().__init__(input_dim, hidden_dim, latent_dim, conv_layer_type, hidden_layers, dropout, Cheb_k, alpha)  
        self.fc_t = nn.Linear(hidden_dim, 1)  # Fully connected layer to generate `t`  
        self.apply(self._init_weights)  # Initialize weights  

    def _build_output_layer(self, hidden_dim: int, latent_dim: int, Cheb_k: int, alpha: float) -> None:  
        self.conv_mean = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)  
        self.conv_logvar = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)  

    def forward(  
        self,  
        x: torch.Tensor,  
        edge_index: torch.Tensor,  
        edge_weight: Optional[torch.Tensor] = None,  
        use_residual: bool = True,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        Forward pass for the graph encoder.  
        """  
        residual = None  
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):  
            x = self._process_layer(x, conv, edge_index, edge_weight)  
            x = bn(x)  
            x = self.relu(x)  
            x = dropout(x)  
            if use_residual and i == 0:  
                residual = x  
        if use_residual and residual is not None:  
            x = x + residual  

        # Generate time variable `t` using sigmoid activation  
        t = torch.sigmoid(self.fc_t(x))  

        # Compute mean and log variance for latent space  
        q_m = self._process_layer(x, self.conv_mean, edge_index, edge_weight)  
        q_s = self._process_layer(x, self.conv_logvar, edge_index, edge_weight)  

        # Reparameterization trick  
        std = F.softplus(q_s) + 1e-6  
        dist = Normal(q_m, std)  
        q_z = dist.rsample()  

        return q_z, q_m, q_s, t  


class LinearEncoder_t(BaseLinearModel):  
    def __init__(  
        self,  
        input_dim: int,  
        hidden_dim: int,  
        latent_dim: int,  
        hidden_layers: int = 2,  
        dropout: float = 0.0,  
    ):  
        """  
        Linear encoder with an additional layer to generate time variable `t`.  
        """  
        super().__init__(input_dim, hidden_dim, hidden_dim, hidden_layers, dropout)  
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)  
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)  
        self.fc_t = nn.Linear(hidden_dim, 1)  # Fully connected layer to generate `t`  

        # Initialize weights  
        nn.init.xavier_uniform_(self.mu_layer.weight)  
        nn.init.zeros_(self.mu_layer.bias)  
        nn.init.xavier_uniform_(self.logvar_layer.weight)  
        nn.init.zeros_(self.logvar_layer.bias)  
        nn.init.xavier_uniform_(self.fc_t.weight)  
        nn.init.zeros_(self.fc_t.bias)  

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        Forward pass for the linear encoder.  
        """  
        h = self.network(x)  
        t = torch.sigmoid(self.fc_t(h))  # Generate time variable `t`  

        q_m = self.mu_layer(h)  
        q_s = self.logvar_layer(h)  
        std = F.softplus(q_s) + 1e-6  
        dist = Normal(q_m, std)  
        q_z = dist.rsample()  

        return q_z, q_m, q_s, t  


class GODEVAE_r(nn.Module):  
    def __init__(  
        self,  
        input_dim: int,  
        hidden_dim: int,  
        latent_dim: int,  
        n_ode_hidden: int,  
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
    ):  
        """  
        Graph-based ODE Variational Autoencoder with time variable `t`.  
        """  
        super().__init__()  

        # Validate encoder type  
        if encoder_type not in ['linear', 'graph']:  
            raise ValueError("encoder_type must be 'linear' or 'graph'")  

        # Initialize encoder  
        if encoder_type == 'linear':  
            self.encoder = LinearEncoder_t(input_dim, hidden_dim, latent_dim, hidden_layers, dropout)  
        else:  
            self.encoder = GraphEncoder_t(  
                input_dim, hidden_dim, latent_dim, graph_type, hidden_layers, dropout, Cheb_k, alpha  
            )  

        # Initialize structure decoder  
        self.structure_decoder = GraphStructureDecoder(  
            structure_decoder=structure_decoder_type,  
            latent_dim=latent_dim,  
            hidden_dim=decoder_hidden_dim,  
            threshold=threshold,  
            sparse_threshold=sparse_threshold,  
            symmetric=True,  
            add_self_loops=False,  
        )  

        # Validate and initialize feature decoder  
        if feature_decoder_type not in ['linear', 'graph']:  
            raise ValueError("feature_decoder_type must be 'linear' or 'graph'")  

        if feature_decoder_type == 'linear':  
            self.feature_decoder = LinearDecoder(input_dim, hidden_dim, latent_dim, hidden_layers, dropout)  
        else:  
            raise NotImplementedError("Graph-based feature decoder is not supported for NODE.")  

        self.encoder_type = encoder_type  
        self.feature_decoder_type = feature_decoder_type  
        self.use_residual = use_residual  
        self.lode_func = LatentODEfunc(latent_dim, n_ode_hidden)  

    def forward(  
        self,  
        x: torch.Tensor,  
        edge_index: Optional[torch.Tensor] = None,  
        edge_weight: Optional[torch.Tensor] = None,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        Forward pass for the GODEVAE model.  
        """  
        # Encode  
        if self.encoder_type == 'linear':  
            q_z, q_m, q_s, t = self.encoder(x)  
        else:  
            if edge_index is None:  
                raise ValueError("edge_index is required for graph encoder")  
            q_z, q_m, q_s, t = self.encoder(x, edge_index, edge_weight, self.use_residual)  

        # Sort and filter time variable `t`  
        t = t.ravel()  
        idx1 = torch.argsort(t)  
        t_ordered, q_z_ordered = t[idx1], q_z[idx1]  

        idx2 = (t_ordered[:-1] != t_ordered[1:])  
        idx2 = torch.cat((idx2, torch.tensor([True], device=idx2.device)))  
        t_ordered, q_z_ordered = t_ordered[idx2], q_z_ordered[idx2] 

        self.idx1, self.idx2 = idx1, idx2
        # import pdb
        # pdb.set_trace()
        
        # Solve ODE  
        q_z_ode = odeint(  
            self.lode_func,  
            q_z_ordered[0],  
            t_ordered,  
            method='euler',  
            options=get_step_size(None, t_ordered[0], t_ordered[-1], len(t_ordered)),  
        ).to(q_z.device)  

        # Decode structure  
        pred_a, pred_edge_index, pred_edge_weight = self.structure_decoder(q_z, edge_index)  

        # Decode features  
        pred_x = self.feature_decoder(q_z)  
        pred_x_ode = self.feature_decoder(q_z_ode)  

        return q_z, q_m, q_s, pred_a, pred_x, q_z_ode, pred_x_ode