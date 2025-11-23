"""
GNODEVAE: Graph-based Neural ODE Variational Autoencoder

This module implements the core GNODEVAE model that combines:
- Graph neural networks for encoding cell-cell relationships
- Neural Ordinary Differential Equations (ODE) for modeling continuous dynamics
- Variational Autoencoders for learning latent representations

The model is designed for single-cell RNA sequencing data analysis, enabling:
- Clustering of cell types
- Trajectory inference
- Temporal dynamics modeling
"""

from .module_refined import BaseGraphNetwork, BaseLinearModel, GraphStructureDecoder, LinearDecoder  
from .utils_ODE import get_step_size, LatentODEfunc  
from torchdiffeq import odeint  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.distributions import Normal  
from typing import Optional, Tuple  


class GraphEncoder_t(BaseGraphNetwork):  
    """
    Graph-based encoder with temporal component for GNODEVAE.
    
    This encoder extends the base graph network to include a time variable `t` 
    that represents the developmental or temporal state of each cell. The time
    variable is learned during training and used by the ODE solver to model
    continuous trajectories.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension (number of genes)
    hidden_dim : int
        Hidden layer dimension
    latent_dim : int
        Latent space dimension for cell embeddings
    conv_layer_type : str, default='GAT'
        Type of graph convolutional layer ('GAT', 'GCN', 'SAGE', etc.)
    hidden_layers : int, default=2
        Number of hidden layers
    dropout : float, default=0.05
        Dropout rate for regularization
    Cheb_k : int, default=1
        Order of Chebyshev polynomials for ChebConv
    alpha : float, default=0.5
        Teleport probability for SSGConv
    """
    
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
        super().__init__(input_dim, hidden_dim, latent_dim, conv_layer_type, hidden_layers, dropout, Cheb_k, alpha)  
        # Additional layer to predict pseudo-time for each cell
        self.fc_t = nn.Linear(hidden_dim, 1)  
        self.apply(self._init_weights)  

    def _build_output_layer(self, hidden_dim: int, latent_dim: int, Cheb_k: int, alpha: float) -> None:  
        """Build output layers for mean and log variance of latent distribution."""
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
        Forward pass through the graph encoder with temporal component.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features (gene expression), shape (num_cells, input_dim)
        edge_index : torch.Tensor
            Edge indices for cell-cell graph, shape (2, num_edges)
        edge_weight : torch.Tensor, optional
            Edge weights, shape (num_edges,)
        use_residual : bool, default=True
            Whether to use residual connections
            
        Returns
        -------
        q_z : torch.Tensor
            Sampled latent representation, shape (num_cells, latent_dim)
        q_m : torch.Tensor
            Mean of latent distribution, shape (num_cells, latent_dim)
        q_s : torch.Tensor
            Log variance of latent distribution, shape (num_cells, latent_dim)
        t : torch.Tensor
            Pseudo-time for each cell, shape (num_cells, 1), values in [0, 1]
        """
        residual = None  
        
        # Pass through graph convolutional layers
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):  
            x = self._process_layer(x, conv, edge_index, edge_weight)  
            x = bn(x)  
            x = self.relu(x)  
            x = dropout(x)  
            # Save first layer output for residual connection
            if use_residual and i == 0:  
                residual = x  
        
        # Add residual connection if enabled
        if use_residual and residual is not None:  
            x = x + residual  

        # Predict pseudo-time for each cell (sigmoid ensures values in [0, 1])
        t = torch.sigmoid(self.fc_t(x))  

        # Compute parameters of latent distribution
        q_m = self._process_layer(x, self.conv_mean, edge_index, edge_weight)  
        q_s = self._process_layer(x, self.conv_logvar, edge_index, edge_weight)  

        # Sample from latent distribution using reparameterization trick
        std = F.softplus(q_s) + 1e-6  # Ensure positive standard deviation
        dist = Normal(q_m, std)  
        q_z = dist.rsample()  # Differentiable sampling

        return q_z, q_m, q_s, t  


class LinearEncoder_t(BaseLinearModel):  
    """
    Linear encoder with temporal component for GNODEVAE.
    
    This encoder uses fully connected layers instead of graph convolutions,
    making it suitable for scenarios where graph structure is not available
    or when computational efficiency is critical.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension (number of genes)
    hidden_dim : int
        Hidden layer dimension
    latent_dim : int
        Latent space dimension
    hidden_layers : int, default=2
        Number of hidden layers
    dropout : float, default=0.0
        Dropout rate
    """
    
    def __init__(  
        self,  
        input_dim: int,  
        hidden_dim: int,  
        latent_dim: int,  
        hidden_layers: int = 2,  
        dropout: float = 0.0,  
    ):  
        super().__init__(input_dim, hidden_dim, hidden_dim, hidden_layers, dropout)  
        # Output layers for variational distribution
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)  
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)  
        # Layer for pseudo-time prediction
        self.fc_t = nn.Linear(hidden_dim, 1)  

        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.mu_layer.weight)  
        nn.init.zeros_(self.mu_layer.bias)  
        nn.init.xavier_uniform_(self.logvar_layer.weight)  
        nn.init.zeros_(self.logvar_layer.bias)  
        nn.init.xavier_uniform_(self.fc_t.weight)  
        nn.init.zeros_(self.fc_t.bias)  

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  
        """
        Forward pass through the linear encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (num_cells, input_dim)
            
        Returns
        -------
        q_z : torch.Tensor
            Sampled latent representation, shape (num_cells, latent_dim)
        q_m : torch.Tensor
            Mean of latent distribution, shape (num_cells, latent_dim)
        q_s : torch.Tensor
            Log variance of latent distribution, shape (num_cells, latent_dim)
        t : torch.Tensor
            Pseudo-time for each cell, shape (num_cells, 1)
        """
        # Pass through base network (fully connected layers)
        h = self.network(x)  
        
        # Predict pseudo-time (values in [0, 1])
        t = torch.sigmoid(self.fc_t(h))  

        # Compute latent distribution parameters
        q_m = self.mu_layer(h)  
        q_s = self.logvar_layer(h)  
        
        # Sample from latent distribution
        std = F.softplus(q_s) + 1e-6  
        dist = Normal(q_m, std)  
        q_z = dist.rsample()  

        return q_z, q_m, q_s, t  


class GODEVAE_r(nn.Module):  
    """
    Graph-based Neural ODE Variational Autoencoder (GNODEVAE).
    
    This is the main model class that combines graph neural networks, neural ODEs,
    and variational autoencoders to model single-cell data with temporal dynamics.
    
    The model workflow:
    1. Encode: Graph encoder maps gene expression to latent space with pseudo-time
    2. ODE: Neural ODE models continuous trajectories in latent space
    3. Decode: Decoders reconstruct both graph structure and gene expression
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension (number of genes)
    hidden_dim : int
        Hidden layer dimension
    latent_dim : int
        Latent space dimension
    n_ode_hidden : int
        Number of hidden units in ODE function network
    encoder_type : str, default='graph'
        Type of encoder ('graph' or 'linear')
    graph_type : str, default='GAT'
        Type of graph convolution ('GAT', 'GCN', 'SAGE', etc.)
    structure_decoder_type : str, default='mlp'
        Type of structure decoder ('mlp', 'bilinear', 'inner_product')
    feature_decoder_type : str, default='linear'
        Type of feature decoder ('linear' or 'graph')
    hidden_layers : int, default=2
        Number of hidden layers
    decoder_hidden_dim : int, default=128
        Hidden dimension for structure decoder
    dropout : float, default=0.05
        Dropout rate
    use_residual : bool, default=True
        Whether to use residual connections
    Cheb_k : int, default=1
        Order of Chebyshev polynomials
    alpha : float, default=0.5
        Teleport probability for SSGConv
    threshold : float, default=0
        Threshold for edge probability in structure decoder
    sparse_threshold : int, optional
        Maximum number of edges per node
        
    Attributes
    ----------
    encoder : GraphEncoder_t or LinearEncoder_t
        Encoder that maps input to latent space with time
    structure_decoder : GraphStructureDecoder
        Decoder for reconstructing graph structure
    feature_decoder : LinearDecoder
        Decoder for reconstructing gene expression
    lode_func : LatentODEfunc
        Neural ODE function for modeling dynamics
    """
    
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
        super().__init__()  

        # Validate encoder type  
        if encoder_type not in ['linear', 'graph']:  
            raise ValueError("encoder_type must be 'linear' or 'graph'")  

        # Initialize encoder based on type
        if encoder_type == 'linear':  
            self.encoder = LinearEncoder_t(input_dim, hidden_dim, latent_dim, hidden_layers, dropout)  
        else:  
            self.encoder = GraphEncoder_t(  
                input_dim, hidden_dim, latent_dim, graph_type, hidden_layers, dropout, Cheb_k, alpha  
            )  

        # Initialize structure decoder for graph reconstruction
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
        
        # Neural ODE function for modeling continuous dynamics
        self.lode_func = LatentODEfunc(latent_dim, n_ode_hidden)  

    def forward(  
        self,  
        x: torch.Tensor,  
        edge_index: Optional[torch.Tensor] = None,  
        edge_weight: Optional[torch.Tensor] = None,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  
        """
        Forward pass through the GNODEVAE model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (gene expression), shape (num_cells, input_dim)
        edge_index : torch.Tensor, optional
            Edge indices for graph encoder, shape (2, num_edges)
        edge_weight : torch.Tensor, optional
            Edge weights, shape (num_edges,)
            
        Returns
        -------
        q_z : torch.Tensor
            Initial latent representation, shape (num_cells, latent_dim)
        q_m : torch.Tensor
            Mean of latent distribution, shape (num_cells, latent_dim)
        q_s : torch.Tensor
            Log variance of latent distribution, shape (num_cells, latent_dim)
        pred_a : torch.Tensor
            Predicted adjacency matrix or edge probabilities
        pred_x : torch.Tensor
            Reconstructed features from initial latent, shape (num_cells, input_dim)
        q_z_ode : torch.Tensor
            Latent trajectories from ODE solver, shape (num_timepoints, latent_dim)
        pred_x_ode : torch.Tensor
            Reconstructed features from ODE trajectories
        """
        # Step 1: Encode input to latent space with pseudo-time
        if self.encoder_type == 'linear':  
            q_z, q_m, q_s, t = self.encoder(x)  
        else:  
            if edge_index is None:  
                raise ValueError("edge_index is required for graph encoder")  
            q_z, q_m, q_s, t = self.encoder(x, edge_index, edge_weight, self.use_residual)  

        # Step 2: Sort cells by pseudo-time and remove duplicates
        t = t.ravel()  
        idx1 = torch.argsort(t)  
        t_ordered, q_z_ordered = t[idx1], q_z[idx1]  

        # Remove duplicate time points for ODE solver
        idx2 = (t_ordered[:-1] != t_ordered[1:])  
        idx2 = torch.cat((idx2, torch.tensor([True], device=idx2.device)))  
        t_ordered, q_z_ordered = t_ordered[idx2], q_z_ordered[idx2]  

        # Store indices for later reordering if needed
        self.idx1, self.idx2 = idx1, idx2  
        
        # Step 3: Solve ODE to get continuous trajectories in latent space
        # Uses Euler method to integrate from initial state to all time points
        q_z_ode = odeint(  
            self.lode_func,  
            q_z_ordered[0],  # Initial state
            t_ordered,        # Time points
            method='euler',  
            options=get_step_size(None, t_ordered[0], t_ordered[-1], len(t_ordered)),  
        ).to(q_z.device)  

        # Step 4: Decode structure (reconstruct cell-cell graph)
        pred_a, pred_edge_index, pred_edge_weight = self.structure_decoder(q_z, edge_index)  

        # Step 5: Decode features (reconstruct gene expression)
        pred_x = self.feature_decoder(q_z)          # From initial latent
        pred_x_ode = self.feature_decoder(q_z_ode)  # From ODE trajectories

        return q_z, q_m, q_s, pred_a, pred_x, q_z_ode, pred_x_ode