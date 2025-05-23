import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch_geometric.nn import (GCNConv,
                                ChebConv,
                                SAGEConv,
                                GraphConv,
                                TAGConv,
                                ARMAConv,
                                GATConv,
                                TransformerConv,
                                SGConv,
                                SSGConv)
from torch.distributions import Normal, kl_divergence 
from typing import Dict, Optional, Tuple, Type, Union, Literal  
from sklearn.neighbors import kneighbors_graph
import numpy as np
from .utils import GraphStructureDecoder

class BaseGraphNetwork(nn.Module):  
    """  
    Base class for graph neural networks with various convolution types.  
    
    Parameters  
    ----------  
    input_dim : int  
        Input feature dimension  
    hidden_dim : int  
        Hidden layer dimension  
    output_dim : int  
        Output feature dimension  
    conv_layer_type : str  
        Type of graph convolutional layer  
    hidden_layers : int  
        Number of hidden layers  
    dropout : float  
        Dropout rate  
    Cheb_k : int, optional  
        Order of Chebyshev polynomials for ChebConv  
    alpha : float, optional  
        Teleport probability for SSGConv  
    """  
    
    CONV_LAYERS: Dict[str, Type[nn.Module]] = {  
        'GCN': GCNConv,  
        'Cheb': ChebConv,  
        'SAGE': SAGEConv,  
        'Graph': GraphConv,  
        'TAG': TAGConv,  
        'ARMA': ARMAConv,  
        'GAT': GATConv,  
        'Transformer': TransformerConv,  
        'SG': SGConv,  
        'SSG': SSGConv  
    }  

    def __init__(  
        self,  
        input_dim: int,  
        hidden_dim: int,  
        output_dim: int,  
        conv_layer_type: str = 'GAT',  
        hidden_layers: int = 2,  
        dropout: float = 0.05,  
        Cheb_k: int = 1,  
        alpha: float = 0.5  
    ):  
        super().__init__()  
        self._validate_conv_type(conv_layer_type)  
        self._init_attributes(conv_layer_type, hidden_layers, dropout)  
        self._build_network(input_dim, hidden_dim, output_dim, Cheb_k, alpha)  
        self.disp = nn.Parameter(torch.randn(output_dim))
        self.apply(self._init_weights)  

    def _validate_conv_type(self, conv_layer_type: str) -> None:  
        if conv_layer_type not in self.CONV_LAYERS:  
            raise ValueError(f"Unsupported layer type: {conv_layer_type}. Choose from {list(self.CONV_LAYERS.keys())}")  

    def _init_attributes(self, conv_layer_type: str, hidden_layers: int, dropout: float) -> None:  
        self.conv_layer_type = conv_layer_type  
        self.conv_layer = self.CONV_LAYERS[conv_layer_type]  
        self.hidden_layers = hidden_layers  
        self.dropout = dropout  
        self.convs = nn.ModuleList()  
        self.bns = nn.ModuleList()  
        self.dropouts = nn.ModuleList()  
        self.relu = nn.ReLU()  

    def _create_conv_layer(self, in_dim: int, out_dim: int, Cheb_k: int, alpha: float) -> nn.Module:  
        if self.conv_layer_type == 'Transformer':  
            return self.conv_layer(in_dim, out_dim, edge_dim=1)  
        elif self.conv_layer_type == 'Cheb':  
            return self.conv_layer(in_dim, out_dim, Cheb_k)  
        elif self.conv_layer_type == 'SSG':  
            return self.conv_layer(in_dim, out_dim, alpha=alpha)  
        return self.conv_layer(in_dim, out_dim)  

    def _build_network(self, input_dim: int, hidden_dim: int, output_dim: int, Cheb_k: int, alpha: float) -> None:  
        # Input layer  
        self.convs.append(self._create_conv_layer(input_dim, hidden_dim, Cheb_k, alpha))  
        self.bns.append(nn.BatchNorm1d(hidden_dim))  
        self.dropouts.append(nn.Dropout(self.dropout))  

        # Hidden layers  
        for _ in range(self.hidden_layers - 1):  
            self.convs.append(self._create_conv_layer(hidden_dim, hidden_dim, Cheb_k, alpha))  
            self.bns.append(nn.BatchNorm1d(hidden_dim))  
            self.dropouts.append(nn.Dropout(self.dropout))  

        # Output layer - implemented by subclasses  
        self._build_output_layer(hidden_dim, output_dim, Cheb_k, alpha)  

    def _build_output_layer(self, hidden_dim: int, output_dim: int, Cheb_k: int, alpha: float) -> None:  
        raise NotImplementedError("Subclasses must implement _build_output_layer")  

    @staticmethod  
    def _init_weights(m: nn.Module) -> None:  
        if isinstance(m, (nn.Linear, nn.Conv1d)):  
            nn.init.xavier_uniform_(m.weight)  
            if m.bias is not None:  
                nn.init.zeros_(m.bias)  

    def _process_layer(self, x: torch.Tensor, conv: nn.Module, edge_index: torch.Tensor,  
                      edge_weight: Optional[torch.Tensor]) -> torch.Tensor:  
        if isinstance(conv, SAGEConv):  
            return conv(x, edge_index)  
        elif isinstance(conv, TransformerConv):  
            return conv(x, edge_index, edge_weight.view(-1, 1))  
        return conv(x, edge_index, edge_weight)  

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,  
                edge_weight: Optional[torch.Tensor] = None,  
                use_residual: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:  
        raise NotImplementedError("Subclasses must implement forward")  


class GraphEncoder(BaseGraphNetwork):  
    """Graph encoder with variational output."""  
    
    def _build_output_layer(self, hidden_dim: int, latent_dim: int, Cheb_k: int, alpha: float) -> None:  
        self.conv_mean = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)  
        self.conv_logvar = self._create_conv_layer(hidden_dim, latent_dim, Cheb_k, alpha)  

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,  
                edge_weight: Optional[torch.Tensor] = None,  
                use_residual: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        residual = None  

        # Process hidden layers  
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):  
            x = self._process_layer(x, conv, edge_index, edge_weight)  
            x = bn(x)  
            x = self.relu(x)  
            x = dropout(x)  
            
            if use_residual and i == 0:  
                residual = x  

        if use_residual and residual is not None:  
            x = x + residual  

        # Compute variational parameters  
        q_m = self._process_layer(x, self.conv_mean, edge_index, edge_weight)  
        q_s = self._process_layer(x, self.conv_logvar, edge_index, edge_weight)  
        
        # Sample from the latent distribution  
        std = F.softplus(q_s) + 1e-6  
        dist = Normal(q_m, std)  
        q_z = dist.rsample()  

        return q_z, q_m, q_s  


class GraphDecoder(BaseGraphNetwork):  
    """Graph decoder with softmax output."""  
    
    def _build_output_layer(self, hidden_dim: int, output_dim: int, Cheb_k: int, alpha: float) -> None:  
        self.output_conv = self._create_conv_layer(hidden_dim, output_dim, Cheb_k, alpha)  
        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,  
                edge_weight: Optional[torch.Tensor] = None,  
                use_residual: bool = True) -> torch.Tensor:  
        residual = None  

        # Process hidden layers  
        for i, (conv, bn, dropout) in enumerate(zip(self.convs, self.bns, self.dropouts)):  
            x = self._process_layer(x, conv, edge_index, edge_weight)  
            x = bn(x)  
            x = self.relu(x)  
            x = dropout(x)  
            
            if use_residual and i == 0:  
                residual = x  

        if use_residual and residual is not None:  
            x = x + residual  

        # Apply output layer with softmax  
        x = self._process_layer(x, self.output_conv, edge_index, edge_weight)  
        return self.softmax(x)


class BaseLinearModel(nn.Module):  
    """  
    Base linear model for neural network encoders/decoders.  
    
    Parameters  
    ----------  
    input_dim : int  
        Input feature dimension.  
    hidden_dim : int  
        Hidden layer dimension.  
    output_dim : int  
        Output feature dimension.  
    hidden_layers : int, optional  
        Number of hidden layers, by default 2.  
    dropout : float, optional  
        Dropout rate, by default 0.0.  
    """  
    def __init__(  
        self,   
        input_dim: int,   
        hidden_dim: int,   
        output_dim: int,   
        hidden_layers: int = 2,  
        dropout: float = 0.0  
    ):  
        super().__init__()  
        self.input_dim = input_dim  
        self.hidden_dim = hidden_dim  
        self.output_dim = output_dim  
        self.hidden_layers = hidden_layers  
        
        # Build network layers  
        layers = []  
        # Input layer  
        layers.extend([  
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),  
            nn.Dropout(dropout)  
        ])  
        
        # Hidden layers  
        for _ in range(hidden_layers - 1):  
            layers.extend([  
                nn.Linear(hidden_dim, hidden_dim),  
                nn.ReLU(),  
                nn.Dropout(dropout)  
            ])  
        
        # Output layer  
        layers.append(nn.Linear(hidden_dim, output_dim))  
        
        self.network = nn.Sequential(*layers)  
        self.apply(self._init_weights)  
    
    def _init_weights(self, module: nn.Module) -> None:  
        """Initialize network weights using Xavier uniform initialization."""  
        if isinstance(module, nn.Linear):  
            nn.init.xavier_uniform_(module.weight)  
            if module.bias is not None:  
                nn.init.zeros_(module.bias)  
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """  
        Forward pass through the network.  
        
        Parameters  
        ----------  
        x : torch.Tensor  
            Input tensor of shape (batch_size, input_dim)  
            
        Returns  
        -------  
        torch.Tensor  
            Output tensor of shape (batch_size, output_dim)  
        """  
        return self.network(x)  


class LinearEncoder(BaseLinearModel):  
    """  
    Feature encoder network that maps input features to latent space.  
    
    Parameters  
    ----------  
    input_dim : int  
        Original input feature dimension.  
    hidden_dim : int  
        Hidden layer dimension.  
    latent_dim : int  
        Latent space dimension.  
    hidden_layers : int, optional  
        Number of hidden layers, by default 2.  
    dropout : float, optional  
        Dropout rate, by default 0.0.  
    """  
    def __init__(  
        self,   
        input_dim: int,   
        hidden_dim: int,   
        latent_dim: int,   
        hidden_layers: int = 2,  
        dropout: float = 0.0  
    ):  
        super().__init__(  
            input_dim=input_dim,  
            hidden_dim=hidden_dim,  
            output_dim=hidden_dim,  
            hidden_layers=hidden_layers,  
            dropout=dropout  
        )  
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)  
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim) 
        # Initialize weights for the new layers  
        nn.init.xavier_uniform_(self.mu_layer.weight)  
        nn.init.zeros_(self.mu_layer.bias)  
        nn.init.xavier_uniform_(self.logvar_layer.weight)  
        nn.init.zeros_(self.logvar_layer.bias)  
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        Forward pass through the encoder network.  
        
        Parameters  
        ----------  
        x : torch.Tensor  
            Input tensor of shape (batch_size, input_dim)  
            
        Returns  
        -------  
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]  
            Tuple containing:  
            - z: Sampled latent vector  
            - mu: Mean of the latent distribution  
            - logvar: Log variance of the latent distribution  
        """  
        h = self.network(x)  
        
        q_m = self.mu_layer(h)  
        q_s = self.logvar_layer(h)  
        std = F.softplus(q_s) + 1e-6  
        dist = Normal(q_m, std)  
        q_z = dist.rsample()  
        return q_z, q_m, q_s

class LinearDecoder(BaseLinearModel):  
    """  
    Feature decoder network that maps latent representations back to feature space.  
    
    Parameters  
    ----------  
    input_dim : int  
        Original input feature dimension.  
    hidden_dim : int  
        Hidden layer dimension.  
    latent_dim : int  
        Latent space dimension.  
    hidden_layers : int, optional  
        Number of hidden layers, by default 2.  
    dropout : float, optional  
        Dropout rate, by default 0.0.  
    """  
    def __init__(  
        self,   
        input_dim: int,   
        hidden_dim: int,   
        latent_dim: int,   
        hidden_layers: int = 2,  
        dropout: float = 0.0  
    ):  
        super().__init__(  
            input_dim=latent_dim,  
            hidden_dim=hidden_dim,  
            output_dim=input_dim,  
            hidden_layers=hidden_layers,  
            dropout=dropout  
        )  
        self.disp = nn.Parameter(torch.randn(input_dim))
        # Add softmax activation for the output layer  
        self.network = nn.Sequential(  
            self.network,  
            nn.Softmax(dim=-1)  
        )


class GraphVAE_r(nn.Module):  
    """  
    Variational Graph Autoencoder (VGAE) with flexible encoder and decoder options.  
    
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
        sparse_threshold: Optional[int] = None  
    ):  
        super().__init__()  
        
        # Validate encoder type  
        if encoder_type not in ['linear', 'graph']:  
            raise ValueError("encoder_type must be 'linear' or 'graph'")  
        
        # Initialize encoder  
        if encoder_type == 'linear':  
            self.encoder = LinearEncoder(  
                input_dim=input_dim,  
                hidden_dim=hidden_dim,  
                latent_dim=latent_dim,  
                hidden_layers=hidden_layers,  
                dropout=dropout  
            )  
        else:  # graph  
            self.encoder = GraphEncoder(  
                input_dim=input_dim,  
                hidden_dim=hidden_dim,  
                output_dim=latent_dim,  
                conv_layer_type=graph_type,  
                hidden_layers=hidden_layers,  
                dropout=dropout,  
                Cheb_k=Cheb_k,  
                alpha=alpha  
            )  
        
        # Initialize structure decoder  
        self.structure_decoder = GraphStructureDecoder(  
            structure_decoder=structure_decoder_type,  
            latent_dim=latent_dim,  
            hidden_dim=decoder_hidden_dim,
            threshold=threshold,
            sparse_threshold=sparse_threshold,  
            symmetric=True,  
            add_self_loops=False  
        )  
        
        # Validate and initialize feature decoder  
        if feature_decoder_type not in ['linear', 'graph']:  
            raise ValueError("feature_decoder_type must be 'linear' or 'graph'")  
            
        if feature_decoder_type == 'linear':  
            self.feature_decoder = LinearDecoder(  
                input_dim=input_dim,  
                hidden_dim=hidden_dim,  
                latent_dim=latent_dim,  
                hidden_layers=hidden_layers,  
                dropout=dropout  
            )  
        else:  # graph  
            self.feature_decoder = GraphDecoder(  
                input_dim=latent_dim,  
                hidden_dim=hidden_dim,  
                output_dim=input_dim,  
                conv_layer_type=graph_type,  
                hidden_layers=hidden_layers,  
                dropout=dropout,  
                Cheb_k=Cheb_k,  
                alpha=alpha  
            )  
        
        self.encoder_type = encoder_type  
        self.feature_decoder_type = feature_decoder_type  
        self.use_residual = use_residual  

    def forward(  
        self,  
        x: torch.Tensor,  
        edge_index: Optional[torch.Tensor] = None,  
        edge_weight: Optional[torch.Tensor] = None  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        Forward pass of the GraphVAE.  
        
        Parameters  
        ----------  
        x : torch.Tensor  
            Node features (num_nodes, input_dim)  
        edge_index : torch.Tensor, optional  
            Edge indices (2, num_edges), required for graph encoder  
        edge_weight : torch.Tensor, optional  
            Edge weights (num_edges,), required for graph encoder
            
        Returns  
        -------  
        q_z : torch.Tensor  
            Sampled latent representation  
        q_m : torch.Tensor  
            Mean of latent distribution  
        q_s : torch.Tensor  
            Log variance of latent distribution  
        pred_adj : torch.Tensor  
            Reconstructed adjacency matrix  
        pred_x : torch.Tensor  
            Reconstructed node features  
        """  
        # Encode  
        if self.encoder_type == 'linear':  
            q_z, q_m, q_s = self.encoder(x)  
        else:  # graph  
            if edge_index is None:  
                raise ValueError("edge_index required for graph encoder")  
            q_z, q_m, q_s = self.encoder(x, edge_index, edge_weight, self.use_residual)  
        
        # Decode structure  
        pred_a, pred_edge_index, pred_edge_weight = self.structure_decoder(q_z, edge_index)  
        
        # Decode features  
        if self.feature_decoder_type == 'linear':  
            pred_x = self.feature_decoder(q_z)  
        else:  # graph  
            pred_x = self.feature_decoder(  
                q_z, pred_edge_index, pred_edge_weight  
            )  
        
        return q_z, q_m, q_s, pred_a, pred_x


        