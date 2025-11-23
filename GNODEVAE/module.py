"""
Graph Variational Autoencoder (GraphVAE) Modules

This module implements the core neural network architectures for GNODEVAE's
GraphVAE models. It provides flexible encoder and decoder components that can
be combined to create various VAE architectures for single-cell data analysis.

Key Components:
- GraphEncoder: Graph-based encoder using various GNN layers
- BilinearDecoder: Bilinear graph structure decoder
- InnerProductDecoder: Inner product graph structure decoder
- MLPDecoder: MLP-based graph structure decoder
- FeatureEncoder: Feature encoding network
- FeatureDecoder: Feature decoding network
- GraphVAE: Complete VAE model combining encoder and decoders
- iGraphVAE: Interpretable VAE with additional interpretable latent space
"""

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
from typing import Optional, Tuple  
from sklearn.neighbors import kneighbors_graph
import numpy as np

class GraphEncoder(nn.Module):  
    """
    Flexible graph neural network encoder with variational output.
    
    This encoder supports multiple types of graph convolution layers and produces
    a probabilistic latent representation suitable for variational autoencoders.
    It uses batch normalization and dropout for regularization, and optionally
    includes residual connections for improved gradient flow.
    
    The encoder architecture:
    1. Multiple graph convolutional layers with batch norm and dropout
    2. ReLU activation between layers
    3. Optional residual connection from first layer
    4. Two output layers for mean and log variance of latent distribution  
    
    Parameters  
    ----------  
    input_dim : int  
        Input feature dimension (number of genes after preprocessing).
    hidden_dim : int  
        Hidden layer dimension for intermediate representations.
    latent_dim : int  
        Latent space dimension for cell embeddings.
    conv_layer_type : str, default='GAT'
        Type of graph convolutional layer. Options:
        - 'GCN': Graph Convolutional Network
        - 'Cheb': Chebyshev spectral graph convolution
        - 'SAGE': GraphSAGE
        - 'Graph': Basic graph convolution
        - 'TAG': Topology Adaptive Graph convolution
        - 'ARMA': ARMA graph convolution
        - 'GAT': Graph Attention Network (recommended)
        - 'Transformer': Graph Transformer
        - 'SG': Simplifying Graph convolution
        - 'SSG': Simple Spectral Graph convolution
    hidden_layers : int, default=2
        Number of hidden layers (excluding input and output layers).
    dropout : float, default=0.05
        Dropout rate for regularization (applied after each layer).
    Cheb_k : int, default=1
        Order of Chebyshev polynomials for ChebConv layer.
    alpha : float, default=0.5
        Teleport probability for SSGConv layer.
        
    Attributes
    ----------
    convs : nn.ModuleList
        List of graph convolutional layers
    bns : nn.ModuleList
        List of batch normalization layers
    dropouts : nn.ModuleList
        List of dropout layers
    conv_mean : nn.Module
        Output layer for latent mean
    conv_logvar : nn.Module
        Output layer for latent log variance
        
    Notes
    -----
    Different graph convolution types have different properties:
    - GAT: Uses attention mechanism, good for heterogeneous graphs
    - GCN: Classic choice, computationally efficient
    - SAGE: Handles variable neighborhood sizes well
    - Transformer: Most expressive but computationally expensive
    """  

    def __init__(  
        self,  
        input_dim: int,  
        hidden_dim: int,  
        latent_dim: int,  
        conv_layer_type: str = 'GAT',  
        hidden_layers: int = 2,  
        dropout: float = 0.05,  
        Cheb_k: Optional[int] = 1,  
        alpha: Optional[float] = 0.5,    
    ):  
        super(GraphEncoder, self).__init__()  

        # Map string to the corresponding Torch Geometric layer  
        layer_map = {  
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
        if conv_layer_type not in layer_map:  
            raise ValueError(f"Unsupported layer type: {conv_layer_type}. Choose from {list(layer_map.keys())}.")  
        self.conv_layer_type = conv_layer_type  
        self.conv_layer = layer_map[conv_layer_type]  

        self.hidden_layers = hidden_layers  
        self.dropout = dropout  

        # Define the layers dynamically  
        self.convs = nn.ModuleList()  
        self.bns = nn.ModuleList()  
        self.dropouts = nn.ModuleList()  

        # Input layer  
        if self.conv_layer_type == 'Transformer':  
            self.convs.append(self.conv_layer(input_dim, hidden_dim, edge_dim=1))  # Set edge_dim=1 for TransformerConv  
        elif self.conv_layer_type == 'Cheb':  
            self.convs.append(self.conv_layer(input_dim, hidden_dim, Cheb_k))  # Set k=Cheb_k for ChebConv  
        elif self.conv_layer_type == 'SSG':  
            self.convs.append(self.conv_layer(input_dim, hidden_dim, alpha=alpha))  # Set alpha for SSGConv  
        else:  
            self.convs.append(self.conv_layer(input_dim, hidden_dim))  
        self.bns.append(nn.BatchNorm1d(hidden_dim))  
        self.dropouts.append(nn.Dropout(dropout))  

        # Hidden layers  
        for _ in range(hidden_layers - 1):  
            if self.conv_layer_type == 'Transformer':  
                self.convs.append(self.conv_layer(hidden_dim, hidden_dim, edge_dim=1))  # Set edge_dim=1 for TransformerConv  
            elif self.conv_layer_type == 'Cheb':  
                self.convs.append(self.conv_layer(hidden_dim, hidden_dim, Cheb_k))  # Set k=Cheb_k for ChebConv  
            elif self.conv_layer_type == 'SSG':  
                self.convs.append(self.conv_layer(hidden_dim, hidden_dim, alpha=alpha))  # Set alpha for SSGConv  
            else:  
                self.convs.append(self.conv_layer(hidden_dim, hidden_dim))  
            self.bns.append(nn.BatchNorm1d(hidden_dim))  
            self.dropouts.append(nn.Dropout(dropout))  

        # Latent layers for mean and log variance  
        if self.conv_layer_type == 'Transformer':  
            self.conv_mean = self.conv_layer(hidden_dim, latent_dim, edge_dim=1)  # Set edge_dim=1 for TransformerConv  
            self.conv_logvar = self.conv_layer(hidden_dim, latent_dim, edge_dim=1)  # Set edge_dim=1 for TransformerConv  
        elif self.conv_layer_type == 'Cheb':  
            self.conv_mean = self.conv_layer(hidden_dim, latent_dim, Cheb_k)  # Set k=Cheb_k for ChebConv  
            self.conv_logvar = self.conv_layer(hidden_dim, latent_dim, Cheb_k)  # Set k=Cheb_k for ChebConv  
        elif self.conv_layer_type == 'SSG':  
            self.conv_mean = self.conv_layer(hidden_dim, latent_dim, alpha=alpha)  # Set alpha for SSGConv  
            self.conv_logvar = self.conv_layer(hidden_dim, latent_dim, alpha=alpha)  # Set alpha for SSGConv  
        else:  
            self.conv_mean = self.conv_layer(hidden_dim, latent_dim)  
            self.conv_logvar = self.conv_layer(hidden_dim, latent_dim)  

        self.relu = nn.ReLU()  
        self.apply(self.weight_init) 

    def weight_init(self, m: nn.Module) -> None:  
        """  
        Initialize weights for the layers.  

        Parameters  
        ----------  
        m : nn.Module  
            A module to initialize.  
        """  
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):  
            nn.init.xavier_uniform_(m.weight)  
            if m.bias is not None:  
                nn.init.zeros_(m.bias)  

    def forward(  
        self,  
        x: torch.Tensor,  
        edge_index: torch.Tensor,  
        edge_weight: Optional[torch.Tensor] = None,  
        use_residual: bool = True,    
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        Forward pass of the Graph Encoder.  
    
        Parameters  
        ----------  
        x : torch.Tensor  
            Node features, shape (num_nodes, input_dim).  
        edge_index : torch.Tensor  
            Edge indices, shape (2, num_edges).  
        edge_weight : torch.Tensor, optional  
            Edge weights, shape (num_edges,), by default None.  
        use_residual : bool, optional  
            Whether to use residual connections, by default True.    
    
        Returns  
        -------  
        q_z : torch.Tensor  
            Latent representation, shape (num_nodes, latent_dim).  
        q_m : torch.Tensor  
            Mean of the latent distribution, shape (num_nodes, latent_dim).  
        q_s : torch.Tensor  
            Standard deviation of the latent distribution, shape (num_nodes, latent_dim).  
        """  
        residual = None  
    
        # Apply convolutional layers  
        for i in range(self.hidden_layers):  
            # Handle different convolutional layers based on their type  
            if isinstance(self.convs[i], SAGEConv):  
                x = self.convs[i](x, edge_index)  # Ignore edge_weight for SAGEConv    
            elif isinstance(self.convs[i], TransformerConv):  
                x = self.convs[i](x, edge_index, edge_weight.view(-1, 1))  # Reshape edge_weight for TransformerConv  
            else:  
                x = self.convs[i](x, edge_index, edge_weight)  # Pass edge_weight for other layers  
    
            # Apply batch normalization, activation, and dropout  
            x = self.bns[i](x)   
            x = self.relu(x)  
            x = self.dropouts[i](x)  
    
            # Add residual connection for the first layer if enabled  
            if use_residual and i == 0:  
                residual = x  
    
        # Add residual connection after the last hidden layer if enabled  
        if use_residual and residual is not None:  
            x += residual  
    
        # Compute mean and log variance for the latent space  
        if isinstance(self.conv_mean, SAGEConv):  
            q_m = self.conv_mean(x, edge_index)  # Ignore edge_weight for SAGEConv      
        elif isinstance(self.conv_mean, TransformerConv):  
            q_m = self.conv_mean(x, edge_index, edge_weight.view(-1, 1))  # Reshape edge_weight for TransformerConv  
        else:  
            q_m = self.conv_mean(x, edge_index, edge_weight)  # Pass edge_weight for other layers  
    
        if isinstance(self.conv_logvar, SAGEConv):  
            q_s = self.conv_logvar(x, edge_index)  # Ignore edge_weight for SAGEConv    
        elif isinstance(self.conv_logvar, TransformerConv):  
            q_s = self.conv_logvar(x, edge_index, edge_weight.view(-1, 1))  # Reshape edge_weight for TransformerConv  
        else:  
            q_s = self.conv_logvar(x, edge_index, edge_weight)  # Pass edge_weight for other layers  
    
        # Compute standard deviation and sample from the latent distribution  
        s = F.softplus(q_s) + 1e-6  
        n = Normal(q_m, s)  
        q_z = n.rsample()  
    
        return q_z, q_m, q_s



class BilinearDecoder(nn.Module):  
    """
    Bilinear graph structure decoder.
    
    This decoder reconstructs the adjacency matrix using a learned bilinear
    transformation. It computes edge probabilities as:
    A_pred = sigmoid(Z * W * Z^T)
    where Z is the latent embedding matrix and W is a learned weight matrix.
    
    Parameters  
    ----------  
    latent_dim : int  
        Latent space dimension.
        
    Attributes
    ----------
    weight : nn.Parameter
        Learnable bilinear transformation matrix, shape (latent_dim, latent_dim).
        
    Notes
    -----
    The bilinear decoder is more expressive than inner product but has
    O(latent_dim^2) parameters, which can be memory-intensive for large
    latent dimensions.
    """  

    def __init__(self, latent_dim: int):  
        super(BilinearDecoder, self).__init__()  
        self.weight = nn.Parameter(torch.Tensor(latent_dim, latent_dim))  
        nn.init.xavier_uniform_(self.weight)  

    def forward(self, z: torch.Tensor) -> torch.Tensor:  
        """
        Reconstruct adjacency matrix using bilinear transformation.
        
        Parameters  
        ----------  
        z : torch.Tensor  
            Latent node embeddings, shape (num_nodes, latent_dim).
            
        Returns  
        -------  
        adj_recon : torch.Tensor  
            Reconstructed adjacency matrix with edge probabilities,
            shape (num_nodes, num_nodes). Values in [0, 1].
        """  
        adj_recon = torch.sigmoid(torch.matmul(z @ self.weight, z.t()))  
        return adj_recon  


class InnerProductDecoder(nn.Module):  
    """
    Inner product graph structure decoder.
    
    This is the simplest graph decoder that reconstructs edge probabilities
    using the inner product of latent embeddings:
    A_pred = sigmoid(Z * Z^T)
    
    This decoder has no learnable parameters and is computationally efficient,
    making it suitable for large graphs. It assumes that similar cells (high
    inner product) should be connected.
    
    Notes
    -----
    While parameter-free, this decoder is less expressive than bilinear or
    MLP decoders. It works well when the latent space naturally encodes
    cell similarity.
    """  

    def __init__(self):  
        super(InnerProductDecoder, self).__init__()  

    def forward(self, z: torch.Tensor) -> torch.Tensor:  
        """
        Reconstruct adjacency matrix using inner product.
        
        Parameters  
        ----------  
        z : torch.Tensor  
            Latent node embeddings, shape (num_nodes, latent_dim).
            
        Returns  
        -------  
        adj_recon : torch.Tensor  
            Reconstructed adjacency matrix with edge probabilities,
            shape (num_nodes, num_nodes). Values in [0, 1].
        """  
        adj_recon = torch.sigmoid(torch.matmul(z, z.t()))  
        return adj_recon  


class MLPDecoder(nn.Module):  
    """
    Multi-layer perceptron (MLP) graph structure decoder.
    
    This decoder uses a neural network to predict edge probabilities from
    concatenated node embeddings. It's the most expressive decoder type,
    capable of learning complex edge probability functions.
    
    Architecture: [latent_dim*2] -> [hidden_dim] -> ReLU -> [1] -> Sigmoid
    
    Parameters  
    ----------  
    latent_dim : int  
        Latent space dimension.
    hidden_dim : int  
        Hidden layer dimension for the MLP.
        
    Attributes
    ----------
    mlp : nn.Sequential
        MLP network for edge prediction.
        
    Notes
    -----
    This decoder only computes edge probabilities for existing edges
    (specified by edge_index), making it memory-efficient for sparse graphs.
    For dense graphs, consider bilinear or inner product decoders.
    """  

    def __init__(self, latent_dim: int, hidden_dim: int):  
        super(MLPDecoder, self).__init__()  
        self.mlp = nn.Sequential(  
            nn.Linear(latent_dim * 2, hidden_dim),  
            nn.ReLU(),  
            nn.Linear(hidden_dim, 1),  
            nn.Sigmoid(),  
        )  

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  
        """
        Reconstruct adjacency matrix using an MLP.
        
        Parameters  
        ----------  
        z : torch.Tensor  
            Latent node embeddings, shape (num_nodes, latent_dim).
        edge_index : torch.Tensor  
            Edge indices, shape (2, num_edges). Specifies which edges to predict.
            
        Returns  
        -------  
        adj_recon : torch.Tensor  
            Reconstructed adjacency matrix with edge probabilities,
            shape (num_nodes, num_nodes). Only predicted edges have non-zero values.
        """  
        num_nodes = z.size(0)  
        row, col = edge_index  
        # Concatenate source and target node embeddings
        edge_features = torch.cat([z[row], z[col]], dim=1)  
        edge_probs = self.mlp(edge_features).squeeze()  
        # Convert to adjacency matrix format
        adj_recon = torch.zeros((num_nodes, num_nodes), device=z.device)  
        adj_recon[row, col] = edge_probs  
        return adj_recon  


class FeatureEncoder(nn.Module):  
    """
    Feature encoder for mapping high-dimensional data to latent space.
    
    This encoder uses fully connected layers to encode features. Unlike
    GraphEncoder, it doesn't use graph structure, making it suitable for
    scenarios where graph information is unavailable or for auxiliary tasks.
    
    Parameters  
    ----------  
    input_dim : int  
        Original input feature dimension (e.g., number of genes).
    hidden_dim : int  
        Hidden layer dimension.
    latent_dim : int  
        Latent space dimension (output dimension).
    hidden_layers : int, default=2
        Number of hidden layers in the encoder.
        
    Attributes
    ----------
    nn : nn.Sequential
        Sequential neural network for encoding.
    disp : nn.Parameter
        Dispersion parameter for negative binomial distribution (for gene expression modeling).
    """  

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, hidden_layers: int = 2):  
        super(FeatureEncoder, self).__init__()  
        self.hidden_layers = hidden_layers  

        layers = []  
        layers.append(nn.Linear(input_dim, hidden_dim))  
        layers.append(nn.ReLU())  

        for _ in range(hidden_layers - 1):  
            layers.append(nn.Linear(hidden_dim, hidden_dim))  
            layers.append(nn.ReLU())  

        layers.append(nn.Linear(hidden_dim, latent_dim))    

        self.nn = nn.Sequential(*layers)  
        self.disp = nn.Parameter(torch.randn(input_dim))  
        self.apply(self.weight_init)  

    def weight_init(self, m: nn.Module) -> None:  
        """
        Initialize weights using Xavier initialization.
        
        Parameters  
        ----------  
        m : nn.Module  
            Module to initialize.
        """  
        if isinstance(m, nn.Linear):  
            nn.init.xavier_uniform_(m.weight)  
            if m.bias is not None:  
                nn.init.zeros_(m.bias)  

    def forward(self, z: torch.Tensor) -> torch.Tensor:  
        """
        Encode features to latent space.
        
        Parameters  
        ----------  
        z : torch.Tensor  
            Input features, shape (num_cells, input_dim).
            
        Returns  
        -------  
        torch.Tensor
            Latent representation, shape (num_cells, latent_dim).
        """  
        return self.nn(z)  



class FeatureDecoder(nn.Module):  
    """
    Feature decoder for reconstructing gene expression from latent space.
    
    This decoder takes latent embeddings and reconstructs the original gene
    expression profiles. It uses a softmax output to produce normalized
    expression values, which is suitable for count-based data.
    
    Parameters  
    ----------  
    input_dim : int  
        Original input feature dimension (number of genes to reconstruct).
    hidden_dim : int  
        Hidden layer dimension.
    latent_dim : int  
        Latent space dimension (input dimension).
    hidden_layers : int, default=2
        Number of hidden layers in the decoder.
        
    Attributes
    ----------
    nn : nn.Sequential
        Sequential neural network for decoding.
    disp : nn.Parameter
        Dispersion parameter for modeling gene expression variance.
        
    Notes
    -----
    The softmax output ensures that reconstructed values sum to a constant,
    which is appropriate for compositional data like gene expression after
    normalization. For count data, consider using a negative binomial
    likelihood in the loss function.
    """  

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, hidden_layers: int = 2):  
        super(FeatureDecoder, self).__init__()  
        self.hidden_layers = hidden_layers  

        layers = []  
        layers.append(nn.Linear(latent_dim, hidden_dim))  
        layers.append(nn.ReLU())  

        for _ in range(hidden_layers - 1):  
            layers.append(nn.Linear(hidden_dim, hidden_dim))  
            layers.append(nn.ReLU())  

        layers.append(nn.Linear(hidden_dim, input_dim))  
        layers.append(nn.Softmax(dim=-1))  # Convert to probability distribution

        self.nn = nn.Sequential(*layers)  
        self.disp = nn.Parameter(torch.randn(input_dim))  
        self.apply(self.weight_init)  

    def weight_init(self, m: nn.Module) -> None:  
        """
        Initialize weights using Xavier initialization.
        
        Parameters  
        ----------  
        m : nn.Module  
            Module to initialize.
        """  
        if isinstance(m, nn.Linear):  
            nn.init.xavier_uniform_(m.weight)  
            if m.bias is not None:  
                nn.init.zeros_(m.bias)  

    def forward(self, z: torch.Tensor) -> torch.Tensor:  
        """
        Reconstruct gene expression from latent embeddings.
        
        Parameters  
        ----------  
        z : torch.Tensor  
            Latent node embeddings, shape (num_cells, latent_dim).
            
        Returns  
        -------  
        recon_features : torch.Tensor  
            Reconstructed gene expression, shape (num_cells, input_dim).
            Values are normalized via softmax.
        """  
        return self.nn(z)  


class GraphVAE(nn.Module):  
    """
    Complete Graph Variational Autoencoder model.
    
    This class combines a graph encoder, structure decoder, and feature decoder
    to create a complete VAE model for single-cell data. The model learns to:
    1. Encode cells into a low-dimensional latent space using graph structure
    2. Reconstruct the cell-cell similarity graph
    3. Reconstruct gene expression profiles
    
    The VAE framework enables:
    - Learning of robust latent representations
    - Handling of noise and batch effects
    - Uncertainty quantification via the probabilistic latent space
    
    Parameters  
    ----------  
    input_dim : int  
        Input feature dimension (number of genes).
    hidden_dim : int  
        Hidden layer dimension for encoder and decoder.
    latent_dim : int  
        Latent space dimension for cell embeddings.
    encoder_type : str, default='GAT'
        Type of graph convolutional layer for the encoder.
        Options: 'GCN', 'Cheb', 'SAGE', 'TAG', 'ARMA', 'GAT', 'Transformer'.
    encoder_hidden_layers : int, default=2
        Number of hidden layers in the graph encoder.
    decoder_type : str, default='MLP'
        Type of graph decoder for structure reconstruction.
        Options: 'Bilinear', 'InnerProduct', 'MLP'.
    decoder_hidden_dim : int, default=128
        Hidden dimension for the MLP decoder (if used).
    feature_decoder_hidden_layers : int, default=2
        Number of hidden layers in the feature decoder.
    dropout : float, default=0.05
        Dropout rate for regularization.
    use_residual : bool, default=True
        Whether to use residual connections in the encoder.
    Cheb_k : int, default=1
        Order of Chebyshev polynomials (for ChebConv).
    alpha : float, default=0.5
        Teleport probability (for SSGConv).
        
    Attributes
    ----------
    g_encoder : GraphEncoder
        Graph-based encoder network.
    a_decoder : BilinearDecoder, InnerProductDecoder, or MLPDecoder
        Decoder for reconstructing graph structure.
    x_decoder : FeatureDecoder
        Decoder for reconstructing gene expression.
    use_residual : bool
        Whether residual connections are enabled.
        
    Examples
    --------
    >>> model = GraphVAE(
    ...     input_dim=2000,
    ...     hidden_dim=128,
    ...     latent_dim=10,
    ...     encoder_type='GAT'
    ... )
    >>> q_z, q_m, q_s, pred_a, pred_x = model(x, edge_index, edge_weight)
    
    Notes
    -----
    The model uses the reparameterization trick for backpropagation through
    the stochastic latent variables. The KL divergence between the learned
    latent distribution and a standard normal prior is used as a regularizer.
    """  

    def __init__(  
        self,  
        input_dim: int,  
        hidden_dim: int,  
        latent_dim: int,  
        encoder_type: str = 'GAT',  
        encoder_hidden_layers: int = 2,  
        decoder_type: str = 'MLP',  
        decoder_hidden_dim: int = 128,  
        feature_decoder_hidden_layers: int = 2,  
        dropout: float = 0.05,  
        use_residual: bool = True,
        Cheb_k: Optional[int] = 1,
        alpha: Optional[float] = .5
    ):  
        super(GraphVAE, self).__init__()  

        # Initialize the graph encoder  
        self.g_encoder = GraphEncoder(  
            input_dim=input_dim,  
            hidden_dim=hidden_dim,  
            latent_dim=latent_dim,  
            conv_layer_type=encoder_type,  
            hidden_layers=encoder_hidden_layers,  
            dropout=dropout,
            Cheb_k=Cheb_k,
            alpha=alpha
        )  

        # Initialize the graph structure decoder based on type
        if decoder_type == 'Bilinear':  
            self.a_decoder = BilinearDecoder(latent_dim)  
        elif decoder_type == 'InnerProduct':  
            self.a_decoder = InnerProductDecoder()  
        elif decoder_type == 'MLP':  
            if decoder_hidden_dim is None:  
                raise ValueError("`decoder_hidden_dim` must be specified for MLPDecoder.")  
            self.a_decoder = MLPDecoder(latent_dim, decoder_hidden_dim)  
        else:  
            raise ValueError(f"Unsupported decoder type: {decoder_type}. Choose from ['Bilinear', 'InnerProduct', 'MLP'].")  

        # Initialize the feature decoder  
        self.x_decoder = FeatureDecoder(  
            input_dim=input_dim,  
            hidden_dim=hidden_dim,  
            latent_dim=latent_dim,  
            hidden_layers=feature_decoder_hidden_layers,  
        )  
        self.use_residual = use_residual
        
    def forward(  
        self,  
        x: torch.Tensor,  
        edge_index: torch.Tensor,  
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  
        """
        Forward pass through the GraphVAE model.
        
        Parameters  
        ----------  
        x : torch.Tensor  
            Node features (gene expression), shape (num_cells, input_dim).
        edge_index : torch.Tensor  
            Edge indices for cell-cell graph, shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights, shape (num_edges,).
            
        Returns  
        -------  
        q_z : torch.Tensor  
            Sampled latent representation, shape (num_cells, latent_dim).
        q_m : torch.Tensor  
            Mean of the latent distribution, shape (num_cells, latent_dim).
        q_s : torch.Tensor  
            Log variance of the latent distribution, shape (num_cells, latent_dim).
        pred_a : torch.Tensor  
            Reconstructed adjacency matrix or edge probabilities,
            shape (num_cells, num_cells).
        pred_x : torch.Tensor  
            Reconstructed gene expression, shape (num_cells, input_dim).
            
        Notes
        -----
        The latent variables are sampled using the reparameterization trick:
        z = μ + σ * ε, where ε ~ N(0, 1)
        This enables backpropagation through the sampling operation.
        """  
        # Step 1: Encode to latent space
        q_z, q_m, q_s = self.g_encoder(x, edge_index, edge_weight, self.use_residual)  

        # Step 2: Decode graph structure
        if isinstance(self.a_decoder, MLPDecoder):  
            # MLPDecoder requires edge_index to know which edges to predict
            pred_a = self.a_decoder(q_z, edge_index)  
        else:  
            # Bilinear and InnerProduct decoders predict all possible edges
            pred_a = self.a_decoder(q_z)  

        # Step 3: Decode gene expression
        pred_x = self.x_decoder(q_z)  

        return q_z, q_m, q_s, pred_a, pred_x


class iGraphVAE(nn.Module):
    """
    Interpretable Variational Graph Autoencoder.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension for the encoder.
    latent_dim : int
        Latent space dimension.
    idim : int
        Interpretable latent space dimension.
    encoder_type : str, optional
        Type of graph convolutional layer, e.g., 'GCN', 'Cheb', 'SAGE', etc.
    encoder_hidden_layers : int, optional
        Number of hidden layers in the graph encoder.
    decoder_type : str, optional
        Type of graph decoder, e.g., 'Bilinear', 'InnerProduct', 'MLP'.
    decoder_hidden_dim : int, optional
        Hidden dimension for the MLPDecoder (if used).
    feature_decoder_hidden_layers : int, optional
        Number of hidden layers in the feature decoder.
    dropout : float, optional
        Dropout rate.
    use_residual : bool, optional
        Whether to use residual connections.
    Cheb_k : int, optional
        The order of Chebyshev polynomials for ChebConv.
    alpha : float, optional
        Teleport probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        idim: int,
        encoder_type: str = 'GAT',
        encoder_hidden_layers: int = 2,
        decoder_type: str = 'MLP',
        decoder_hidden_dim: int = 128,
        feature_decoder_hidden_layers: int = 2,
        dropout: float = 0.05,
        use_residual: bool = True,
        Cheb_k: int = 1,
        alpha: float = 0.5
    ):
        super(iGraphVAE, self).__init__()

        # Initialize the primary graph encoder
        self.g_encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            conv_layer_type=encoder_type,
            hidden_layers=encoder_hidden_layers,
            dropout=dropout,
            Cheb_k=Cheb_k,
            alpha=alpha
        )

        # Initialize the interpretable graph encoder
        self.ig_encoder = FeatureEncoder(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            latent_dim=idim,
            hidden_layers=encoder_hidden_layers
        )

        # Initialize the graph decoders
        if decoder_type == 'Bilinear':
            self.a_decoder = BilinearDecoder(latent_dim)
        elif decoder_type == 'InnerProduct':
            self.a_decoder = InnerProductDecoder()
        elif decoder_type == 'MLP':
            if decoder_hidden_dim is None:
                raise ValueError("`decoder_hidden_dim` must be specified for MLPDecoder.")
            self.a_decoder = MLPDecoder(latent_dim, decoder_hidden_dim)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}. Choose from ['Bilinear', 'InnerProduct', 'MLP'].")

        # Initialize the feature decoders
        self.x_decoder = FeatureDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            hidden_layers=feature_decoder_hidden_layers,
        )
        self.ix_decoder = FeatureDecoder(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            latent_dim=idim,
            hidden_layers=feature_decoder_hidden_layers,
        )

        self.use_residual = use_residual

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass of the Interpretable GraphVAE.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (num_nodes, input_dim).
        edge_index : torch.Tensor
            Edge indices, shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights, shape (num_edges,).

        Returns
        -------
        q_z : torch.Tensor
            Latent representation, shape (num_nodes, latent_dim).
        q_m : torch.Tensor
            Mean of the latent distribution, shape (num_nodes, latent_dim).
        q_s : torch.Tensor
            Standard deviation of the latent distribution, shape (num_nodes, latent_dim).
        pred_a : torch.Tensor
            Reconstructed adjacency matrix or edge probabilities.
        pred_x : torch.Tensor
            Reconstructed node features, shape (num_nodes, input_dim).
        iloss : torch.Tensor
            Loss associated with the interpretable components.
        """
        # Encode the graph to obtain primary latent variables
        q_z, q_m, q_s = self.g_encoder(x, edge_index, edge_weight, self.use_residual)
        device = q_z.device
        
        # Interpretable encoding
        # Create k-nearest neighbor graph from latent representation
        ix = q_z.detach().cpu().numpy()
        knn_graph = kneighbors_graph(ix, n_neighbors=5, mode='connectivity', include_self=False)
        iedge_index_np, iedge_weight_np = knn_graph.nonzero()
        iedge_index = torch.tensor(np.array([iedge_index_np, iedge_weight_np]), dtype=torch.long).to(device)
        iedge_weight = torch.tensor(knn_graph.data, dtype=torch.float).to(device)

        # Interpretable graph encoding
        i_q_z = self.ig_encoder(q_z)

        # Decode the node features
        pred_x = self.x_decoder(q_z)
        
        pred_q_z = self.ix_decoder(i_q_z)
        
        pred_ix = self.x_decoder(pred_q_z)
        
        # Decode the primary graph structure
        if isinstance(self.a_decoder, MLPDecoder):
            # MLPDecoder requires edge_index
            pred_a = self.a_decoder(q_z, edge_index)
            pred_ia = self.a_decoder(pred_q_z, edge_index)
        else:
            # Other decoders do not require edge_index
            pred_a = self.a_decoder(q_z)
            pred_ia = self.a_decoder(pred_q_z)
       
        return q_z, q_m, q_s, pred_a, pred_x, i_q_z, pred_ia, pred_ix

