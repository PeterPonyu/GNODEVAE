import torch  
import torch.nn as nn  
from typing import Tuple, Union, Optional  
from torch_sparse import SparseTensor  
import numpy as np  

class AdjToEdge:  
    """  
    Convert adjacency matrix to edge indices and weights for graph decoders.  

    Parameters  
    ----------  
    threshold : float, optional  
        Probability threshold for edge existence, by default 0  
    sparse_threshold : int, optional  
        Maximum number of edges to keep per node, by default None  
    symmetric : bool, optional  
        Whether to ensure symmetric edges, by default True  
    add_self_loops : bool, optional  
        Whether to add self-loops, by default False  
    """  
    def __init__(  
        self,  
        threshold: float = 0,  
        sparse_threshold: Optional[int] = None,  
        symmetric: bool = True,  
        add_self_loops: bool = False  
    ):  
        self.threshold = threshold  
        self.sparse_threshold = sparse_threshold  
        self.symmetric = symmetric  
        self.add_self_loops = add_self_loops  

    def _sparsify(  
        self,  
        adj: np.ndarray,  
        k: int  
    ) -> np.ndarray:  
        """Keep top-k edges per node."""  
        sparse_adj = np.zeros_like(adj)  
        for i in range(adj.shape[0]):  
            top_k_idx = np.argpartition(adj[i], -k)[-k:]  
            mask = adj[i, top_k_idx] > self.threshold  
            sparse_adj[i, top_k_idx] = adj[i, top_k_idx] * mask  
        return sparse_adj  

    def _symmetrize(  
        self,  
        edge_index: np.ndarray,  
        edge_weight: np.ndarray  
    ) -> Tuple[np.ndarray, np.ndarray]:  
        """Make edges symmetric by averaging weights of bidirectional edges."""  
        if edge_index.size == 0 or edge_weight.size == 0:  
            return np.zeros((2, 0), dtype=np.int64), np.array([], dtype=edge_weight.dtype)
        n = max(edge_index[0].max(), edge_index[1].max()) + 1  
        adj = np.zeros((n, n))  
        adj[edge_index[0], edge_index[1]] = edge_weight  
        
        # Symmetrize the adjacency matrix  
        adj = (adj + adj.T) / 2  
        
        # Convert back to edge index and weight format  
        rows, cols = np.nonzero(adj)  
        edge_index = np.stack([rows, cols])  
        edge_weight = adj[rows, cols]  
        
        return edge_index, edge_weight  

    def _add_self_loops(  
        self,  
        edge_index: np.ndarray,  
        edge_weight: np.ndarray,  
        num_nodes: int  
    ) -> Tuple[np.ndarray, np.ndarray]:  
        """Add self-loops to the graph."""  
        self_loops = np.arange(num_nodes)  
        self_loops = np.stack([self_loops, self_loops])  
        
        edge_index = np.concatenate([edge_index, self_loops], axis=1)  
        edge_weight = np.concatenate([edge_weight, np.ones(num_nodes)])  
        
        return edge_index, edge_weight  

    def convert(  
        self,  
        adj: np.ndarray  
    ) -> Tuple[np.ndarray, np.ndarray]:  
        """  
        Convert adjacency matrix to edge index and weights.  

        Parameters  
        ----------  
        adj : np.ndarray  
            Adjacency matrix with edge probabilities (num_nodes, num_nodes)  

        Returns  
        -------  
        edge_index : np.ndarray  
            Edge indices (2, num_edges)  
        edge_weight : np.ndarray  
            Edge weights (num_edges,)  
        """  
        # Sparsify if needed  
        if self.sparse_threshold is not None:  
            adj = self._sparsify(adj, self.sparse_threshold)  

        # Get edges above threshold  
        mask = adj > self.threshold  
        rows, cols = np.nonzero(mask)  
        edge_index = np.stack([rows, cols])  
        edge_weight = adj[rows, cols]  

        # Symmetrize if needed  
        if self.symmetric:  
            edge_index, edge_weight = self._symmetrize(edge_index, edge_weight)  

        # Add self-loops if needed  
        if self.add_self_loops:  
            edge_index, edge_weight = self._add_self_loops(  
                edge_index, edge_weight, adj.shape[0]  
            )  

        return edge_index, edge_weight  


class GraphStructureDecoder(nn.Module):  
    """  
    Combines structure decoder with edge conversion for graph reconstruction.  
    
    Parameters  
    ----------  
    structure_decoder : str  
        Type of structure decoder ('bilinear', 'inner_product', or 'mlp')  
    latent_dim : int  
        Latent space dimension  
    hidden_dim : int, optional  
        Hidden dimension for MLP decoder, by default None  
    threshold : float, optional  
        Edge probability threshold, by default 0.5  
    sparse_threshold : int, optional  
        Maximum edges per node, by default None  
    symmetric : bool, optional  
        Whether to ensure symmetric edges, by default True  
    add_self_loops : bool, optional  
        Whether to add self-loops, by default False  
    """  
    def __init__(  
        self,  
        structure_decoder: str,  
        latent_dim: int,  
        hidden_dim: Optional[int] = None,  
        threshold: float = 0.5,  
        sparse_threshold: Optional[int] = None,  
        symmetric: bool = True,  
        add_self_loops: bool = False  
    ):  
        super().__init__()
        # Initialize structure decoder  
        if structure_decoder == 'bilinear':  
            self.decoder = BilinearDecoder(latent_dim)  
        elif structure_decoder == 'inner_product':  
            self.decoder = InnerProductDecoder()  
        elif structure_decoder == 'mlp':  
            if hidden_dim is None:  
                raise ValueError("hidden_dim must be specified for MLP decoder")  
            self.decoder = MLPDecoder(latent_dim, hidden_dim)  
        else:  
            raise ValueError(  
                f"Unknown decoder type: {structure_decoder}. "  
                "Choose from: bilinear, inner_product, mlp"  
            )  
        
        # Initialize edge converter  
        self.edge_converter = AdjToEdge(  
            threshold=threshold,  
            sparse_threshold=sparse_threshold,  
            symmetric=symmetric,  
            add_self_loops=add_self_loops  
        )  
        
        self.structure_decoder = structure_decoder  
    
    def forward(  
        self,  
        z: torch.Tensor,  
        edge_index: Optional[torch.Tensor] = None  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        """  
        Convert latent embeddings to edge index and weights.  
        
        Parameters  
        ----------  
        z : torch.Tensor  
            Latent node embeddings (num_nodes, latent_dim)  
        edge_index : torch.Tensor, optional  
            Edge indices for MLP decoder (2, num_edges)  
            
        Returns  
        -------  
        adj : torch.Tensor  
            Adjacency matrix (num_nodes, num_nodes)  
        edge_index : torch.Tensor  
            Edge indices (2, num_edges)  
        edge_weight : torch.Tensor  
            Edge weights (num_edges,)  
        """  
        # Get adjacency matrix from structure decoder  
        if self.structure_decoder == 'mlp':  
            if edge_index is None:  
                raise ValueError("edge_index required for MLP decoder")  
            adj = self.decoder(z, edge_index)  
        else:  
            adj = self.decoder(z)  
        
        # Convert torch tensor to numpy array  
        adj_np = adj.detach().cpu().numpy()  
        
        # Convert to edge index and weights using numpy-based edge converter  
        edge_index_np, edge_weight_np = self.edge_converter.convert(adj_np)  
        
        # Convert back to torch tensors  
        device = z.device  
        edge_index = torch.from_numpy(edge_index_np).to(device)  
        edge_weight = torch.from_numpy(edge_weight_np).to(device)  
        
        return adj, edge_index, edge_weight  
        

class BilinearDecoder(nn.Module):  
    """  
    Bilinear Decoder for graph reconstruction.  

    Parameters  
    ----------  
    latent_dim : int  
        Latent space dimension.  
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
            Reconstructed adjacency matrix, shape (num_nodes, num_nodes).  
        """  
        adj_recon = torch.sigmoid(torch.matmul(z @ self.weight, z.t()))  
        return adj_recon  


class InnerProductDecoder(nn.Module):  
    """  
    Inner Product Decoder for graph reconstruction.  
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
            Reconstructed adjacency matrix, shape (num_nodes, num_nodes).  
        """  
        adj_recon = torch.sigmoid(torch.matmul(z, z.t()))  
        return adj_recon  


class MLPDecoder(nn.Module):  
    """  
    MLP Decoder for graph reconstruction.  

    Parameters  
    ----------  
    latent_dim : int  
        Latent space dimension.  
    hidden_dim : int  
        Hidden layer dimension.  
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
            Edge indices, shape (2, num_edges).  

        Returns  
        -------  
        adj_recon : torch.Tensor  
            Reconstructed adjacency matrix, shape (num_nodes, num_nodes).  
        """  
        num_nodes = z.size(0)  
        row, col = edge_index  
        edge_features = torch.cat([z[row], z[col]], dim=1)  
        edge_probs = self.mlp(edge_features).squeeze()  
        adj_recon = torch.zeros((num_nodes, num_nodes), device=z.device)  
        adj_recon[row, col] = edge_probs  
        return adj_recon  
       