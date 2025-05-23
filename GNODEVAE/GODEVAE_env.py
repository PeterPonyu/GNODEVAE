from .GODEVAE_trainer import GODEVAE_Trainer_r
from .mixin import envMixin, scMixin  
import numpy as np  
import torch  
from torch_geometric.loader import ClusterData  
from torch_geometric.data import Data  
from sklearn.cluster import KMeans 
from typing import Optional, List, Tuple  
from anndata import AnnData
import scanpy as sc

class GNODEVAE_Env_r(GODEVAE_Trainer_r, envMixin, scMixin):  
    """  
    Environment class for training and evaluating the GraphVAE model.  

    This class extends the `Trainer` and `envMixin` classes to provide functionality  
    for handling AnnData objects, clustering, and training the model in a batched manner  
    using `ClusterData`.  

    Parameters  
    ----------  
    adata : AnnData  
        Annotated data matrix (e.g., single-cell data).  
    layer : str  
        Layer of the AnnData object to use for input features.  
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
    hidden_dim : int  
        Hidden layer dimension for the encoder.  
    latent_dim : int  
        Latent space dimension.  
    encoder_type : str  
        Type of graph convolutional layer ('GCN', 'Cheb', 'SAGE', 'Graph', 'TAG', 'ARMA', 'GAT', 'Transformer', 'SG', 'SSG').  
    encoder_hidden_layers : int  
        Number of hidden layers in the graph encoder.  
    decoder_type : str  
        Type of graph decoder ('Bilinear', 'InnerProduct', 'MLP').  
    decoder_hidden_dim : int  
        Hidden dimension for the MLPDecoder (if used).  
    feature_decoder_hidden_layers : int  
        Number of hidden layers in the feature decoder.  
    dropout : float  
        Dropout rate.
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
    num_parts : int  
        Number of partitions for clustering the graph data.  
    *args : tuple  
        Additional positional arguments.  
    **kwargs : dict  
        Additional keyword arguments.  

    Attributes  
    ----------  
    X : np.ndarray  
        Input feature matrix (log-transformed).  
    n_obs : int  
        Number of observations (nodes).  
    n_var : int  
        Number of variables (features).  
    labels : np.ndarray  
        Cluster labels for the input data.  
    edge_index : np.ndarray  
        Edge indices of the graph.  
    edge_weight : np.ndarray  
        Edge weights of the graph.  
    y : np.ndarray  
        Node indices.  
    cdata : ClusterData  
        Clustered graph data for batched training.  
    idx : np.ndarray  
        Node indices for all clusters.  
    score : List[Tuple[float, float, float, float, float, float]]  
        List of clustering and correlation scores for each training step.  
    """  

    def __init__(  
        self,  
        adata: AnnData,  
        layer: str,
        n_var: int,  
        tech: str,  
        n_neighbors: int,      
        batch_tech: Optional[str],
        all_feat: bool,
        hidden_dim: int,  
        latent_dim: int,  
        ode_hidden_dim: int,
        encoder_type: str,  
        graph_type: str,  
        structure_decoder_type: str,  
        feature_decoder_type: str,  
        hidden_layers: int,  
        decoder_hidden_dim: int,  
        dropout: float,  
        use_residual: bool,  
        Cheb_k: int,  
        alpha: float, 
        threshold: float,
        sparse_threshold: Optional[int],
        lr: float,  
        beta: float,  
        graph: float,  
        device: torch.device,  
        num_parts: int,  
        *args,  
        **kwargs,  
    ):  
        self._register_adata(adata, layer, n_var, tech, n_neighbors, latent_dim, batch_tech, all_feat)  
        super().__init__(
            self.n_var,  
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
            lr,  
            beta,  
            graph,  
            device,  
        )  
        self._register_cdata(num_parts)  
        self.score: List[Tuple[float, float, float, float, float, float]] = []  
        self.mix_score: List[Tuple[float, float, float, float, float, float]] = []
        
    def step(self) -> None:  
        """  
        Perform a single training step.  

        This method iterates over the clustered data, updates the model parameters,  
        computes the latent representation, and calculates clustering and correlation scores.  

        Returns  
        -------  
        None  
        """  
        ls_l = []  
        for cd in self.cdata:  
            self.update(cd)  
            latent = self.take_latent(cd)  
            ls_l.append(latent)  
        latent = np.vstack(ls_l)  
        score = self._calc_score(latent)  
        self.score.append(score)  
    
    def step_ode(self) -> None:
        ls_mix = []
        for cd in self.cdata:  
            self.update(cd)  
            latent1 = self.take_latent(cd) 
            latent2 = self.take_odelatent(cd)
            latent = self.scale1 * latent1 + self.scale2 * latent2
            ls_mix.append(latent)
            
        latent_mix = np.vstack(ls_mix)  
        
        score_mix = self._calc_score(latent_mix)
        
        self.mix_score.append(score_mix)

    def _register_adata(  
        self,  
        adata: AnnData,  
        layer: str,  
        n_var: int,  
        tech: str,  
        n_neighbors: int,  
        latent_dim: int,  
        batch_tech: Optional[str],  
        all_feat: bool  
    ) -> None:  
        """  
        Register AnnData object and preprocess the data.  
    
        This method extracts the input features, graph structure, and cluster labels  
        from the AnnData object.  
    
        Parameters  
        ----------  
        adata : AnnData  
            Annotated data matrix (e.g., single-cell data).  
        layer : str  
            Layer of the AnnData object to store original features.  
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
            Whether to use all features or only highly variable ones.  
    
        Returns  
        -------  
        None  
        """  
        self._preprocess(adata, layer, n_var)  
        self._decomposition(adata, tech, latent_dim)  
    
        if batch_tech:  
            self._batchcorrect(adata, batch_tech, tech, layer)  
    
        # Determine the representation to use for neighborhood graph  
        if batch_tech == 'harmony':  
            use_rep = f'X_harmony_{tech}'  
        elif batch_tech == 'scvi':  
            use_rep = 'X_scvi'  
        else:  
            use_rep = f'X_{tech}'  
    
        # Construct the neighborhood graph  
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)  
    
        # Select features  
        if all_feat:  
            self.X = np.log1p(adata.layers[layer].toarray())  
        else:  
            self.X = adata[:, adata.var['highly_variable']].X.toarray()  
    
        self.n_obs, self.n_var = self.X.shape  
    
        # Perform clustering  
        self.labels = KMeans(n_clusters=latent_dim).fit_predict(self.X)  
    
        # Extract graph information  
        coo = adata.obsp['connectivities'].tocoo()  
        self.edge_index = np.array([coo.row, coo.col])  
        self.edge_weight = coo.data  
    
        # Assign indices  
        self.y = np.arange(adata.shape[0]) 
        
    def _register_cdata(  
        self,  
        num_parts: int,  
    ) -> None:  
        """  
        Register clustered graph data for batched training.  

        This method partitions the graph data into clusters for efficient training.  

        Parameters  
        ----------  
        num_parts : int  
            Number of partitions for clustering the graph data.  

        Returns  
        -------  
        None  
        """  
        data = Data(  
            x=torch.tensor(self.X, dtype=torch.float, device=self.device),  
            edge_index=torch.tensor(self.edge_index, dtype=torch.long, device=self.device),  
            edge_attr=torch.tensor(self.edge_weight, dtype=torch.float, device=self.device),  
            y=torch.tensor(self.y, dtype=torch.long, device=self.device),  
        )  
        self.cdata = ClusterData(data, num_parts=num_parts)  
        ls_y = []  
        for cd in self.cdata:  
            ls_y.append(cd.y.cpu().numpy())  
        self.idx = np.hstack(ls_y)