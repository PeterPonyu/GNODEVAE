from .GODEVAE_trainer import GODEVAE_Trainer_r
from .mixin import envMixin, scMixin  
import numpy as np  
import torch  
from torch.utils.data import Dataset
from torch_geometric.loader import ClusterData  
from torch_geometric.data import Data, DataLoader  
from sklearn.cluster import KMeans 
from typing import Optional, List, Tuple  
from anndata import AnnData
import scanpy as sc



class SubgraphDataset(Dataset):
    """
    用于子图采样的数据集类 - 返回torch_geometric.data.Data对象
    """
    def __init__(self, 
                 node_features: np.ndarray, 
                 edge_index: np.ndarray,
                 edge_weight: np.ndarray,
                 node_labels: np.ndarray,
                 device: torch.device,
                 subgraph_size: int = 512):
        """
        Parameters
        ----------
        node_features : np.ndarray
            节点特征矩阵
        edge_index : np.ndarray
            边索引 (2, num_edges)
        edge_weight : np.ndarray
            边权重
        node_labels : np.ndarray
            节点标签（y值，通常是节点索引）
        device : torch.device
            设备
        subgraph_size : int
            子图大小（每个batch的节点数）
        """
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.node_labels = node_labels  # 这应该是self.y
        self.device = device
        self.subgraph_size = subgraph_size
        self.num_nodes = node_features.shape[0]
        
        # 预计算每个节点的邻居（用于采样）
        self.neighbors = self._compute_neighbors()
    
    def _compute_neighbors(self):
        """计算每个节点的邻居"""
        neighbors = [[] for _ in range(self.num_nodes)]
        for i, j in self.edge_index.T:
            neighbors[i].append(j)
            if i != j:  # 避免重复添加自环
                neighbors[j].append(i)
        return neighbors
    
    def __len__(self):
        # 返回可以采样的子图数量
        return max(1, self.num_nodes // self.subgraph_size * 2)  # 增加采样次数
    
    def __getitem__(self, idx):
        """
        采样子图并返回torch_geometric.data.Data对象
        """
        # 随机采样节点
        selected_nodes = self._random_node_sampling()
        
        # 创建torch_geometric.data.Data对象
        subgraph_data = self._create_data_object(selected_nodes)
        
        return subgraph_data
    
    def _random_node_sampling(self):
        """随机采样节点"""
        num_sample = min(self.subgraph_size, self.num_nodes)
        selected_nodes = np.random.choice(
            self.num_nodes, 
            size=num_sample, 
            replace=False
        )
        return selected_nodes
    
    def _neighbor_sampling(self, num_seeds=None):
        """基于邻居的采样"""
        if num_seeds is None:
            num_seeds = min(self.subgraph_size // 4, self.num_nodes)
            
        # 选择种子节点
        seed_nodes = np.random.choice(
            self.num_nodes, 
            size=num_seeds, 
            replace=False
        )
        
        selected_nodes = set(seed_nodes)
        
        # 扩展到邻居节点
        for seed in seed_nodes:
            neighbors = self.neighbors[seed]
            if neighbors and len(selected_nodes) < self.subgraph_size:
                # 计算还需要多少邻居
                remaining_slots = self.subgraph_size - len(selected_nodes)
                remaining_seeds = len([s for s in seed_nodes if s not in selected_nodes or s == seed])
                
                if remaining_seeds > 0:
                    num_neighbors = min(
                        len(neighbors), 
                        max(1, remaining_slots // remaining_seeds)
                    )
                    
                    if num_neighbors > 0:
                        selected_neighbors = np.random.choice(
                            neighbors, 
                            size=num_neighbors, 
                            replace=False
                        )
                        selected_nodes.update(selected_neighbors)
        
        return np.array(list(selected_nodes))
    
    def _create_data_object(self, selected_nodes):
        """
        创建torch_geometric.data.Data对象
        这个对象必须与GODEVAE_Trainer_r期望的格式一致
        """
        # 创建节点映射：原始索引 -> 新索引
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}
        
        # 提取子图的边
        edge_mask = np.isin(self.edge_index[0], selected_nodes) & np.isin(self.edge_index[1], selected_nodes)
        subgraph_edges = self.edge_index[:, edge_mask]
        subgraph_weights = self.edge_weight[edge_mask]
        
        # 重新映射边索引到新的节点索引
        new_edge_index = np.array([
            [node_map[i] for i in subgraph_edges[0]],
            [node_map[i] for i in subgraph_edges[1]]
        ])
        
        # 提取节点特征
        subgraph_features = self.node_features[selected_nodes]
        
        # 提取节点标签（重新映射到子图的节点索引）
        subgraph_y = np.array([node_map[original_idx] for original_idx in selected_nodes])
        
        # 创建Data对象 - 确保属性名与原始代码一致
        data = Data(
            x=torch.tensor(subgraph_features, dtype=torch.float, device=self.device),
            edge_index=torch.tensor(new_edge_index, dtype=torch.long, device=self.device),
            edge_attr=torch.tensor(subgraph_weights, dtype=torch.float, device=self.device),
            y=torch.tensor(subgraph_y, dtype=torch.long, device=self.device)
        )
        
        # 添加原始节点索引信息（用于后续重构）
        data.original_node_idx = torch.tensor(selected_nodes, dtype=torch.long, device=self.device)
        
        return data

class GNODEVAE_Env_Subgraph(GODEVAE_Trainer_r, envMixin, scMixin):
    """
    使用子图采样的环境类 - 与原始API完全一致
    """
    
    def __init__(self, 
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
                 w_recon: float,
                 w_kl: float,
                 w_adj: float,
                 w_recon_ode: float,
                 w_z_div: float,
                 device: torch.device,
                 latent_type: str,
                 # 子图采样特有参数
                 subgraph_size: int,
                 num_subgraphs_per_epoch: int,
                 sampling_method: str,  # 'random' or 'neighbor'
                 *args,
                 **kwargs):
        
        # 首先注册AnnData并处理数据
        self._register_adata(adata, layer, n_var, tech, n_neighbors, latent_dim, batch_tech, all_feat)
        
        # 初始化父类GODEVAE_Trainer_r
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
            w_recon,
            w_kl,
            w_adj,
            w_recon_ode,
            w_z_div,
            device,
            latent_type,
        )
        
        # 注册子图采样数据
        self._register_subgraph_data(subgraph_size, num_subgraphs_per_epoch, sampling_method)
        
        # 初始化评分列表
        self.score: List[Tuple[float, float, float, float, float, float]] = []
        self.mix_score: List[Tuple[float, float, float, float, float, float]] = []
    
    def _register_adata(self, 
                       adata: AnnData,
                       layer: str,
                       n_var: int,
                       tech: str,
                       n_neighbors: int,
                       latent_dim: int,
                       batch_tech: Optional[str],
                       all_feat: bool) -> None:
        """
        注册AnnData对象并预处理数据 - 与原版本完全一致
        """
        self._preprocess(adata, layer, n_var)
        self._decomposition(adata, tech, latent_dim)
        
        if batch_tech:
            self._batchcorrect(adata, batch_tech, tech, layer)
        
        # 确定用于邻域图的表示
        if batch_tech == 'harmony':
            use_rep = f'X_harmony_{tech}'
        elif batch_tech == 'scvi':
            use_rep = 'X_scvi'
        else:
            use_rep = f'X_{tech}'
        
        # 构建邻域图
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
        
        # 选择特征
        if all_feat:
            self.X = np.log1p(adata.layers[layer].toarray())
        else:
            self.X = adata[:, adata.var['highly_variable']].X.toarray()
        
        self.n_obs, self.n_var = self.X.shape
        
        # 执行聚类
        self.labels = KMeans(n_clusters=latent_dim).fit_predict(self.X)
        
        # 提取图信息
        coo = adata.obsp['connectivities'].tocoo()
        self.edge_index = np.array([coo.row, coo.col])
        self.edge_weight = coo.data
        
        # 分配索引 - 这个y就是节点索引
        self.y = np.arange(adata.shape[0])
        self.idx = np.arange(adata.shape[0])
        
    def _register_subgraph_data(self, 
                               subgraph_size: int,
                               num_subgraphs_per_epoch: int,
                               sampling_method: str):
        """
        注册子图采样数据
        """
        self.subgraph_size = subgraph_size
        self.num_subgraphs_per_epoch = num_subgraphs_per_epoch
        self.sampling_method = sampling_method
        
        # 创建子图数据集
        self.subgraph_dataset = SubgraphDataset(
            node_features=self.X,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            node_labels=self.y,  # 节点索引
            device=self.device,
            subgraph_size=subgraph_size
        )
        
        # 创建数据加载器
        self.subgraph_loader = DataLoader(
            self.subgraph_dataset,
            batch_size=1,  # 每次处理一个子图
            shuffle=True,
            num_workers=0
        )
        
        # 为了与原版本一致，我们也创建一个cdata-like的接口
        self.cdata = self._create_cdata_interface()
    
    def _create_cdata_interface(self):
        """
        创建一个类似cdata的接口，用于与原始API保持一致
        """
        class SubgraphIterator:
            def __init__(self, subgraph_loader, num_subgraphs_per_epoch):
                self.subgraph_loader = subgraph_loader
                self.num_subgraphs_per_epoch = num_subgraphs_per_epoch
                self._iterator = None
                self._count = 0
            
            def __iter__(self):
                self._iterator = iter(self.subgraph_loader)
                self._count = 0
                return self
            
            def __next__(self):
                if self._count >= self.num_subgraphs_per_epoch:
                    raise StopIteration
                
                try:
                    # DataLoader返回的是[batch]，我们需要取第一个元素
                    batch = next(self._iterator)
                    if isinstance(batch, list) and len(batch) > 0:
                        data = batch[0]  # 取第一个Data对象
                    else:
                        data = batch
                    
                    self._count += 1
                    return data
                except StopIteration:
                    raise StopIteration
        
        return SubgraphIterator(self.subgraph_loader, self.num_subgraphs_per_epoch)
    
    def step(self) -> None:
        """
        执行一个训练步骤 - 与原版本API完全一致
        """
        ls_l = []
        original_indices = []
        
        # 这里的逻辑与原版本完全一致：for cd in self.cdata
        for cd in self.cdata:
            # cd现在是一个torch_geometric.data.Data对象
            self.update(cd)  # 调用父类的update方法
            latent = self.take_latent(cd)  # 调用父类的take_latent方法
            ls_l.append(latent)
            
            # 保存原始节点索引用于重构
            if hasattr(cd, 'original_node_idx'):
                original_indices.append(cd.original_node_idx.cpu().numpy())
        if original_indices:
            self.idx = np.hstack(original_indices)
        if ls_l:
            # 重构完整的潜在表示
            if original_indices:
                full_latent = self._reconstruct_full_latent(ls_l, original_indices)
            else:
                # 如果没有索引信息，简单拼接（可能不完整）
                full_latent = np.vstack(ls_l)
            
            # 计算评分 - 与原版本一致
            score = self._calc_score(full_latent)
            self.score.append(score)
    
    def step_ode(self) -> None:
        """
        ODE版本的训练步骤 - 与原版本API完全一致
        """
        ls_mix_latent1 = []
        ls_mix_latent2 = []
        original_indices = []
        
        for cd in self.cdata:
            self.update(cd)  # 调用父类的update方法
            latent1 = self.take_latent(cd)  # 调用父类的take_latent方法
            latent2 = self.take_odelatent(cd)  # 调用父类的take_odelatent方法
            
            ls_mix_latent1.append(latent1)
            ls_mix_latent2.append(latent2)
            
            if hasattr(cd, 'original_node_idx'):
                original_indices.append(cd.original_node_idx.cpu().numpy())
        if original_indices:
            self.idx = np.hstack(original_indices)
        if ls_mix_latent1 and ls_mix_latent2:
            # 重构完整的潜在表示
            if original_indices:
                full_latent1 = self._reconstruct_full_latent(ls_mix_latent1, original_indices)
                full_latent2 = self._reconstruct_full_latent(ls_mix_latent2, original_indices)
            else:
                full_latent1 = np.vstack(ls_mix_latent1)
                full_latent2 = np.vstack(ls_mix_latent2)
            
            # 混合潜在表示 - 与原版本一致
            latent_mix = self.scale1 * full_latent1 + self.scale2 * full_latent2
            
            score_mix = self._calc_score(latent_mix)
            self.mix_score.append(score_mix)
    
    def _reconstruct_full_latent(self, latent_list, indices_list):
        """
        从子图的潜在表示重构完整的潜在表示
        """
        if not latent_list:
            return np.array([])
        
        # 初始化完整的潜在表示矩阵
        latent_dim = latent_list[0].shape[1]
        full_latent = np.zeros((self.n_obs, latent_dim))
        node_counts = np.zeros(self.n_obs)
        
        # 累加所有子图的贡献
        for latent, indices in zip(latent_list, indices_list):
            for i, original_node_idx in enumerate(indices):
                if 0 <= original_node_idx < self.n_obs:
                    full_latent[original_node_idx] += latent[i]
                    node_counts[original_node_idx] += 1
        
        # 对被多次访问的节点取平均
        for i in range(self.n_obs):
            if node_counts[i] > 0:
                full_latent[i] /= node_counts[i]
            # 如果某个节点从未被访问，保持零向量（或者可以用其他策略）
        
        return full_latent




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
        w_recon: float,
        w_kl: float ,
        w_adj: float,
        w_recon_ode: float,
        w_z_div: float,
        device: torch.device,  
        num_parts: int,  
        latent_type: str,
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
            w_recon,
            w_kl,
            w_adj,
            w_recon_ode,
            w_z_div,
            device,
            latent_type,
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
