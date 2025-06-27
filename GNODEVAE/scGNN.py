import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from tqdm import tqdm

# --- Core Model Components ---

class _VAE(nn.Module):
    """Internal Variational Autoencoder."""
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 400), nn.ReLU())
        self.mu = nn.Linear(400, 20)
        self.logvar = nn.Linear(400, 20)
        self.decoder = nn.Sequential(nn.Linear(20, 400), nn.ReLU(), nn.Linear(400, dim), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z

# --- Dataset Class ---

class _scGNNDataset(Dataset):
    """Internal Dataset for scGNN."""
    def __init__(self, adata):
        self.features = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx].flatten()).float(), idx

# --- Main scGNN Class ---

class scGNN:
    def __init__(self, adata: AnnData, use_highly_variable: bool = True, layer: str = None, latent_dim: int = 20):
        self.adata = adata.copy()
        self.use_highly_variable = use_highly_variable
        self.layer = layer
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._prepare_data()

    def _prepare_data(self):
        if self.layer and self.layer in self.adata.layers:
            self.adata.X = self.adata.layers[self.layer]
        
        # Preprocessing: normalize and log-transform
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)

        if self.use_highly_variable:
            if 'highly_variable' not in self.adata.var.keys():
                sc.pp.highly_variable_genes(self.adata)
            self.adata = self.adata[:, self.adata.var['highly_variable']].copy()
        
        # Check for NaN values in the input data after preprocessing
        if np.isnan(self.adata.X.toarray()).any():
            raise ValueError("Input AnnData.X contains NaN values after preprocessing. Please handle missing data.")

    def _build_graph_regu(self, features, k=10):
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(features)
        distances, indices = nbrs.kneighbors(features)
        adj = np.zeros((features.shape[0], features.shape[0]))
        for i in range(len(features)):
            for j_idx, j in enumerate(indices[i]):
                if i != j:
                    adj[i, j] = 1.0 / (1.0 + distances[i, j_idx])
        return torch.from_numpy(adj).float().to(self.device)

    def _loss_function(self, recon_x, x, mu, logvar, graph_regu, gamma, regu_para):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        graph_loss = torch.mean(torch.matmul(graph_regu, (recon_x - x)**2))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return gamma * recon_loss + regu_para * graph_loss + kld_loss

    def fit(self, regu_epochs=200, em_epochs=100, em_iterations=5, lr=1e-3, k=10, gamma=0.1, regu_para=0.001, verbose=True):
        dataset = _scGNNDataset(self.adata)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        self.model = _VAE(self.adata.n_vars).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Initial training with graph regularization
        initial_graph_regu = self._build_graph_regu(dataset.features, k=k)
        for epoch in tqdm(range(regu_epochs), desc="Initial Graph-VAE Training", disable=not verbose, miniters=10):
            self._train_epoch(train_loader, optimizer, initial_graph_regu, gamma, regu_para)

        # EM-like iterations
        for i in range(em_iterations):
            embeddings = self.get_embedding(internal_call=True)
            graph_regu = self._build_graph_regu(embeddings, k=k)
            for epoch in tqdm(range(em_epochs), desc=f"EM Iteration {i+1}/{em_iterations}", disable=not verbose, miniters=10):
                self._train_epoch(train_loader, optimizer, graph_regu, gamma, regu_para)
        
        return self

    def _train_epoch(self, loader, optimizer, graph_regu, gamma, regu_para):
        self.model.train()
        for data, idx in loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            recon, mu, logvar, _ = self.model(data)
            loss = self._loss_function(recon, data, mu, logvar, graph_regu[idx, :][:, idx], gamma, regu_para)
            loss.backward()
            optimizer.step()

    def get_embedding(self, internal_call=False) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model has not been trained. Please call .fit() first.")

        self.model.eval()
        embeddings = []
        full_data = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        data_loader = DataLoader(_scGNNDataset(self.adata), batch_size=128)

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                _, mu, _, _ = self.model(data)
                embeddings.append(mu.cpu().numpy())
        
        embedding_matrix = np.vstack(embeddings)
        # Check for NaN values in the generated embeddings
        if np.isnan(embedding_matrix).any():
            raise ValueError("Generated embeddings contain NaN values. This indicates a potential issue with the model training or architecture.")
        if not internal_call:
            self.adata.obsm['X_scgnn'] = embedding_matrix
        return embedding_matrix