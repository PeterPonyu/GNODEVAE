import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scanpy as sc
from anndata import AnnData
from copy import deepcopy
from tqdm import tqdm

# --- Core Model Components ---

class _MLPEncoder(nn.Module):
    """Internal MLP encoder for single-cell data."""
    def __init__(self, n_genes, hidden_dim=1024, output_dim=128, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.encoder(x)

class _MoCo(nn.Module):
    """Internal Momentum Contrast model."""
    def __init__(self, n_genes, dim=128, queue_size=1024, momentum=0.999, temperature=0.2):
        super().__init__()
        self.dim = dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        self.encoder_q = _MLPEncoder(n_genes, output_dim=dim)
        self.encoder_k = _MLPEncoder(n_genes, output_dim=dim)
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer("queue", F.normalize(torch.randn(dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if self.queue_size % batch_size == 0:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
    
    def forward(self, query, key):
        q = F.normalize(self.encoder_q(query), dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(key), dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        self._dequeue_and_enqueue(k)
        return logits, labels

    @torch.no_grad()
    def inference(self, x):
        self.encoder_k.eval()
        return F.normalize(self.encoder_k(x), dim=1)

# --- Dataset Class ---

class _CLEAR_Dataset(Dataset):
    """Internal Dataset for CLEAR with augmentations."""
    def __init__(self, adata, aug_prob=0.5):
        self.data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        self.aug_prob = aug_prob
        self.n_cells, self.n_genes = self.data.shape

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        cell_profile = self.data[idx].copy()
        view1 = self._augment(cell_profile)
        view2 = self._augment(cell_profile)
        return torch.from_numpy(view1).float(), torch.from_numpy(view2).float()

    def _augment(self, profile):
        if np.random.rand() < self.aug_prob:
            # Masking
            mask = np.random.choice([True, False], self.n_genes, p=[0.2, 0.8])
            profile[mask] = 0
            # Gaussian Noise
            mask = np.random.choice([True, False], self.n_genes, p=[0.7, 0.3])
            noise = np.random.normal(0, 0.2, np.sum(mask))
            profile[mask] += noise
        return profile

# --- Main CLEAR Class ---

class CLEAR:
    def __init__(self, adata: AnnData, use_highly_variable: bool = True, layer: str = None, latent_dim: int = 128):
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
        
        if self.use_highly_variable:
            if 'highly_variable' not in self.adata.var.keys():
                sc.pp.highly_variable_genes(self.adata)
            self.adata = self.adata[:, self.adata.var['highly_variable']].copy()

    def fit(self, epochs=100, lr=5e-3, batch_size=512, verbose=True):
        dataset = _CLEAR_Dataset(self.adata)
        if len(dataset) < batch_size:
            batch_size = len(dataset)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        self.model = _MoCo(self.adata.n_vars, dim=self.latent_dim, queue_size=batch_size).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=1e-6)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose, miniters=10)
            
            for view1, view2 in progress_bar:
                view1, view2 = view1.to(self.device), view2.to(self.device)
                logits, labels = self.model(view1, view2)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if verbose:
                    progress_bar.set_postfix(loss=loss.item())
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(train_loader):.4f}")
        return self

    def get_embedding(self) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model has not been trained. Please call .fit() first.")

        self.model.eval()
        full_data = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        data_tensor = torch.from_numpy(full_data).float().to(self.device)
        
        with torch.no_grad():
            embedding = self.model.inference(data_tensor).cpu().numpy()
        
        self.adata.obsm['X_clear'] = embedding
        return embedding