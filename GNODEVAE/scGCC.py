import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader # Added import
from anndata import AnnData
import scanpy as sc
import numpy as np
import warnings
from tqdm import tqdm

# --- Helper Functions and Classes ---

def _full_block(in_features, out_features, p_drop=0.0):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )

# --- Core Model Components ---

class GATEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, num_heads=4, dropout=0.4, use_mlp=False):
        super().__init__()
        self.gat_layer_1 = GATConv(in_channels, 128, heads=num_heads, dropout=dropout, concat=True)
        self.gat_layer_2 = GATConv(128 * num_heads, latent_dim, heads=num_heads, dropout=dropout, concat=False)
        self.fc = nn.Sequential(_full_block(latent_dim, 512, 0.4), _full_block(512, latent_dim)) if use_mlp else None

    def forward(self, x, edge_index):
        x = F.relu(self.gat_layer_1(x, edge_index))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.gat_layer_2(x, edge_index)
        return self.fc(x) if self.fc else x

class MoCo(nn.Module):
    def __init__(self, base_encoder, num_genes, latent_dim, r=1024, m=0.99, T=0.2, heads=4, mlp=False):
        super().__init__()
        self.r = r
        self.m = m
        self.T = T
        self.encoder_q = base_encoder(num_genes, latent_dim, num_heads=heads, use_mlp=mlp)
        self.encoder_k = base_encoder(num_genes, latent_dim, num_heads=heads, use_mlp=mlp)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(latent_dim, r), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if self.r % batch_size == 0:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr[0] = (ptr + batch_size) % self.r

    def forward(self, im_q, im_k, edge_index, num_seed_nodes):
        # Encode the full subgraph features
        q_all = self.encoder_q(im_q, edge_index)
        q_seed = F.normalize(q_all[:num_seed_nodes], dim=1) # Extract and normalize seed node features
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_all = self.encoder_k(im_k, edge_index)
            k_seed = F.normalize(k_all[:num_seed_nodes], dim=1) # Extract and normalize seed node features

        l_pos = torch.einsum('nc,nc->n', [q_seed, k_seed]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_seed, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q_seed.device)
        self._dequeue_and_enqueue(k_seed)
        return logits, labels

    @torch.no_grad()
    def inference(self, x, edge_index):
        self.encoder_k.eval()
        return self.encoder_k(x, edge_index)

# --- Main scGCC Class ---

class scGCC:
    def __init__(self, adata: AnnData, use_highly_variable: bool = True, layer: str = None, latent_dim: int = 256):
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

    def _build_graph(self, k=10):
        sc.pp.neighbors(self.adata, n_neighbors=k, use_rep='X')
        edge_index = torch.from_numpy(np.vstack(self.adata.obsp['connectivities'].nonzero())).to(torch.long)
        return Data(x=torch.from_numpy(self.adata.X.toarray()).float(), edge_index=to_undirected(edge_index))

    def _augment_batch(self, x_batch, aug_prob=0.5):
        # Apply masking
        if np.random.rand() < aug_prob:
            mask = torch.rand(x_batch.shape, device=x_batch.device) < 0.2 # 20% masking
            x_batch[mask] = 0

        # Apply Gaussian Noise
        if np.random.rand() < aug_prob:
            mask = torch.rand(x_batch.shape, device=x_batch.device) < 0.3 # 30% noise
            noise = torch.randn_like(x_batch) * 0.2 # Gaussian noise with std 0.2
            x_batch[mask] += noise[mask]
        return x_batch

    def fit(self, epochs=20, lr=0.1, batch_size=512, verbose=True):
        global_graph = self._build_graph().to(self.device)
        
        # Use NeighborLoader to create batches of subgraphs
        train_loader = NeighborLoader(
            global_graph,
            num_neighbors=[-1],  # Sample all neighbors for full graph, or specify for subgraphs
            batch_size=batch_size,
            shuffle=True,
            num_workers=0, # For simplicity, no multiprocessing for now
        )

        self.model = MoCo(GATEncoder, self.adata.n_vars, self.latent_dim, r=batch_size).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose, miniters=10)
            for data in progress_bar: # data is now a Data object from NeighborLoader
                data = data.to(self.device)
                
                im_q = data.x # Original features for the batch
                im_k = self._augment_batch(data.x.clone().detach()) # Augmented features for the batch
                
                logits, labels = self.model(im_q, im_k, data.edge_index, data.batch_size)
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

    def get_embedding(self):
        if not self.model:
            raise RuntimeError("Model has not been trained. Please call .fit() first.")

        self.model.eval()
        global_graph = self._build_graph().to(self.device)
        
        with torch.no_grad():
            full_data = torch.from_numpy(self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X).float().to(self.device)
            embedding = self.model.inference(full_data, global_graph.edge_index).cpu().numpy()
        
        self.adata.obsm['X_scgcc'] = embedding
        return embedding