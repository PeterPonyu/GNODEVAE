# GNODEVAE: A Graph-Based ODE-VAE Enhances Clustering of Single-Cell Data

## Graphical Abstract

![GNODEVAE Graphical Abstract](gnodevae_abs.jpeg)

## Introduction

GNODEVAE is an innovative computational framework that integrates Graph Attention Networks (GAT), Neural Ordinary Differential Equations (NODE), and Variational Autoencoders (VAE). It addresses three critical challenges in single-cell RNA sequencing data analysis:

1. Capturing complex topological relationships between cells
2. Modeling continuous dynamic processes of cell differentiation
3. Handling high levels of technical noise and biological variation

This novel integration significantly improves the accurate identification of cell subpopulations, reconstruction of developmental trajectories, and understanding of cellular heterogeneity.

## Key Contributions

### 1. Dynamic Attention Weighting for Biological Significance

The GAT's attention mechanism adaptively weights gene expression profiles, prioritizing meaningful biological relationships while minimizing technical noise - particularly valuable for heterogeneous cell populations.

### 2. Continuous-Time Developmental Modeling via Neural ODEs

Integration of neural ordinary differential equations transforms static representations into dynamic systems, with time variables providing natural parameterization of developmental processes and enabling predictions at any point in cellular development.

### 3. Biologically Consistent Latent Space Representations

The model's latent space effectively captures biological phenomena like varying rates of cell differentiation, while attention weights align with established developmental relationships between cell types.

### 4. Comprehensive Benchmark Leadership

When compared with six advanced single-cell analysis methods (scVI, DIP-VAE, TC-VAE, β-VAE, Info-VAE, and scTour), GNODEVAE ranked first across all 13 test datasets, demonstrating exceptional versatility across diverse biological contexts.

### 5. Superior Gene Trend Analysis Performance

Quantitative evaluation shows GNODEVAE significantly outperforms existing methods (69.97% improvement over Palantir, 63.58% over Diffmap) in Calinski-Harabasz index, demonstrating clearer clustering and stronger category discrimination.

[![DOI](https://zenodo.org/badge/988780888.svg)](https://doi.org/10.5281/zenodo.15826042
        
        )

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12 or higher (with CUDA support recommended for GPU acceleration)
- PyTorch Geometric

### Install from Source

```bash
# Clone the repository
git clone https://github.com/PeterPonyu/GNODEVAE.git
cd GNODEVAE

# Install dependencies
pip install torch torch-geometric scanpy anndata numpy pandas scikit-learn tqdm psutil torchdiffeq
```

### Dependencies

The main dependencies include:
- `torch` - PyTorch deep learning framework
- `torch-geometric` - Geometric deep learning extension for PyTorch
- `scanpy` - Single-cell analysis toolkit
- `anndata` - Annotated data structures for single-cell data
- `torchdiffeq` - Differentiable ODE solvers for PyTorch
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `tqdm` - Progress bars
- `psutil` - System resource monitoring

## Quick Start

### Basic Usage

```python
import scanpy as sc
from GNODEVAE import agent_r  # GraphVAE with refined architecture
# OR
from GNODEVAE import agent  # Standard GraphVAE
# For full GNODEVAE with ODE support, use:
# from GNODEVAE.GODEVAE_agent import GNODEVAE_agent_r

# Load your single-cell data
adata = sc.read_h5ad('your_data.h5ad')

# Initialize the GNODEVAE agent
model = agent_r(
    adata=adata,
    layer='counts',           # Layer containing count data
    n_var=2000,              # Number of highly variable genes
    tech='PCA',              # Dimensionality reduction technique
    n_neighbors=15,          # Number of neighbors for graph construction
    latent_dim=10,           # Latent space dimension
    hidden_dim=128,          # Hidden layer dimension
    encoder_type='graph',    # Use graph encoder
    graph_type='GAT',        # Graph Attention Network
    lr=1e-4,                 # Learning rate
    device='cuda'            # Use GPU if available
)

# Train the model
model.fit(epochs=300, update_steps=10, silent=False)

# Extract latent representations
latent = model.get_latent()

# Store latent representation in AnnData object
adata.obsm['X_gnodevae'] = latent

# Perform downstream analysis (e.g., clustering)
import scanpy as sc
sc.pp.neighbors(adata, use_rep='X_gnodevae')
sc.tl.leiden(adata)
sc.tl.umap(adata)
```

### Using Standard GraphVAE (without ODE)

```python
from GNODEVAE import agent

# Initialize standard GraphVAE agent
model = agent(
    adata=adata,
    layer='counts',
    n_var=2000,
    tech='PCA',
    n_neighbors=15,
    latent_dim=10,
    hidden_dim=128,
    encoder_type='GAT',
    lr=1e-4
)

# Train and extract embeddings
model.fit(epochs=300)
latent = model.get_latent()
```

## Key Parameters

### Data Preprocessing Parameters
- `layer` (str): Layer of AnnData to use (default: 'counts')
- `n_var` (int): Number of highly variable genes to select (default: None, uses all)
- `tech` (str): Dimensionality reduction method - 'PCA', 'NMF', 'FastICA', 'TruncatedSVD', 'FactorAnalysis', or 'LatentDirichletAllocation' (default: 'PCA')
- `n_neighbors` (int): Number of neighbors for graph construction (default: 15)
- `batch_tech` (str): Batch correction method - 'harmony' or 'scvi' (default: None)
- `all_feat` (bool): Whether to use all features or only highly variable genes (default: True)

### Model Architecture Parameters
- `hidden_dim` (int): Hidden layer dimension (default: 128)
- `latent_dim` (int): Latent space dimension for embeddings (default: 10)
- `encoder_type` (str): Encoder type - 'graph' or 'linear' (default: 'graph')
- `graph_type` (str): Graph convolution type - 'GAT', 'GCN', 'SAGE', 'Transformer', etc. (default: 'GAT')
- `structure_decoder_type` (str): Structure decoder type - 'mlp', 'bilinear', or 'inner_product' (default: 'mlp')
- `feature_decoder_type` (str): Feature decoder type - 'linear' or 'graph' (default: 'linear')
- `hidden_layers` (int): Number of hidden layers (default: 2)
- `dropout` (float): Dropout rate (default: 0.05)
- `use_residual` (bool): Whether to use residual connections (default: True)

### Training Parameters
- `lr` (float): Learning rate for optimizer (default: 1e-4)
- `beta` (float): Weight for KL divergence loss term (default: 1.0)
- `graph` (float): Weight for graph reconstruction loss (default: 1.0)
- `epochs` (int): Number of training epochs (default: 300)
- `device` (str or torch.device): Computing device - 'cuda' or 'cpu' (default: auto-detect)
- `num_parts` (int): Number of graph partitions for mini-batch training (default: 10)

### GNODEVAE-Specific Parameters (agent_r)
- `n_ode_hidden` (int): Number of hidden units in ODE function (default: varies)
- `w_recon` (float): Weight for reconstruction loss (default: 1.0)
- `w_kl` (float): Weight for KL divergence loss (default: 1.0)
- `w_adj` (float): Weight for adjacency matrix loss (default: 1.0)
- `w_recon_ode` (float): Weight for ODE reconstruction loss (default: 1.0)

## Model Architecture

GNODEVAE consists of three main components:

1. **Graph Encoder**: Encodes cell-cell relationships and gene expression using Graph Attention Networks (GAT) or other graph convolution layers
2. **Neural ODE**: Models continuous developmental trajectories in the latent space
3. **Decoder**: Reconstructs both graph structure and gene expression from latent representations

The model learns a low-dimensional latent representation that captures:
- Cell type identity
- Developmental state
- Cell-cell relationships
- Temporal dynamics (with ODE component)

## Output

After training, GNODEVAE produces:
- **Latent representations**: Low-dimensional embeddings for each cell
- **Clustering metrics**: ARI, NMI, Silhouette score, Calinski-Harabasz index, Davies-Bouldin index
- **Pseudo-time**: Developmental trajectory information (for agent_r with ODE)
- **Graph structure**: Learned cell-cell similarity graph

## Advanced Usage

### Custom Graph Construction

```python
# Use custom graph construction parameters
model = agent_r(
    adata=adata,
    n_neighbors=30,      # Increase neighbors for denser graph
    graph_type='Transformer',  # Use Transformer convolution
    alpha=0.5            # Set alpha for specific layers
)
```

### Interpretable Mode

```python
# Use interpretable GraphVAE
model = agent(
    adata=adata,
    interpretable=True,  # Enable interpretable mode
    idim=2              # Interpretable dimension
)
```

### Extract Pseudo-time

```python
# For GNODEVAE models with ODE component
# Note: Use GNODEVAE_agent_r from GODEVAE_agent module for pseudo-time functionality
from GNODEVAE.GODEVAE_agent import GNODEVAE_agent_r

model = GNODEVAE_agent_r(adata=adata, ...)
model.fit(epochs=300)

# Get pseudo-time for cells
pseudotime_df = model.partition_time()
```

## Evaluation Metrics

GNODEVAE automatically computes several clustering quality metrics during training:
- **ARI** (Adjusted Rand Index): Measures clustering agreement with ground truth
- **NMI** (Normalized Mutual Information): Information-theoretic clustering metric
- **ASW** (Average Silhouette Width): Measures cluster separation
- **C_H** (Calinski-Harabasz Index): Ratio of between-cluster to within-cluster variance
- **D_B** (Davies-Bouldin Index): Average similarity between clusters
- **P_C** (Pearson Correlation): Correlation between latent dimensions

## Citation

If you use GNODEVAE in your research, please cite:

```bibtex
@article{fu2025gnodevae,
  title={GNODEVAE: a graph-based ODE-VAE enhances clustering for single-cell data},
  author={Fu, Z. and Chen, C. and Wang, S. and others},
  journal={BMC Genomics},
  volume={26},
  pages={767},
  year={2025},
  doi={10.1186/s12864-025-11946-7}
}
```

**Full Citation:**
Fu, Z., Chen, C., Wang, S. et al. GNODEVAE: a graph-based ODE-VAE enhances clustering for single-cell data. BMC Genomics 26, 767 (2025). https://doi.org/10.1186/s12864-025-11946-7
        
        

## License

See LICENSE file for details.

## Contact

For questions and feedback, please open an issue on the GitHub repository.
