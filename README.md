# GNODEVAE: A Graph-Based ODE-VAE Enhances Clustering of Single-Cell Data

[![PyPI](https://img.shields.io/pypi/v/gnodevae.svg)](https://pypi.org/project/gnodevae/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15826042.svg)](https://doi.org/10.5281/zenodo.15826042)

## Graphical Abstract

![GNODEVAE Graphical Abstract](https://raw.githubusercontent.com/PeterPonyu/GNODEVAE/main/gnodevae_abs.jpeg)

## Introduction

GNODEVAE is a computational framework that integrates Graph Attention Networks (GAT), Neural Ordinary Differential Equations (NODE), and Variational Autoencoders (VAE). It targets three common challenges in single-cell RNA sequencing data analysis:

1. Capturing complex topological relationships between cells
2. Modeling continuous dynamic processes of cell differentiation
3. Handling high levels of technical noise and biological variation

The integration is intended to support identification of cell subpopulations, reconstruction of developmental trajectories, and analysis of cellular heterogeneity.

## Key Contributions

### 1. Dynamic Attention Weighting for Biological Significance

The GAT attention mechanism adaptively weights gene expression profiles, prioritizing biological relationships while reducing technical noise in heterogeneous cell populations.

### 2. Continuous-Time Developmental Modeling via Neural ODEs

Neural ordinary differential equations transform static representations into dynamic systems, with time variables providing a continuous parameterization of developmental processes.

### 3. Biologically Consistent Latent Space Representations

The model's latent space is designed to capture variation in cell differentiation rates, while attention weights can be inspected alongside known developmental relationships between cell types.

### 4. Benchmarking Summary

The accompanying study compares GNODEVAE with several single-cell analysis methods, including scVI, DIP-VAE, TC-VAE, beta-VAE, Info-VAE, and scTour, across 13 datasets.

### 5. Gene Trend Analysis

The accompanying study also reports stronger Calinski-Harabasz scores than selected baselines for gene trend analysis.
        

## Installation

### Prerequisites

- Python 3.9 or higher
- PyTorch 1.13 or higher (with CUDA support recommended for GPU acceleration)
- PyTorch Geometric
- A METIS backend, `pyg-lib` or `torch-sparse`, for training with `agent`, `agent_r` or `GNODEVAE_agent_r`: these split the cell graph with PyTorch Geometric's `ClusterData`, which needs one of the two. `pip install gnodevae` does not install them, because they are published as wheels for each PyTorch and CUDA version on the PyTorch Geometric wheel index rather than on PyPI. Install the build that matches your PyTorch and CUDA versions as described in the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), for example `pip install pyg_lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`. These wheels appear some time after each PyTorch release; if none exist yet for your PyTorch version, install the newest PyTorch version that has them. `GNODEVAE_agent_subgraph` samples subgraphs instead and needs neither, and the package imports without them.

### Install from PyPI

```bash
pip install gnodevae
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/PeterPonyu/GNODEVAE.git
cd GNODEVAE

# Install the package and its dependencies
pip install -e .
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

# Initialize the graph VAE agent (agent_r has no neural ODE component)
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

### GNODEVAE-Specific Parameters (GNODEVAE_agent_r)
- `ode_hidden_dim` (int): Hidden dimension of the latent ODE function (default: 25)
- `w_recon` (float): Weight for reconstruction loss (default: 1.0)
- `w_kl` (float): Weight for KL divergence loss, applied on top of `beta` (default: 1.0)
- `w_adj` (float): Weight for adjacency matrix loss, applied on top of `graph` (default: 1.0)
- `w_recon_ode` (float): Weight for the reconstruction loss of expression decoded from the ODE latent states (default: 1.0)
- `w_z_div` (float): Weight for the squared error between the encoder latent means and the ODE latent states (default: 1.0)

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
- **Clustering metrics**: the training scores described under Evaluation Metrics
- **Pseudo-time**: Developmental trajectory information (for GNODEVAE_agent_r, which has the ODE component)
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

At every epoch, `agent`, `agent_r` and `GNODEVAE_agent_r` score the latent representation of all cells without using cell labels. Reference labels come from k-means clustering (`latent_dim` clusters) of the input feature matrix, computed once when the agent is created; the latent representation is clustered with k-means into the same number of clusters. For `GNODEVAE_agent_r`, the scored representation is `scale1 * latent + scale2 * odelatent`, the mix that `get_mix_latent()` returns (`scale1` and `scale2` default to 0.5). During `fit`, the progress bar shows each score averaged over the last `update_steps` epochs:
- **ARI**: computed with `adjusted_mutual_info_score`, i.e. the adjusted mutual information between the reference labels and the latent clusters (not the adjusted Rand index)
- **NMI**: normalized mutual information between the reference labels and the latent clusters
- **ASW**: silhouette score of the latent clusters
- **C_H**: Calinski-Harabasz score of the latent clusters
- **D_B**: Davies-Bouldin score of the latent clusters (lower is better)
- **P_C**: mean, over latent dimensions, of the summed absolute Pearson correlation with the other latent dimensions

## Citation

If you use GNODEVAE in your research, please cite:

```bibtex
@article{fu2025gnodevae,
  title={GNODEVAE: a graph-based ODE-VAE enhances clustering for single-cell data},
  author={Fu, Zeyu and Chen, Chunlin and Wang, Song and Wang, Junping and Chen, Shilei},
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
