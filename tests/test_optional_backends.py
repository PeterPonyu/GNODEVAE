"""The package imports without torch-sparse or pyg-lib, and CPU training works on a machine with a GPU."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1] / "GNODEVAE"


def test_package_imports_without_sparse_backends():
    # A None entry in sys.modules makes any `import torch_sparse` / `import pyg_lib` raise ImportError.
    code = ("import sys; sys.modules['torch_sparse'] = None; sys.modules['pyg_lib'] = None; "
            "import GNODEVAE; from GNODEVAE.utils import AdjToEdge, GraphStructureDecoder; "
            "from GNODEVAE import agent, agent_r, GNODEVAE_agent_r, GNODEVAE_agent_subgraph")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


def test_package_source_never_imports_torch_sparse():
    # The bundled comparison methods (CLEAR, SC3, scGCC, scGNN folders) are not part of the package.
    sources = [p for p in PACKAGE.glob("*.py")]
    assert sources
    offenders = [p.name for p in sources if "torch_sparse" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_gpu_memory_is_only_read_for_a_cuda_device(monkeypatch):
    from GNODEVAE.utils import gpu_memory_mb

    def cuda_only(device=None):
        if torch.device(device).type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
        return 3 * 1024 ** 2

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", cuda_only)
    assert gpu_memory_mb(torch.device("cpu")) == 0.0
    assert gpu_memory_mb("cpu") == 0.0
    assert gpu_memory_mb(torch.device("cuda:0")) == pytest.approx(3.0)


def test_cpu_training_on_a_machine_with_a_gpu(monkeypatch):
    import anndata as ad
    from GNODEVAE import GNODEVAE_agent_subgraph

    def cuda_only(device=None):
        if torch.device(device).type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
        return 0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", cuda_only)
    rng = np.random.default_rng(0)
    counts = rng.negative_binomial(2, 0.3, size=(120, 150)).astype(np.float32)
    adata = ad.AnnData(counts.copy())
    adata.layers["counts"] = counts
    model = GNODEVAE_agent_subgraph(adata=adata, layer="counts", n_var=100, n_neighbors=10, latent_dim=4,
                                    hidden_dim=16, device=torch.device("cpu"))
    model.fit(epochs=1, update_steps=1, silent=True)
    assert model.get_latent().shape[0] == adata.n_obs
    assert all(gpu == 0.0 for _, _, gpu in model.resource)
