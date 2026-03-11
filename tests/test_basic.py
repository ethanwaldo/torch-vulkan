"""
Basic smoke tests for the torch-vulkan PrivateUse1 backend.

Run:  pytest tests/test_basic.py
Requires the _C extension to be compiled first (pip install -e .).
"""

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def load_backend():
    import torch_vulkan  # noqa: F401 — registers "vulkan" backend + device


def test_backend_registered():
    """PyTorch should recognise 'vulkan' as a known device after import."""
    import torch_vulkan  # noqa: F401
    # rename_privateuse1_backend makes torch.device("vulkan") valid
    d = torch.device("vulkan", 0)
    assert d.type == "vulkan"


def test_add_on_vulkan():
    """Element-wise add on vulkan-device tensors."""
    a = torch.ones(4, device="vulkan")
    b = torch.ones(4, device="vulkan") * 2
    c = a.add_(b)
    expected = torch.tensor([3.0, 3.0, 3.0, 3.0])
    assert torch.allclose(c.cpu(), expected)


def test_mm_on_vulkan():
    """Matrix multiply on vulkan-device tensors."""
    # [2, 3] @ [3, 2] → [2, 2]
    a = torch.eye(2, 3, device="vulkan")       # [[1,0,0],[0,1,0]]
    b = torch.eye(3, 2, device="vulkan")       # [[1,0],[0,1],[0,0]]
    c = torch.mm(a, b)
    expected = torch.eye(2)
    assert torch.allclose(c.cpu(), expected)


def test_device_count():
    from torch_vulkan import _C
    assert _C.device_count() >= 1
