"""
torch-vulkan: PyTorch PrivateUse1 backend backed by Vulkan/Kompute.

Importing this module:
  1. Registers "vulkan" as the PrivateUse1 backend name
  2. Loads the C++ extension which registers op dispatches and the device allocator

Usage:
    import torch
    import torch_vulkan          # activates the backend
    t = torch.tensor([1, 2, 3], device="vulkan")
"""

import torch

# Register the PrivateUse1 slot as "vulkan" before loading the extension so
# that any ops registered inside _C see the right backend string.
torch.utils.backend_registration.rename_privateuse1_backend("vulkan")

# Load the compiled C++ extension.  It registers:
#   - VulkanAllocator with torch's allocator registry
#   - TORCH_LIBRARY_IMPL(aten, PrivateUse1, ...) op dispatches
try:
    from torch_vulkan import _C  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "torch_vulkan._C extension not found. "
        "Build the project first:\n"
        "  pip install -e . --no-build-isolation\n"
        "(requires CMake, MSVC/GCC, and the Vulkan SDK)"
    ) from exc

__all__ = ["_C"]
