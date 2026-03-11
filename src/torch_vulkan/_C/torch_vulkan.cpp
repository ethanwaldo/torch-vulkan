/**
 * torch_vulkan.cpp
 *
 * PyBind11 module entry point + ATen operator registrations for the
 * PyTorch PrivateUse1 ("vulkan") device backend.
 *
 * Op coverage for Stage 7 (LoRA training):
 *   aten::mm          — matrix multiply (core of every linear layer)
 *   aten::add_.Tensor — in-place add   (LoRA: W += B·A)
 *
 * All other ops fall through to an informative "not implemented" error
 * so that missing coverage is surfaced immediately rather than silently
 * falling back to CPU.
 */

#include <torch/library.h>
#include <torch/python.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include "device.h"
#include "ops/mm.h"
#include "ops/add.h"

// ── Unimplemented stub ────────────────────────────────────────────────────

static at::Tensor not_implemented(const char* op_name) {
    TORCH_CHECK(false,
        "torch_vulkan: operator '", op_name, "' is not yet implemented. "
        "Contributions welcome at https://github.com/ethanwaldo/torch-vulkan");
}

// ── ATen operator registrations ───────────────────────────────────────────

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Stage 7 critical ops
    m.impl("mm",          &torch_vulkan::vulkan_mm);
    m.impl("add_.Tensor", &torch_vulkan::vulkan_add_);

    // Stage 7 / 8 ops — stubs, to be implemented
    m.impl("bmm",    [](const at::Tensor&, const at::Tensor&) -> at::Tensor {
        return not_implemented("bmm"); });
    m.impl("silu",   [](const at::Tensor&) -> at::Tensor {
        return not_implemented("silu"); });
    m.impl("softmax.int", [](const at::Tensor&, int64_t, bool) -> at::Tensor {
        return not_implemented("softmax"); });
}

// ── Python module ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_C, m) {
    m.doc() = "torch-vulkan PrivateUse1 backend (Vulkan/Kompute)";

    // Initialize the Vulkan device + allocator on import.
    torch_vulkan::register_vulkan_device();

    m.def("device_count", []() -> int {
        // Minimal: assume 1 Vulkan device when the backend is loaded.
        return 1;
    });
}
