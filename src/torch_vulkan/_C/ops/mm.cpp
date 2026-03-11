/**
 * aten::mm — matrix multiply via Vulkan/Kompute.
 *
 * C = A @ B  where A is [M, K], B is [K, N], C is [M, N]
 *
 * Dispatch path:
 *   1. Upload A and B to Vulkan tensor buffers
 *   2. Dispatch mm.comp GLSL shader with push constants (M, K, N)
 *   3. Download result into a new CPU-side ATen tensor
 *
 * The current implementation uses host-visible buffers so upload/download
 * are simple memcpy.  A future revision can use device-local buffers with
 * staging transfers for better throughput.
 */

#include "mm.h"
#include "../device.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <Kompute.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

namespace torch_vulkan {

// ── Shader loading ────────────────────────────────────────────────────────

static std::vector<uint32_t> load_spirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    TORCH_CHECK(file.is_open(), "torch_vulkan: cannot open SPIR-V shader: ", path);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> buf(size / 4);
    file.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

// Locate shaders relative to this shared library at runtime.
// When installed via pip, shaders/ is copied next to _C.so by setup.
static std::string shader_path(const std::string& name) {
    // __FILE__ gives the .cpp path at compile time — walk up to the package root.
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    // During development: _C/ops/ → _C/ → torch_vulkan/ → shaders/
    auto candidate = here.parent_path().parent_path() / "shaders" / name;
    return candidate.string();
}

// ── vulkan_mm ─────────────────────────────────────────────────────────────

at::Tensor vulkan_mm(const at::Tensor& self, const at::Tensor& mat2) {
    TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2,
        "torch_vulkan::mm: expected 2-D tensors");
    TORCH_CHECK(self.size(1) == mat2.size(0),
        "torch_vulkan::mm: size mismatch: ", self.sizes(), " vs ", mat2.sizes());

    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const int64_t N = mat2.size(1);

    // Work in float32 for Vulkan shader compatibility.
    auto a = self.to(at::kFloat).contiguous();
    auto b = mat2.to(at::kFloat).contiguous();

    auto& mgr = VulkanDevice::instance();

    // Create Kompute tensors (host-visible by default)
    auto kA = mgr.tensor(a.data_ptr<float>(), a.numel());
    auto kB = mgr.tensor(b.data_ptr<float>(), b.numel());
    auto kC = mgr.tensor<float>(M * N);  // output, zero-initialised

    // Push constants: M, K, N
    struct PushConstants { uint32_t M, K, N; } pc{
        static_cast<uint32_t>(M),
        static_cast<uint32_t>(K),
        static_cast<uint32_t>(N),
    };

    auto spirv = load_spirv(shader_path("mm.spv"));

    auto algo = mgr.algorithm<float, PushConstants>(
        {kA, kB, kC},
        spirv,
        kp::Workgroup{
            static_cast<uint32_t>((N + 15) / 16),
            static_cast<uint32_t>((M + 15) / 16),
            1u,
        },
        {},     // spec constants
        {pc}
    );

    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>({kA, kB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpTensorSyncLocal>({kC})
        ->eval();

    // Copy result back into an ATen tensor on the original device.
    auto result = at::empty({M, N}, self.options().device(at::kCPU).dtype(at::kFloat));
    std::memcpy(result.data_ptr<float>(), kC->data<float>(), M * N * sizeof(float));

    return result.to(self.dtype()).to(self.device());
}

} // namespace torch_vulkan
