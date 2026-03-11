/**
 * aten::add_.Tensor — in-place element-wise add via Vulkan/Kompute.
 *
 * self += alpha * other  (element-wise, broadcast supported for scalar alpha)
 *
 * Used in LoRA weight application: W += B·A
 */

#include "add.h"
#include "../device.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <Kompute.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

namespace torch_vulkan {

static std::vector<uint32_t> load_spirv_add(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    TORCH_CHECK(file.is_open(), "torch_vulkan: cannot open SPIR-V shader: ", path);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> buf(size / 4);
    file.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

static std::string shader_path_add(const std::string& name) {
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    return (here.parent_path().parent_path() / "shaders" / name).string();
}

at::Tensor& vulkan_add_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    TORCH_CHECK(self.numel() == other.numel() || other.numel() == 1,
        "torch_vulkan::add_: size mismatch: ", self.sizes(), " vs ", other.sizes());

    const float alpha_val = alpha.to<float>();
    const uint32_t n = static_cast<uint32_t>(self.numel());

    auto a = self.to(at::kFloat).contiguous();
    auto b = other.to(at::kFloat).contiguous().expand_as(a).contiguous();

    auto& mgr = VulkanDevice::instance();

    auto kA = mgr.tensor(a.data_ptr<float>(), a.numel());
    auto kB = mgr.tensor(b.data_ptr<float>(), b.numel());

    struct PushConstants { uint32_t n; float alpha; } pc{ n, alpha_val };

    auto spirv = load_spirv_add(shader_path_add("add.spv"));

    auto algo = mgr.algorithm<float, PushConstants>(
        {kA, kB},
        spirv,
        kp::Workgroup{ (n + 255) / 256, 1u, 1u },
        {},
        {pc}
    );

    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>({kA, kB})
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpTensorSyncLocal>({kA})   // kA is both input and output
        ->eval();

    // If self is already float32, `a` shares its data pointer, so writing
    // to `a.data_ptr<float>()` modifies `self` directly.
    std::memcpy(a.data_ptr<float>(), kA->data<float>(), n * sizeof(float));

    // If self was a different dtype (e.g., bfloat16, float16), `a` is a 
    // separate float32 copy. We need to cast back. 
    // `self.copy_(a)` handles both dtype casting and device transfers if needed.
    if (!self.is_same(a)) {
        self.copy_(a);
    }
    
    return self;
}

} // namespace torch_vulkan
