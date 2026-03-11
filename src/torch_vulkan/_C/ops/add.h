#pragma once
#include <ATen/ATen.h>

namespace torch_vulkan {

/** aten::add_.Tensor — in-place element-wise add via Vulkan/Kompute. */
at::Tensor& vulkan_add_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);

} // namespace torch_vulkan
