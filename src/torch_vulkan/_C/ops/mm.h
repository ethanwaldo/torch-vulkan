#pragma once
#include <ATen/ATen.h>

namespace torch_vulkan {

/** aten::mm — matrix multiply via Vulkan/Kompute. */
at::Tensor vulkan_mm(const at::Tensor& self, const at::Tensor& mat2);

} // namespace torch_vulkan
