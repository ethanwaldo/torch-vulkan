#pragma once
#include <memory>
#include <Kompute.hpp>

namespace torch_vulkan {

/**
 * Singleton that owns the Kompute manager (Vulkan instance, device, queues).
 * Call VulkanDevice::instance() to get the manager.
 */
class VulkanDevice {
public:
    static kp::Manager& instance();

private:
    VulkanDevice();
    std::unique_ptr<kp::Manager> manager_;
};

/** Register the custom device and allocator with PyTorch. */
void register_vulkan_device();

} // namespace torch_vulkan
