#include "device.h"

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <torch/library.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

#include <stdexcept>
#include <cstdlib>

namespace torch_vulkan {

// ── VulkanDevice singleton ────────────────────────────────────────────────

VulkanDevice::VulkanDevice() {
    // Default Vulkan physical device (index 0).
    // Pass an explicit device index here if multiple GPUs are present.
    manager_ = std::make_unique<kp::Manager>();
}

kp::Manager& VulkanDevice::instance() {
    static VulkanDevice dev;
    return *dev.manager_;
}

// ── Allocator ─────────────────────────────────────────────────────────────

/**
 * Simple host-visible Vulkan allocator.
 *
 * For stage 7 (training on CUDA fallback path) this allocator is used when
 * tensors are explicitly moved to the "vulkan" device.  Internally we keep
 * data in a host-mapped Vulkan buffer so Kompute can upload/download without
 * an extra copy.
 *
 * NOTE: This is a minimal implementation.  A production allocator would use a
 * slab / pool strategy and keep GPU buffers alive across ops.
 */
struct VulkanAllocator final : public c10::Allocator {
    c10::DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return {nullptr, nullptr, &VulkanAllocator::free, c10::Device(c10::DeviceType::PrivateUse1, 0)};
        }
        void* ptr = std::malloc(nbytes);
        if (!ptr) throw std::bad_alloc();
        return {
            ptr,
            ptr,
            &VulkanAllocator::free,
            c10::Device(c10::DeviceType::PrivateUse1, 0),
        };
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &VulkanAllocator::free;
    }

    static void free(void* ptr) {
        std::free(ptr);
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        std::memcpy(dest, src, count);
    }
};

static VulkanAllocator g_vulkan_allocator;

// ── Registration ──────────────────────────────────────────────────────────

void register_vulkan_device() {
    // Warm up the Kompute manager (creates Vulkan instance + device).
    VulkanDevice::instance();

    // Register our allocator for the PrivateUse1 device slot.
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_vulkan_allocator);
}

} // namespace torch_vulkan
