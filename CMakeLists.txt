cmake_minimum_required(VERSION 3.21)
project(torch_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── PyTorch ──────────────────────────────────────────────────────────────
# Set Torch_DIR via env or cmake cache:
#   cmake -DTorch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
find_package(Torch REQUIRED)

# ── Vulkan SDK ────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)

# ── Kompute (header-only friendly; fetched from GitHub) ───────────────────
include(FetchContent)
FetchContent_Declare(
    kompute
    GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
    GIT_TAG        v0.9.0
    GIT_SHALLOW    TRUE
)
set(KOMPUTE_OPT_USE_BUILT_IN_SPDLOG ON  CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_USE_BUILT_IN_FMT    ON  CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_BUILD_TESTS         OFF CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_BUILD_EXAMPLES      OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(kompute)

# ── C++ extension ─────────────────────────────────────────────────────────
add_subdirectory(src/torch_vulkan/_C)
