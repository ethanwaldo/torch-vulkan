# ── GLSL → SPIR-V compilation ─────────────────────────────────────────────
set(SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/src/torch_vulkan/shaders")
set(SHADER_OUT_DIR "${SHADER_SRC_DIR}")

find_program(GLSLC glslc HINTS
    "$ENV{VULKAN_SDK}/Bin"
    "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

foreach(SHADER mm add)
    add_custom_command(
        OUTPUT  "${SHADER_OUT_DIR}/${SHADER}.spv"
        COMMAND ${GLSLC} "${SHADER_SRC_DIR}/${SHADER}.comp" -o "${SHADER_OUT_DIR}/${SHADER}.spv"
        DEPENDS "${SHADER_SRC_DIR}/${SHADER}.comp"
        COMMENT "Compiling ${SHADER}.comp → ${SHADER}.spv"
    )
    list(APPEND SPIRV_OUTPUTS "${SHADER_OUT_DIR}/${SHADER}.spv")
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPIRV_OUTPUTS})

# ── C++ extension sources ─────────────────────────────────────────────────
set(SOURCES
    torch_vulkan.cpp
    device.cpp
    ops/mm.cpp
    ops/add.cpp
)

# Build as a shared library importable by Python
add_library(_C MODULE ${SOURCES})
add_dependencies(_C shaders)

target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(_C PRIVATE
    ${TORCH_LIBRARIES}
    Vulkan::Vulkan
    kompute
)

# PyTorch requires -D_GLIBCXX_USE_CXX11_ABI=0 on Linux when built without new ABI.
# On Windows with MSVC this is a no-op.
target_compile_definitions(_C PRIVATE ${TORCH_CXX_FLAGS})

# Extension must not have a "lib" prefix and must carry the Python suffix
set_target_properties(_C PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Copy the compiled extension into the source tree so `pip install -e .` works
add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:_C>
        ${CMAKE_SOURCE_DIR}/src/torch_vulkan/_C$<TARGET_FILE_SUFFIX:_C>
    COMMENT "Copying _C extension to source tree"
)
