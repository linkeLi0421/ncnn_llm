#pragma once

#include <string>

struct Options {
    std::string model_path = "./assets/qwen3_0.6b";
    std::string image_path;
    bool use_vulkan = false;
    bool enable_builtin_tools = true;
    int num_threads = 0;  // 0 = use ncnn::get_cpu_count()
    int vulkan_device = 0;  // Vulkan device index
};

Options parse_options(int argc, char** argv);
