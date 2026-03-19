#pragma once

#include "options.h"

#include "ncnn_llm_gpt.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

using nlohmann::json;

TemplateType detect_template_type(const std::string& model_path);

int run_cli(const Options& opt,
            ncnn_llm_gpt& model,
            const std::vector<json>& builtin_tools,
            const std::unordered_map<std::string, std::function<json(const json&)>>& builtin_router,
            TemplateType template_type,
            const ncnn::Mat& image = ncnn::Mat());