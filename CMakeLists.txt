cmake_minimum_required(VERSION 3.16)

project(ncnn_llm LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NCNN_LLM_BUILD_QWEN3_ASR "Build the Qwen3-ASR example runner" ON)

find_package(ncnn REQUIRED)

find_package(nlohmann_json QUIET)
set(NCNN_LLM_NLOHMANN_JSON_INCLUDE_DIR "" CACHE PATH "Path containing nlohmann/json.hpp")

if(NOT TARGET nlohmann_json::nlohmann_json AND NOT NCNN_LLM_NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR
        "nlohmann_json was not found. Install nlohmann_json, or pass "
        "-DNCNN_LLM_NLOHMANN_JSON_INCLUDE_DIR=/path/to/include")
endif()

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME qwen3_asr_platform_smoke_py
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen3_asr/test_collect_platform_smoke.py
        )
    endif()
endif()

if(MSVC)
    add_compile_definitions(NOMINMAX)
    add_compile_options(/utf-8)
endif()

add_library(ncnn_tokenizer STATIC
    src/utils/tokenizer/bpe_tokenizer.cpp
    src/utils/tokenizer/unigram_tokenizer.cpp
)
target_include_directories(ncnn_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(NCNN_LLM_BUILD_QWEN3_ASR)
    add_library(ncnn_qwen3_asr_runtime STATIC
        src/ncnn_qwen3_asr.cpp
        src/utils/audio_utils.cpp
        src/utils/rope_embed.cpp
    )
    target_include_directories(ncnn_qwen3_asr_runtime PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(ncnn_qwen3_asr_runtime PUBLIC
        ncnn
        ncnn_tokenizer
    )
    if(TARGET nlohmann_json::nlohmann_json)
        target_link_libraries(ncnn_qwen3_asr_runtime PUBLIC nlohmann_json::nlohmann_json)
    else()
        target_include_directories(ncnn_qwen3_asr_runtime PUBLIC ${NCNN_LLM_NLOHMANN_JSON_INCLUDE_DIR})
    endif()

    add_executable(qwen3_asr_main
        examples/qwen3_asr_main.cpp
    )
    target_link_libraries(qwen3_asr_main PRIVATE
        ncnn_qwen3_asr_runtime
    )
endif()
