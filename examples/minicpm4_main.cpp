#include "minicpm4_0.5b.h"
#include <cstdio>


int main() {
    
    minicpm4_0_5b model("./assets/minicpm4_0.5b/minicpm4_embed_token.ncnn.param",
                        "./assets/minicpm4_0.5b/minicpm4_embed_token.ncnn.bin",
                        "./assets/minicpm4_0.5b/minicpm4_proj_out.ncnn.param",
                        "./assets/minicpm4_0.5b/minicpm4_decoder.ncnn.param",
                        "./assets/minicpm4_0.5b/minicpm4_decoder.ncnn.bin",
                        "./assets/minicpm4_0.5b/vocab.txt",
                        "./assets/minicpm4_0.5b/merges.txt",
                       /*use_vulkan=*/false);
                       
    auto ctx = model.prefill("写一个cpp的快速排序代码, \n");
    model.decode(ctx, [](const std::string& token) {
        std::string output = token;

        // replace /to /n with actual characters
        for (size_t pos = 0; (pos = output.find("\\n", pos)) != std::string::npos; pos += 1) {
            output.replace(pos, 2, "\n");
        }
        for (size_t pos = 0; (pos = output.find("\\t", pos)) != std::string::npos; pos += 1) {
            output.replace(pos, 2, "\t");
        }
        // replace ▁ with space
        for (size_t pos = 0; (pos = output.find("▁", pos)) != std::string::npos; pos += 1) {
            output.replace(pos, 3, " ");
        }

        printf("%s", output.c_str());
    });

    return 0;
}