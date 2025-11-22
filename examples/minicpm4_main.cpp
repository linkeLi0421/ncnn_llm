#include "minicpm4_0.5b.h"


int main() {
    
    minicpm4_0_5b model("./assets/minicpm4_0.5b/minicpm4_embed_token.ncnn.param",
                        "./assets/minicpm4_0.5b/minicpm4_embed_token.ncnn.bin",
                        "./assets/minicpm4_0.5b/minicpm4_proj_out.ncnn.param",
                        "./assets/minicpm4_0.5b/minicpm4_decoder.ncnn.param",
                        "./assets/minicpm4_0.5b/minicpm4_decoder.ncnn.bin",
                        "./assets/minicpm4_0.5b/vocab.txt",
                        "./assets/minicpm4_0.5b/merges.txt",
                       /*use_vulkan=*/false);
                       
    auto ctx = model.prefill("什么是ncnn?");
    ctx = model.prefill("ncnn是", ctx);
    model.decode(ctx, [](const std::string& token) {
        printf("%s", token.c_str());
    });

    return 0;
}