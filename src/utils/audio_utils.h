#pragma once

#include <string>
#include <vector>

#include <mat.h>

bool load_wav_pcm16(const std::string& path, std::vector<float>& mono, int& sample_rate);

ncnn::Mat wav_to_whisper_log_mel(const std::vector<float>& mono,
                                 int sample_rate,
                                 int mel_bins = 128,
                                 int frames = 256);
