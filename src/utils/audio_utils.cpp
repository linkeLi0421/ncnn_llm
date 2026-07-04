#include "utils/audio_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

static uint16_t read_u16(const unsigned char* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32(const unsigned char* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float hz_to_mel(float hz) {
    constexpr float f_sp = 200.0f / 3.0f;
    float mel = hz / f_sp;
    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = min_log_hz / f_sp;
    constexpr float logstep = 0.068751777f;
    if (hz >= min_log_hz) {
        mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
    }
    return mel;
}

static float mel_to_hz(float mel) {
    constexpr float f_sp = 200.0f / 3.0f;
    float hz = mel * f_sp;
    constexpr float min_log_hz = 1000.0f;
    constexpr float min_log_mel = min_log_hz / f_sp;
    constexpr float logstep = 0.068751777f;
    if (mel >= min_log_mel) {
        hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
    }
    return hz;
}

static std::vector<float> build_mel_filters(int mel_bins, int n_fft, int sample_rate) {
    const int fft_bins = n_fft / 2 + 1;
    std::vector<float> filters((size_t)mel_bins * fft_bins, 0.0f);
    const float min_mel = hz_to_mel(0.0f);
    const float max_mel = hz_to_mel((float)sample_rate / 2.0f);

    std::vector<float> hz_points(mel_bins + 2);
    for (int i = 0; i < mel_bins + 2; i++) {
        float mel = min_mel + (max_mel - min_mel) * (float)i / (float)(mel_bins + 1);
        hz_points[i] = mel_to_hz(mel);
    }

    for (int m = 0; m < mel_bins; m++) {
        const float left = hz_points[m];
        const float center = hz_points[m + 1];
        const float right = hz_points[m + 2];
        const float enorm = 2.0f / std::max(1e-6f, right - left);
        for (int k = 0; k < fft_bins; k++) {
            const float freq = (float)k * (float)sample_rate / (float)n_fft;
            float weight = 0.0f;
            if (freq >= left && freq <= center) {
                weight = (freq - left) / std::max(1e-6f, center - left);
            } else if (freq > center && freq <= right) {
                weight = (right - freq) / std::max(1e-6f, right - center);
            }
            filters[(size_t)m * fft_bins + k] = weight * enorm;
        }
    }
    return filters;
}

} // namespace

bool load_wav_pcm16(const std::string& path, std::vector<float>& mono, int& sample_rate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        return false;
    }

    int channels = 0;
    int bits_per_sample = 0;
    int audio_format = 0;
    const unsigned char* data_ptr = nullptr;
    uint32_t data_size = 0;

    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const unsigned char* chunk = bytes.data() + pos;
        uint32_t chunk_size = read_u32(chunk + 4);
        pos += 8;
        if (pos + chunk_size > bytes.size()) break;
        if (std::memcmp(chunk, "fmt ", 4) == 0 && chunk_size >= 16) {
            audio_format = read_u16(bytes.data() + pos);
            channels = read_u16(bytes.data() + pos + 2);
            sample_rate = (int)read_u32(bytes.data() + pos + 4);
            bits_per_sample = read_u16(bytes.data() + pos + 14);
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data_ptr = bytes.data() + pos;
            data_size = chunk_size;
        }
        pos += chunk_size + (chunk_size & 1u);
    }

    if (!data_ptr || audio_format != 1 || bits_per_sample != 16 || channels <= 0) {
        return false;
    }

    const int samples = (int)(data_size / (uint32_t)(channels * 2));
    mono.assign(samples, 0.0f);
    for (int i = 0; i < samples; i++) {
        float sum = 0.0f;
        for (int ch = 0; ch < channels; ch++) {
            const unsigned char* p = data_ptr + (size_t)(i * channels + ch) * 2;
            int16_t v = (int16_t)read_u16(p);
            sum += (float)v / 32768.0f;
        }
        mono[i] = sum / (float)channels;
    }
    return true;
}

ncnn::Mat wav_to_whisper_log_mel(const std::vector<float>& mono,
                                 int sample_rate,
                                 int mel_bins,
                                 int frames) {
    constexpr int n_fft = 400;
    constexpr int hop = 160;
    const int fft_bins = n_fft / 2 + 1;
    std::vector<float> filters = build_mel_filters(mel_bins, n_fft, sample_rate);
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; i++) {
        window[i] = 0.5f - 0.5f * std::cos(2.0f * 3.14159265358979323846f * (float)i / (float)n_fft);
    }

    ncnn::Mat mel(frames, mel_bins);
    mel.fill(0.0f);
    float max_log = -1e30f;

    std::vector<float> power(fft_bins);
    for (int t = 0; t < frames; t++) {
        const int offset = t * hop;
        std::fill(power.begin(), power.end(), 0.0f);
        for (int k = 0; k < fft_bins; k++) {
            double real = 0.0;
            double imag = 0.0;
            for (int n = 0; n < n_fft; n++) {
                float x = 0.0f;
                int idx = offset + n;
                if (idx >= 0 && idx < (int)mono.size()) {
                    x = mono[(size_t)idx] * window[n];
                }
                double angle = -2.0 * 3.14159265358979323846 * (double)k * (double)n / (double)n_fft;
                real += (double)x * std::cos(angle);
                imag += (double)x * std::sin(angle);
            }
            power[k] = (float)(real * real + imag * imag);
        }

        for (int m = 0; m < mel_bins; m++) {
            double e = 0.0;
            const float* filter = filters.data() + (size_t)m * fft_bins;
            for (int k = 0; k < fft_bins; k++) {
                e += (double)power[k] * (double)filter[k];
            }
            float v = std::log10(std::max(1e-10f, (float)e));
            mel.row(m)[t] = v;
            max_log = std::max(max_log, v);
        }
    }

    const float floor_val = max_log - 8.0f;
    for (int m = 0; m < mel_bins; m++) {
        float* row = mel.row(m);
        for (int t = 0; t < frames; t++) {
            row[t] = (std::max(row[t], floor_val) + 4.0f) / 4.0f;
        }
    }
    return mel;
}
