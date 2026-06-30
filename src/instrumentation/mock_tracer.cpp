#include "instrumentation/mock_tracer.hpp"

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace scope {

namespace {

constexpr int kLayers = 32;
constexpr int kHidden = 4096;
constexpr int kHeads = 32;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// Growing "prompt" so the attention matrix changes size as decoding proceeds.
const std::vector<std::string>& vocab() {
    static const std::vector<std::string> v = {
        "I",   "want",   "it",     "to",      "be",   "keyboard",
        "driven", "and", "fast",   ",",       "like", "btop",
        "or",  "lazygit", "running", "locally"};
    return v;
}

}  // namespace

MockTracer::MockTracer(EventBus& bus, DeviceMonitor& device) : bus_(bus) {
    (void)device;  // reserved for future device-aware simulation
}

MockTracer::~MockTracer() { stop(); }

void MockTracer::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread(&MockTracer::loop, this);
}

void MockTracer::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
}

void MockTracer::loop() {
    TelemetryEvent info;
    info.event_type = "model_info";
    info.timestamp = now_ms();
    info.model_info = {"llama-3-8b", kLayers, kHidden, kHeads, 128256, "Q4_K_M"};
    bus_.publish(info);

    int64_t event_id = 100;
    int token = 0;
    while (running_) {
        emit_token_pass(token, event_id);
        token = (token + 1) % static_cast<int>(vocab().size());
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }
}

void MockTracer::emit_token_pass(int token_index, int64_t& event_id) {
    std::uniform_real_distribution<double> jitter(0.85, 1.15);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // Sequence seen so far grows up to the full prompt.
    int seq_len = std::min(token_index + 4, static_cast<int>(vocab().size()));

    auto publish_trace = [&](const std::string& name, const std::string& type,
                             double base_latency, double mean, double maxv,
                             double sparsity, const std::string& device) {
        if (!running_) return;
        TelemetryEvent ev;
        ev.event_type = "layer_trace";
        ev.timestamp = now_ms();
        ev.event_id = event_id++;
        ev.layer_name = name;
        ev.layer_type = type;
        ev.device = device;
        ev.latency_ms = base_latency * jitter(rng_);
        ev.input_tensor = {{1, seq_len, kHidden}, "float16",
                           int64_t{1} * seq_len * kHidden * 2, device};
        ev.output_tensor = ev.input_tensor;
        ev.activation_stats.mean = mean;
        ev.activation_stats.variance = 0.4 + unit(rng_) * 0.3;
        ev.activation_stats.min_val = -maxv * (0.6 + unit(rng_) * 0.4);
        ev.activation_stats.max_val = maxv;
        ev.activation_stats.sparsity = sparsity;
        bus_.publish(ev);
    };

    publish_trace("embed_tokens", "Embedding", 0.20, 0.01, 1.8, 2.0, "CUDA:0");

    for (int l = 0; l < kLayers && running_; ++l) {
        std::string prefix = "layers." + std::to_string(l);
        publish_trace(prefix + ".input_layernorm", "RMSNorm", 0.05, 0.0, 1.2,
                      0.0, "CUDA:0");

        // Self attention, slightly heavier on deeper layers.
        double attn_lat = 1.0 + l * 0.03;
        publish_trace(prefix + ".self_attn", "SelfAttention", attn_lat, 0.03,
                      4.5 + l * 0.05, 52.0 + unit(rng_) * 6.0, "CUDA:0");

        // Attention probabilities for this layer (downsampled to seq_len).
        TelemetryEvent attn;
        attn.event_type = "attention_weights";
        attn.timestamp = now_ms();
        attn.attention.layer_name = prefix + ".self_attn";
        attn.attention.num_heads = 3;  // a few representative heads
        attn.attention.token_count = seq_len;
        for (int t = 0; t < seq_len; ++t) attn.attention.tokens.push_back(vocab()[t]);
        attn.attention.matrices.resize(3,
            std::vector<std::vector<float>>(seq_len, std::vector<float>(seq_len, 0.0f)));
        for (int h = 0; h < 3; ++h) {
            for (int q = 0; q < seq_len; ++q) {
                // Head 0: local banded; head 1: attend-to-first; head 2: diffuse.
                std::vector<float> row(seq_len, 0.0f);
                float sum = 0.0f;
                for (int k = 0; k <= q; ++k) {
                    float w;
                    if (h == 0)
                        w = std::exp(-0.9f * (q - k));
                    else if (h == 1)
                        w = (k == 0 ? 1.5f : 0.2f) + 0.05f * k;
                    else
                        w = 0.5f + 0.5f * static_cast<float>(unit(rng_));
                    row[k] = w;
                    sum += w;
                }
                for (int k = 0; k <= q; ++k) attn.attention.matrices[h][q][k] = row[k] / sum;
            }
        }
        bus_.publish(attn);

        publish_trace(prefix + ".post_attention_layernorm", "RMSNorm", 0.05, 0.0,
                      1.2, 0.0, "CUDA:0");

        // MLP, the usual latency hot spot. Occasionally explode / fall back.
        double mlp_max = 5.5 + l * 0.06;
        std::string mlp_device = "CUDA:0";
        if (unit(rng_) < 0.015) {
            mlp_max = 12.5;  // exploding activations
        }
        if (unit(rng_) < 0.01) {
            mlp_device = "CPU";  // simulated OOM fallback
            TelemetryEvent oom;
            oom.event_type = "anomaly";
            oom.timestamp = now_ms();
            oom.anomaly.severity = "ERROR";
            oom.anomaly.layer_name = prefix + ".mlp";
            oom.anomaly.description = "CUDA OOM fallback: MLP executed on CPU host memory";
            bus_.publish(oom);
        }
        publish_trace(prefix + ".mlp", "MLP", 2.4 + l * 0.05, 0.05, mlp_max,
                      48.0 + unit(rng_) * 8.0, mlp_device);
    }

    publish_trace("norm", "RMSNorm", 0.06, 0.0, 1.1, 0.0, "CUDA:0");
    publish_trace("lm_head", "LMHead", 0.9, 0.02, 9.5, 5.0, "CUDA:0");
}

}  // namespace scope
