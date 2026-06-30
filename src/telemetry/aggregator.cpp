#include "telemetry/aggregator.hpp"

#include <algorithm>

namespace scope {

TelemetryAggregator::TelemetryAggregator()
    : session_start_(std::chrono::steady_clock::now()) {}

void TelemetryAggregator::process_event(const TelemetryEvent& event) {
    std::lock_guard<std::mutex> lk(mutex_);
    ++total_events_;
    if (event.event_type != "layer_trace") return;

    auto& info = layers_[event.layer_name];
    info.name = event.layer_name;
    info.type = event.layer_type;
    info.avg_latency_ms =
        (info.avg_latency_ms * info.call_count + event.latency_ms) /
        (info.call_count + 1);
    info.max_latency_ms = std::max(info.max_latency_ms, event.latency_ms);
    ++info.call_count;

    device_counts_[event.device]++;
    total_latency_ms_ += event.latency_ms;
    ++latency_samples_;

    // Each token-generation cycle restarts at the embedding lookup.
    if (event.layer_name == "embed_tokens") ++token_count_;
}

double TelemetryAggregator::tokens_per_sec() const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto now = std::chrono::steady_clock::now();
    double secs =
        std::chrono::duration<double>(now - session_start_).count();
    if (secs <= 0.0 || token_count_ == 0) return 0.0;
    return token_count_ / secs;
}

double TelemetryAggregator::avg_layer_latency() const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (latency_samples_ == 0) return 0.0;
    return total_latency_ms_ / static_cast<double>(latency_samples_);
}

std::vector<LayerLatency> TelemetryAggregator::slowest_layers(
    std::size_t limit) const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<LayerLatency> all;
    all.reserve(layers_.size());
    for (const auto& [name, info] : layers_) all.push_back(info);
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
        return a.avg_latency_ms > b.avg_latency_ms;
    });
    if (all.size() > limit) all.resize(limit);
    return all;
}

std::unordered_map<std::string, int>
TelemetryAggregator::device_distribution() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return device_counts_;
}

std::size_t TelemetryAggregator::total_events() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return total_events_;
}

void TelemetryAggregator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    layers_.clear();
    device_counts_.clear();
    total_events_ = 0;
    token_count_ = 0;
    total_latency_ms_ = 0.0;
    latency_samples_ = 0;
    session_start_ = std::chrono::steady_clock::now();
}

}  // namespace scope
