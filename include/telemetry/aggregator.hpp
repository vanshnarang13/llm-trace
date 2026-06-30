// Rolling statistics over the telemetry stream: per-layer latency profiles,
// generation throughput (tokens/sec) and a device distribution. Drives the
// performance dashboard and the flame graph.
#pragma once

#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "telemetry/types.hpp"

namespace scope {

struct LayerLatency {
    std::string name;
    std::string type;
    double avg_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    int call_count = 0;
};

class TelemetryAggregator {
public:
    TelemetryAggregator();

    void process_event(const TelemetryEvent& event);

    double tokens_per_sec() const;
    double avg_layer_latency() const;
    std::vector<LayerLatency> slowest_layers(std::size_t limit = 5) const;
    std::unordered_map<std::string, int> device_distribution() const;
    std::size_t total_events() const;
    void reset();

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, LayerLatency> layers_;
    std::unordered_map<std::string, int> device_counts_;
    std::size_t total_events_ = 0;
    int token_count_ = 0;
    double total_latency_ms_ = 0.0;
    std::size_t latency_samples_ = 0;
    std::chrono::steady_clock::time_point session_start_;
};

}  // namespace scope
