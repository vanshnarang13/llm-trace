// Fixed-capacity ring buffer for telemetry events. Oldest events are evicted
// once capacity is reached, bounding RAM regardless of how long inference runs.
//
// Attention matrices can be large, so they are NOT stored in the ring: only the
// most recent matrix per layer is retained, keyed by layer name.
#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "telemetry/types.hpp"

namespace scope {

class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity = 10000);

    void push(const TelemetryEvent& event);

    // Chronological snapshot, oldest first. Attention events are appended after
    // the ring contents so consumers that want them still see the latest.
    std::vector<TelemetryEvent> get_all() const;

    // Most recent attention matrix captured for a given layer, if any.
    std::optional<AttentionData> latest_attention(const std::string& layer) const;

    std::size_t size() const;
    std::size_t capacity() const { return capacity_; }
    void clear();

private:
    std::size_t capacity_;
    std::vector<TelemetryEvent> buffer_;
    std::size_t write_ptr_ = 0;
    std::size_t size_ = 0;
    std::unordered_map<std::string, AttentionData> latest_attn_;
    mutable std::mutex mutex_;
};

}  // namespace scope
