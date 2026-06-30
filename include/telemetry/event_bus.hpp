// Tiny publish/subscribe hub. Producers (TCP server, mock tracer) publish
// TelemetryEvents; consumers (ring buffer, aggregator, anomaly detector, the
// TUI) subscribe either to a specific event_type or to "*" for everything.
#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "telemetry/types.hpp"

namespace scope {

class EventBus {
public:
    using Callback = std::function<void(const TelemetryEvent&)>;

    // event_type may be a concrete type or "*" to receive all events.
    void subscribe(const std::string& event_type, Callback cb);

    // Fan an event out to matching subscribers. Safe to call from any thread.
    void publish(const TelemetryEvent& event);

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::vector<Callback>> subscribers_;
};

}  // namespace scope
