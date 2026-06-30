// Audits each layer_trace for numerical instability and records alerts. Also
// passes through anomaly events that senders flag themselves (e.g. CPU/OOM
// fallbacks). Drives the Numerical Anomaly Ledger panel.
#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "telemetry/event_bus.hpp"
#include "telemetry/types.hpp"

namespace scope {

class AnomalyDetector {
public:
    struct Config {
        double exploding_threshold = 10.0;       // |max| activation
        double dead_variance_threshold = 1e-6;   // near-zero variance => dead
        double high_sparsity_threshold = 95.0;   // percent
    };

    explicit AnomalyDetector(EventBus& bus) : AnomalyDetector(bus, Config{}) {}
    AnomalyDetector(EventBus& bus, Config config);

    void process_event(const TelemetryEvent& event);

    std::vector<AnomalyAlert> alerts() const;
    void clear();

private:
    void record(const std::string& severity, const std::string& layer,
                const std::string& desc, int64_t timestamp);

    Config config_;
    std::vector<AnomalyAlert> alerts_;
    mutable std::mutex mutex_;
};

}  // namespace scope
