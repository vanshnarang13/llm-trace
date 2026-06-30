#include "anomaly/anomaly_detector.hpp"

#include <cmath>
#include <ctime>
#include <sstream>

namespace scope {

namespace {
std::string fmt_time(int64_t epoch_ms) {
    if (epoch_ms <= 0) {
        epoch_ms = static_cast<int64_t>(std::time(nullptr)) * 1000;
    }
    std::time_t secs = epoch_ms / 1000;
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&secs));
    std::ostringstream ss;
    ss << buf << '.' << (epoch_ms % 1000);
    return ss.str();
}
}  // namespace

AnomalyDetector::AnomalyDetector(EventBus& bus, Config config)
    : config_(config) {
    // bus is accepted for symmetry with the other consumers; this detector
    // currently only records into its own ledger rather than re-publishing.
    (void)bus;
}

void AnomalyDetector::process_event(const TelemetryEvent& event) {
    if (event.event_type == "anomaly") {
        // Sender-flagged anomaly (e.g. CUDA OOM -> CPU fallback). Pass through.
        std::lock_guard<std::mutex> lk(mutex_);
        AnomalyAlert a = event.anomaly;
        if (a.timestamp.empty()) a.timestamp = fmt_time(event.timestamp);
        alerts_.push_back(a);
        return;
    }
    if (event.event_type != "layer_trace") return;

    const auto& s = event.activation_stats;
    if (std::isnan(s.mean) || std::isinf(s.mean) || std::isnan(s.max_val) ||
        std::isinf(s.max_val)) {
        record("ERROR", event.layer_name,
               "NaN/Inf detected in activations", event.timestamp);
        return;
    }
    double abs_max = std::max(std::abs(s.max_val), std::abs(s.min_val));
    if (abs_max > config_.exploding_threshold) {
        std::ostringstream d;
        d << "Exploding activations: |max| " << abs_max << " exceeds threshold "
          << config_.exploding_threshold;
        record("WARNING", event.layer_name, d.str(), event.timestamp);
    }
    if (s.variance >= 0.0 && s.variance < config_.dead_variance_threshold &&
        event.layer_type != "Embedding") {
        record("WARNING", event.layer_name,
               "Dead layer: near-zero activation variance", event.timestamp);
    }
    if (s.sparsity > config_.high_sparsity_threshold) {
        std::ostringstream d;
        d << "High sparsity " << s.sparsity << "% may indicate clipping";
        record("WARNING", event.layer_name, d.str(), event.timestamp);
    }
}

void AnomalyDetector::record(const std::string& severity,
                             const std::string& layer, const std::string& desc,
                             int64_t timestamp) {
    AnomalyAlert a;
    a.timestamp = fmt_time(timestamp);
    a.severity = severity;
    a.layer_name = layer;
    a.description = desc;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        // Cap the ledger so a misbehaving layer cannot grow it without bound.
        if (alerts_.size() >= 500) alerts_.erase(alerts_.begin());
        alerts_.push_back(a);
    }
}

std::vector<AnomalyAlert> AnomalyDetector::alerts() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return alerts_;
}

void AnomalyDetector::clear() {
    std::lock_guard<std::mutex> lk(mutex_);
    alerts_.clear();
}

}  // namespace scope
