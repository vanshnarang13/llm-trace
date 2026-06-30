// The interactive dashboard. Owns all UI state, subscribes to the EventBus to
// keep its caches (ring buffer, aggregator, anomaly ledger) warm, and runs the
// FTXUI event/render loop until the user quits.
#pragma once

#include <string>
#include <unordered_map>

#include <ftxui/dom/elements.hpp>

#include "anomaly/anomaly_detector.hpp"
#include "instrumentation/device_monitor.hpp"
#include "instrumentation/mock_tracer.hpp"
#include "instrumentation/telemetry_server.hpp"
#include "storage/ring_buffer.hpp"
#include "telemetry/aggregator.hpp"
#include "telemetry/event_bus.hpp"
#include "telemetry/types.hpp"

namespace scope {

class Dashboard {
public:
    Dashboard(EventBus& bus, RingBuffer& ring, TelemetryAggregator& agg,
              AnomalyDetector& anomaly, DeviceMonitor& device,
              TelemetryServer& server, MockTracer& tracer);

    void run();

    // Render a single frame of the given tab to a fixed-size off-screen buffer
    // and return it as a string (ANSI-colored). Lets telemetry settle for
    // settle_ms first so the panels are populated. Used for docs/CI snapshots.
    std::string snapshot(int tab = 0, int width = 150, int height = 44,
                         int settle_ms = 1500);

private:
    ftxui::Element build_ui();  // builds the full frame from current state
    void on_event(const TelemetryEvent& event);
    void select_topology_node();  // sync caches to the selected tree node

    EventBus& bus_;
    RingBuffer& ring_;
    TelemetryAggregator& agg_;
    AnomalyDetector& anomaly_;
    DeviceMonitor& device_;
    TelemetryServer& server_;
    MockTracer& tracer_;

    // Navigation state.
    int tab_ = 0;
    int focus_ = 0;

    int topo_idx_ = 0;
    int stream_idx_ = 0;
    int journey_idx_ = 0;

    int attn_head_ = 0;
    int attn_pan_x_ = 0;
    int attn_pan_y_ = 0;
    int attn_viewport_ = 8;
    float attn_contrast_ = 1.0f;

    std::string flame_selected_ = "Inference";
    std::unordered_map<std::string, bool> flame_expanded_;

    // Cached "current selection" derived from the stream / topology.
    ModelInfo model_;
    TelemetryEvent selected_event_;
    AttentionData active_attention_;
};

}  // namespace scope
