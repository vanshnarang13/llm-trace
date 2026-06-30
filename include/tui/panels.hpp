// Pure rendering helpers: each takes immutable telemetry state plus a focus
// flag and returns an FTXUI Element. They hold no state of their own so the
// dashboard can re-render freely on every frame.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <ftxui/dom/elements.hpp>

#include "anomaly/anomaly_detector.hpp"
#include "instrumentation/device_monitor.hpp"
#include "telemetry/aggregator.hpp"
#include "telemetry/types.hpp"

namespace scope::tui {

// Flat list of topology node paths for the given model, in display order.
// Shared by the renderer and the dashboard's keyboard navigation so selection
// indices always line up.
std::vector<std::string> topology_nodes(const ModelInfo& model);

ftxui::Element render_topology(const ModelInfo& model, int selected_idx,
                               const std::string& active_layer, bool focused);

ftxui::Element render_live_stream(const std::vector<TelemetryEvent>& events,
                                  int selected_idx, bool focused);

ftxui::Element render_attention(const AttentionData& attn, int head, int pan_x,
                                int pan_y, int viewport, float contrast,
                                bool focused);

ftxui::Element render_metrics(const TelemetryEvent& event, bool focused);

ftxui::Element render_anomalies(const std::vector<AnomalyAlert>& alerts,
                                bool focused);

ftxui::Element render_performance(const TelemetryAggregator& agg,
                                  const SystemStats& stats, bool focused);

ftxui::Element render_flame_graph(
    const TelemetryAggregator& agg, const ModelInfo& model,
    const std::unordered_map<std::string, bool>& expanded,
    const std::string& selected, bool focused);

ftxui::Element render_token_journey(const std::vector<TelemetryEvent>& events,
                                    int token_idx, bool focused);

}  // namespace scope::tui
