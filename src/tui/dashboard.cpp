#include "tui/dashboard.hpp"

#include <algorithm>
#include <vector>

#include <chrono>
#include <thread>

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include "tui/panels.hpp"

namespace scope {

using namespace ftxui;

namespace {
ScreenInteractive* g_screen = nullptr;

// Focus-panel count per tab, so Tab cycling stays in range.
int focus_count(int tab) {
    switch (tab) {
        case 0: return 2;  // topology, stream
        case 1: return 2;  // attention, journey
        case 2: return 1;  // flame graph
        default: return 1;
    }
}
}  // namespace

Dashboard::Dashboard(EventBus& bus, RingBuffer& ring, TelemetryAggregator& agg,
                     AnomalyDetector& anomaly, DeviceMonitor& device,
                     TelemetryServer& server, MockTracer& tracer)
    : bus_(bus),
      ring_(ring),
      agg_(agg),
      anomaly_(anomaly),
      device_(device),
      server_(server),
      tracer_(tracer) {
    model_.name = "no model connected";
    model_.quantization = "N/A";

    // Open the top two flame-graph levels so the first frame is informative.
    flame_expanded_["Inference"] = true;
    flame_expanded_["layers"] = true;

    bus_.subscribe("*", [this](const TelemetryEvent& e) {
        on_event(e);
        if (g_screen) g_screen->PostEvent(Event::Custom);
    });
}

void Dashboard::on_event(const TelemetryEvent& e) {
    ring_.push(e);
    agg_.process_event(e);
    anomaly_.process_event(e);

    if (e.event_type == "model_info") {
        model_ = e.model_info;
    } else if (e.event_type == "layer_trace") {
        // Keep the inspector following the live stream until the user pins a
        // selection by focusing the topology / stream panels.
        if (focus_ == 0 && tab_ == 0 && selected_event_.layer_name.empty())
            selected_event_ = e;
    }
}

void Dashboard::select_topology_node() {
    auto nodes = tui::topology_nodes(model_);
    if (topo_idx_ < 0 || topo_idx_ >= static_cast<int>(nodes.size())) return;
    const std::string& node = nodes[topo_idx_];

    // Pull the most recent layer_trace for this node out of the ring.
    auto all = ring_.get_all();
    for (auto it = all.rbegin(); it != all.rend(); ++it) {
        if (it->event_type == "layer_trace" && it->layer_name == node) {
            selected_event_ = *it;
            break;
        }
    }
    // And its attention matrix, if it is an attention submodule.
    if (auto a = ring_.latest_attention(node)) {
        active_attention_ = *a;
        attn_pan_x_ = attn_pan_y_ = 0;
        attn_head_ = 0;
    }
}

Element Dashboard::build_ui() {
    {
        SystemStats stats = device_.get_stats();

        // Header.
        std::string mode = server_.client_connected()
                               ? "LIVE [tcp client]"
                               : (tracer_.is_running() ? "SIM [mock tracer]"
                                                       : "IDLE");
        Color mode_col = server_.client_connected()
                             ? Color::Green
                             : (tracer_.is_running() ? Color::Yellow
                                                     : Color::GrayLight);
        auto header = vbox({hbox({
                          text(" LLM-TRACE ") | bold | bgcolor(Color::Cyan) |
                              color(Color::Black),
                          text(" local transformer telemetry ") | color(Color::Cyan),
                          text(" mode: ") | bold,
                          text(mode) | color(mode_col),
                          filler(),
                          text("[1]Dashboard [2]Attention [3]CallTree [4]Help") |
                              dim,
                      }), separator()});

        // Tab content.
        Element content;
        if (tab_ == 0) {
            auto left = tui::render_topology(model_, topo_idx_,
                                             selected_event_.layer_name,
                                             focus_ == 0) |
                        size(WIDTH, EQUAL, 30);
            auto mid = tui::render_live_stream(ring_.get_all(), stream_idx_,
                                               focus_ == 1) |
                       flex;
            auto right = vbox({
                             tui::render_metrics(selected_event_, false) |
                                 size(HEIGHT, EQUAL, 16),
                             tui::render_anomalies(anomaly_.alerts(), false) | flex,
                             tui::render_performance(agg_, stats, false) |
                                 size(HEIGHT, EQUAL, 13),
                         }) |
                         size(WIDTH, EQUAL, 52);
            content = hbox({left, mid, right}) | flex;
        } else if (tab_ == 1) {
            auto left = tui::render_attention(active_attention_, attn_head_,
                                              attn_pan_x_, attn_pan_y_,
                                              attn_viewport_, attn_contrast_,
                                              focus_ == 0) |
                        flex;
            auto right = tui::render_token_journey(ring_.get_all(), journey_idx_,
                                                   focus_ == 1) |
                         size(WIDTH, EQUAL, 56);
            content = hbox({left, right}) | flex;
        } else if (tab_ == 2) {
            auto left = tui::render_flame_graph(agg_, model_, flame_expanded_,
                                                flame_selected_, focus_ == 0) |
                        flex;
            auto right = tui::render_performance(agg_, stats, false) |
                         size(WIDTH, EQUAL, 54);
            content = hbox({left, right}) | flex;
        } else {
            auto key = [](const std::string& k, const std::string& d) {
                return hbox({text("  " + k) | color(Color::Yellow) | bold |
                                 size(WIDTH, EQUAL, 20),
                             text(d)});
            };
            content =
                window(text(" KEYBINDINGS ") | bold | color(Color::Cyan),
                       vbox({
                           key("1 – 4", "switch tab"),
                           key("Tab / Shift+Tab", "cycle panel focus"),
                           key("j / k", "navigate lists (down / up)"),
                           key("Enter / Space", "select layer · toggle flame node"),
                           key("h j k l", "pan attention matrix (tab 2)"),
                           key("+ / -", "attention contrast"),
                           key("< / >", "previous / next attention head"),
                           key("[ / ]", "shrink / grow attention viewport"),
                           key("p", "pause / resume mock simulator"),
                           key("q / Esc", "quit"),
                       })) |
                flex;
        }

        auto footer = vbox(
            {separator(),
             hbox({text(" [Tab] focus  [1-4] tabs  [j/k] nav  [p] sim  [q] quit ") |
                   dim})});

        return vbox({header, content, footer}) | flex;
    }
}

void Dashboard::run() {
    auto screen = ScreenInteractive::Fullscreen();
    g_screen = &screen;

    auto renderer = Renderer([this] { return build_ui(); });
    auto component = CatchEvent(renderer, [this, &screen](Event e) {
        if (e == Event::Character('q') || e == Event::Escape) {
            screen.ExitLoopClosure()();
            return true;
        }
        // Tab switching.
        for (int i = 0; i < 4; ++i) {
            if (e == Event::Character(std::string(1, char('1' + i)))) {
                tab_ = i;
                focus_ = 0;
                return true;
            }
        }
        if (e == Event::Tab) {
            focus_ = (focus_ + 1) % focus_count(tab_);
            return true;
        }
        if (e == Event::TabReverse) {
            int n = focus_count(tab_);
            focus_ = (focus_ + n - 1) % n;
            return true;
        }
        if (e == Event::Character('p')) {
            if (tracer_.is_running()) tracer_.stop();
            else tracer_.start();
            return true;
        }

        const bool down = e == Event::Character('j') || e == Event::ArrowDown;
        const bool up = e == Event::Character('k') || e == Event::ArrowUp;
        const bool select = e == Event::Return || e == Event::Character(' ');

        // Tab 0.
        if (tab_ == 0 && focus_ == 0) {  // topology
            int max = static_cast<int>(tui::topology_nodes(model_).size()) - 1;
            if (down) { topo_idx_ = std::min(max, topo_idx_ + 1); return true; }
            if (up) { topo_idx_ = std::max(0, topo_idx_ - 1); return true; }
            if (select) { select_topology_node(); return true; }
        } else if (tab_ == 0 && focus_ == 1) {  // stream
            int count = 0;
            for (const auto& ev : ring_.get_all())
                if (ev.event_type == "layer_trace") ++count;
            auto sync = [&] {
                auto all = ring_.get_all();
                std::vector<const TelemetryEvent*> tr;
                for (const auto& ev : all)
                    if (ev.event_type == "layer_trace") tr.push_back(&ev);
                if (!tr.empty() && stream_idx_ < static_cast<int>(tr.size()))
                    selected_event_ = *tr[tr.size() - 1 - stream_idx_];
            };
            if (down) { stream_idx_ = std::min(std::max(0, count - 1), stream_idx_ + 1); sync(); return true; }
            if (up) { stream_idx_ = std::max(0, stream_idx_ - 1); sync(); return true; }
        }

        // Tab 1.
        if (tab_ == 1 && focus_ == 0) {  // attention matrix
            if (e == Event::Character('h') || e == Event::ArrowLeft) { attn_pan_x_ = std::max(0, attn_pan_x_ - 1); return true; }
            if (e == Event::Character('l') || e == Event::ArrowRight) { ++attn_pan_x_; return true; }
            if (e == Event::Character('k') || e == Event::ArrowUp) { attn_pan_y_ = std::max(0, attn_pan_y_ - 1); return true; }
            if (e == Event::Character('j') || e == Event::ArrowDown) { ++attn_pan_y_; return true; }
            if (e == Event::Character('+') || e == Event::Character('=')) { attn_contrast_ = std::min(5.0f, attn_contrast_ + 0.2f); return true; }
            if (e == Event::Character('-')) { attn_contrast_ = std::max(0.2f, attn_contrast_ - 0.2f); return true; }
            if (e == Event::Character('<') || e == Event::Character(',')) { attn_head_ = std::max(0, attn_head_ - 1); return true; }
            if (e == Event::Character('>') || e == Event::Character('.')) { attn_head_ = std::min(std::max(0, active_attention_.num_heads - 1), attn_head_ + 1); return true; }
            if (e == Event::Character('[')) { attn_viewport_ = std::max(4, attn_viewport_ - 1); return true; }
            if (e == Event::Character(']')) { attn_viewport_ = std::min(24, attn_viewport_ + 1); return true; }
        } else if (tab_ == 1 && focus_ == 1) {  // token journey
            if (down) { ++journey_idx_; return true; }
            if (up) { journey_idx_ = std::max(0, journey_idx_ - 1); return true; }
        }

        // Tab 2: flame graph.
        if (tab_ == 2 && focus_ == 0) {
            // Build the currently visible node list to navigate.
            std::vector<std::string> visible = {"Inference"};
            auto open = [&](const std::string& p) {
                auto it = flame_expanded_.find(p);
                return it != flame_expanded_.end() && it->second;
            };
            if (open("Inference")) {
                visible.push_back("embed_tokens");
                visible.push_back("layers");
                if (open("layers")) {
                    for (int i = 0; i < model_.layers; ++i) {
                        std::string p = "layers." + std::to_string(i);
                        visible.push_back(p);
                        if (open(p)) {
                            visible.push_back(p + ".input_layernorm");
                            visible.push_back(p + ".self_attn");
                            visible.push_back(p + ".post_attention_layernorm");
                            visible.push_back(p + ".mlp");
                        }
                    }
                }
                visible.push_back("norm");
                visible.push_back("lm_head");
            }
            auto it = std::find(visible.begin(), visible.end(), flame_selected_);
            int idx = it == visible.end() ? 0 : std::distance(visible.begin(), it);
            if (down) { idx = std::min<int>(visible.size() - 1, idx + 1); flame_selected_ = visible[idx]; return true; }
            if (up) { idx = std::max(0, idx - 1); flame_selected_ = visible[idx]; return true; }
            if (select) { flame_expanded_[flame_selected_] = !open(flame_selected_); return true; }
        }
        return false;
    });

    screen.Loop(component);
    g_screen = nullptr;
}

std::string Dashboard::snapshot(int tab, int width, int height, int settle_ms) {
    // Let the (already running) telemetry source populate the caches.
    if (settle_ms > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
    tab_ = tab;
    // The attention tab needs a layer pinned; pick the first one that has
    // attention data so the heatmap renders instead of the empty state.
    if (tab == 1) {
        auto nodes = tui::topology_nodes(model_);
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (ring_.latest_attention(nodes[i])) {
                topo_idx_ = i;
                select_topology_node();
                break;
            }
        }
    }
    auto screen = Screen::Create(Dimension::Fixed(width), Dimension::Fixed(height));
    Element frame = build_ui();
    Render(screen, frame);
    return screen.ToString();
}

}  // namespace scope
