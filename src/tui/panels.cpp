#include "tui/panels.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace scope::tui {

using namespace ftxui;

namespace {

Element bar(double pct, Color col, int width = 14) {
    pct = std::clamp(pct, 0.0, 100.0);
    int filled = static_cast<int>(std::lround(pct / 100.0 * width));
    std::string s;
    for (int i = 0; i < filled; ++i) s += "█";
    for (int i = filled; i < width; ++i) s += "░";
    return text(s) | color(col);
}

std::string fixed(double v, int prec) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

std::string shape_str(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) s += ", ";
    }
    return s + "]";
}

std::string clock_str(int64_t epoch_ms) {
    if (epoch_ms <= 0) return "--:--:--";
    std::time_t secs = epoch_ms / 1000;
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&secs));
    std::ostringstream ss;
    ss << buf << '.' << std::setw(3) << std::setfill('0') << (epoch_ms % 1000);
    return ss.str();
}

bool is_cuda(const std::string& dev) {
    return dev.find("CUDA") != std::string::npos ||
           dev.find("cuda") != std::string::npos;
}

Element titled(const std::string& title, int index, bool focused, Element body) {
    std::string label = " " + std::to_string(index) + ". " + title +
                        (focused ? " ◀ " : " ");
    Element header = text(label) | bold;
    if (focused) header = header | color(Color::Cyan);
    return window(header, std::move(body));
}

}  // namespace

std::vector<std::string> topology_nodes(const ModelInfo& model) {
    std::vector<std::string> nodes = {"embed_tokens"};
    for (int i = 0; i < model.layers; ++i) {
        std::string p = "layers." + std::to_string(i);
        nodes.push_back(p);
        nodes.push_back(p + ".self_attn");
        nodes.push_back(p + ".mlp");
    }
    nodes.push_back("norm");
    nodes.push_back("lm_head");
    return nodes;
}

Element render_topology(const ModelInfo& model, int selected_idx,
                        const std::string& active_layer, bool focused) {
    Elements items;
    items.push_back(text("Model: " + model.name) | bold | color(Color::Green));
    items.push_back(hbox({text("layers ") | dim, text(std::to_string(model.layers)),
                          text("  heads ") | dim, text(std::to_string(model.num_heads))}));
    items.push_back(hbox({text("hidden ") | dim, text(std::to_string(model.hidden_size)),
                          text("  quant ") | dim,
                          text(model.quantization) | color(Color::Yellow)}));
    items.push_back(separator());

    auto nodes = topology_nodes(model);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const std::string& path = nodes[i];
        // Indentation + glyph based on nesting depth.
        std::string label;
        if (path == "embed_tokens" || path == "norm" || path == "lm_head") {
            label = "▪ " + path;
        } else if (path.find(".self_attn") != std::string::npos) {
            label = "   ● self_attn";
        } else if (path.find(".mlp") != std::string::npos) {
            label = "   ● mlp";
        } else {
            label = " ▸ " + path;  // layers.N
        }

        bool selected = static_cast<int>(i) == selected_idx;
        bool active = !active_layer.empty() && path == active_layer;

        Element el = text(label);
        if (active) el = hbox({el, text("  [capture]") | color(Color::Cyan)});
        if (selected) {
            el = el | bold | color(Color::White) |
                 bgcolor(focused ? Color::Blue : Color::GrayDark);
        } else if (active) {
            el = el | color(Color::Cyan);
        }
        items.push_back(el);
    }

    return titled("MODEL TOPOLOGY", 1, focused,
                  vbox(std::move(items)) | yframe | flex);
}

Element render_live_stream(const std::vector<TelemetryEvent>& events,
                           int selected_idx, bool focused) {
    Elements rows;
    rows.push_back(hbox({
                       text("ID") | size(WIDTH, EQUAL, 7) | bold,
                       text("TIMESTAMP") | size(WIDTH, EQUAL, 13) | bold,
                       text("TYPE") | size(WIDTH, EQUAL, 14) | bold,
                       text("LAYER") | flex | bold,
                       text("DEVICE") | size(WIDTH, EQUAL, 9) | bold,
                       text("LATENCY") | size(WIDTH, EQUAL, 11) | bold,
                   }));
    rows.push_back(separator());

    std::vector<const TelemetryEvent*> traces;
    for (const auto& e : events)
        if (e.event_type == "layer_trace") traces.push_back(&e);

    int idx = 0;
    for (auto it = traces.rbegin(); it != traces.rend(); ++it, ++idx) {
        const auto& e = **it;
        bool sel = idx == selected_idx;
        Color dev_col = is_cuda(e.device) ? Color::Cyan : Color::GrayLight;
        auto row = hbox({
            text(std::to_string(e.event_id)) | size(WIDTH, EQUAL, 7) | dim,
            text(clock_str(e.timestamp)) | size(WIDTH, EQUAL, 13) | dim,
            text(e.layer_type) | size(WIDTH, EQUAL, 14),
            text(e.layer_name) | flex,
            text(e.device) | size(WIDTH, EQUAL, 9) | color(dev_col),
            text(fixed(e.latency_ms, 3) + " ms") | size(WIDTH, EQUAL, 11) |
                color(Color::Cyan),
        });
        if (sel)
            row = row | bgcolor(focused ? Color::Blue : Color::GrayDark) |
                  color(Color::White);
        rows.push_back(row);
    }
    if (traces.empty())
        rows.push_back(text("No packets yet — run --sim or connect a tracer.") |
                       dim | center);

    return titled("LIVE PACKET STREAM", 2, focused,
                  vbox(std::move(rows)) | yframe | flex);
}

Element render_attention(const AttentionData& attn, int head, int pan_x,
                         int pan_y, int viewport, float contrast, bool focused) {
    if (attn.matrices.empty() || attn.token_count == 0 ||
        head >= static_cast<int>(attn.matrices.size())) {
        return titled("ATTENTION MATRIX", 3, focused,
                      vbox({text("No attention weights for the selected layer.") | dim,
                            text("Pick a *.self_attn node in the topology.") | dim}) |
                          center | flex);
    }

    const auto& m = attn.matrices[head];
    int n = attn.token_count;
    int vp = std::clamp(viewport, 4, 24);
    int r0 = std::clamp(pan_y, 0, std::max(0, n - vp));
    int c0 = std::clamp(pan_x, 0, std::max(0, n - vp));
    int r1 = std::min(n, r0 + vp);
    int c1 = std::min(n, c0 + vp);

    // Shannon entropy of the head, averaged over query rows.
    double entropy = 0.0;
    for (int q = 0; q < n; ++q)
        for (int k = 0; k < n; ++k) {
            float w = m[q][k];
            if (w > 1e-6f) entropy -= w * std::log2(w);
        }
    entropy /= std::max(1, n);

    Elements lines;
    std::ostringstream vp_info;
    vp_info << "view [" << c0 << "-" << (c1 - 1) << "] x [" << r0 << "-"
            << (r1 - 1) << "]";
    lines.push_back(hbox({text("head ") | dim,
                          text(std::to_string(head)) | color(Color::Yellow) | bold,
                          text("  hjkl pan · +/- contrast · </> head") | dim |
                              flex,
                          text(vp_info.str()) | color(Color::Yellow)}));
    lines.push_back(separator());

    Elements header = {text("query \\ key") | size(WIDTH, EQUAL, 12), separator()};
    for (int c = c0; c < c1; ++c) {
        std::string lab = attn.tokens[c];
        if (lab.size() > 7) lab = lab.substr(0, 6) + "…";
        header.push_back(text(lab) | size(WIDTH, EQUAL, 6) | center);
    }
    lines.push_back(hbox(std::move(header)));
    lines.push_back(separator());

    for (int r = r0; r < r1; ++r) {
        std::string lab = attn.tokens[r];
        if (lab.size() > 11) lab = lab.substr(0, 10) + "…";
        Elements cells = {text(lab) | size(WIDTH, EQUAL, 12), separator()};
        for (int c = c0; c < c1; ++c) {
            float w = m[r][c] * contrast;
            std::string glyph = "  ";
            Color col = Color::Default;
            if (w >= 0.35f) { glyph = "██"; col = Color::Cyan; }
            else if (w >= 0.18f) { glyph = "▓▓"; col = Color::Cyan; }
            else if (w >= 0.08f) { glyph = "▒▒"; col = Color::Blue; }
            else if (w >= 0.02f) { glyph = "░░"; col = Color::GrayDark; }
            cells.push_back(text(glyph) | size(WIDTH, EQUAL, 6) | center | color(col));
        }
        lines.push_back(hbox(std::move(cells)));
    }

    lines.push_back(separator());
    lines.push_back(text("avg entropy " + fixed(entropy, 3) + " bits · contrast " +
                         fixed(contrast, 1) + "x") |
                    color(Color::Green) | bold);

    std::string title = "ATTENTION · " + attn.layer_name;
    return titled(title, 3, focused, vbox(std::move(lines)) | flex);
}

Element render_metrics(const TelemetryEvent& event, bool focused) {
    if (event.event_type != "layer_trace" || event.layer_name.empty()) {
        return titled("RUNTIME METRICS", 4, focused,
                      text("Select a layer to inspect tensors.") | dim | center |
                          flex);
    }
    const auto& s = event.activation_stats;
    const auto& in = event.input_tensor;
    const auto& out = event.output_tensor;
    double out_mb = out.size_bytes / (1024.0 * 1024.0);

    Elements items = {
        hbox({text("path   ") | bold, text(event.layer_name) | color(Color::Green)}),
        hbox({text("type   ") | bold, text(event.layer_type)}),
        hbox({text("device ") | bold,
              text(event.device) |
                  color(is_cuda(event.device) ? Color::Cyan : Color::GrayLight)}),
        separator(),
        text("TENSORS") | bold | color(Color::Yellow),
        hbox({text("  in   ") | dim, text(shape_str(in.shape))}),
        hbox({text("  out  ") | dim, text(shape_str(out.shape)),
              text("  " + out.dtype) | dim}),
        hbox({text("  size ") | dim, text(fixed(out_mb, 3) + " MB")}),
        separator(),
        text("ACTIVATIONS") | bold | color(Color::Yellow),
        hbox({text("  mean ") | dim, text(fixed(s.mean, 5))}),
        hbox({text("  var  ") | dim, text(fixed(s.variance, 5))}),
        hbox({text("  min  ") | dim, text(fixed(s.min_val, 4)),
              text("  max ") | dim, text(fixed(s.max_val, 4))}),
        hbox({text("  spars") | dim, text(" "), bar(s.sparsity, Color::Yellow),
              text(" " + fixed(s.sparsity, 1) + "%")}),
    };
    return titled("RUNTIME METRICS", 4, focused, vbox(std::move(items)) | flex);
}

Element render_anomalies(const std::vector<AnomalyAlert>& alerts, bool focused) {
    Elements items;
    for (auto it = alerts.rbegin(); it != alerts.rend(); ++it) {
        Color c = (it->severity == "ERROR") ? Color::Red : Color::Yellow;
        std::string mark = (it->severity == "ERROR") ? "✖" : "⚠";
        items.push_back(hbox({text(it->timestamp + " ") | dim,
                              text(mark + " ") | color(c) | bold,
                              text(it->layer_name + ": ") | bold,
                              text(it->description) | color(c)}));
    }
    if (alerts.empty())
        items.push_back(text("No anomalies flagged.") | dim | center | flex);
    return titled("NUMERICAL ANOMALY LEDGER", 5, focused,
                  vbox(std::move(items)) | yframe | flex);
}

Element render_performance(const TelemetryAggregator& agg,
                           const SystemStats& stats, bool focused) {
    double ram_pct = stats.ram_total_gb > 0
                         ? stats.ram_used_gb / stats.ram_total_gb * 100.0
                         : 0.0;
    Elements items;
    items.push_back(hbox({text("CPU  ") | dim, bar(stats.cpu_usage, Color::Cyan),
                          text(" " + fixed(stats.cpu_usage, 0) + "%")}));
    items.push_back(hbox({text("RAM  ") | dim, bar(ram_pct, Color::GrayLight),
                          text(" " + fixed(stats.ram_used_gb, 1) + "/" +
                               fixed(stats.ram_total_gb, 1) + " GB")}));
    if (stats.gpu_available) {
        items.push_back(hbox({text("GPU  ") | dim,
                              bar(stats.gpu_utilization, Color::Cyan),
                              text(" " + fixed(stats.gpu_utilization, 0) + "%")}));
    } else {
        items.push_back(hbox({text("GPU  ") | dim, text(stats.gpu_name) | dim}));
    }
    items.push_back(separator());
    items.push_back(hbox({text("speed ") | bold,
                          text(fixed(agg.tokens_per_sec(), 2) + " tok/s") |
                              color(Color::Green) | bold,
                          text("  avg ") | bold,
                          text(fixed(agg.avg_layer_latency(), 2) + " ms") |
                              color(Color::Cyan),
                          text("  events ") | bold,
                          text(std::to_string(agg.total_events())) | dim}));
    items.push_back(separator());
    items.push_back(text("SLOWEST LAYERS") | bold | color(Color::Yellow));
    auto slow = agg.slowest_layers(5);
    double maxl = 0.0;
    for (const auto& s : slow) maxl = std::max(maxl, s.avg_latency_ms);
    for (const auto& s : slow) {
        double ratio = maxl > 0 ? s.avg_latency_ms / maxl * 100.0 : 0.0;
        items.push_back(hbox({text(s.name) | size(WIDTH, EQUAL, 34) | dim,
                              bar(ratio, Color::Cyan, 10),
                              text(" " + fixed(s.avg_latency_ms, 3) + " ms") |
                                  color(Color::Cyan)}));
    }
    if (slow.empty()) items.push_back(text("profiling…") | dim);
    return titled("PERFORMANCE", 6, focused, vbox(std::move(items)) | flex);
}

Element render_flame_graph(
    const TelemetryAggregator& agg, const ModelInfo& model,
    const std::unordered_map<std::string, bool>& expanded,
    const std::string& selected, bool focused) {
    // Look up averaged latency per layer path.
    std::unordered_map<std::string, double> avg;
    for (const auto& l : agg.slowest_layers(100000)) avg[l.name] = l.avg_latency_ms;

    auto sum_layer = [&](int i) {
        std::string p = "layers." + std::to_string(i);
        return avg[p + ".input_layernorm"] + avg[p + ".self_attn"] +
               avg[p + ".post_attention_layernorm"] + avg[p + ".mlp"];
    };
    double layers_total = 0.0;
    for (int i = 0; i < model.layers; ++i) layers_total += sum_layer(i);
    double total = avg["embed_tokens"] + layers_total + avg["norm"] + avg["lm_head"];

    Elements rows;
    auto is_open = [&](const std::string& p) {
        auto it = expanded.find(p);
        return it == expanded.end() ? false : it->second;
    };
    auto add = [&](const std::string& path, const std::string& label, double t,
                   int indent, bool has_children) {
        double ratio = total > 0 ? t / total * 100.0 : 0.0;
        std::string pad(indent * 2, ' ');
        std::string glyph =
            has_children ? (is_open(path) ? "▼ " : "▶ ") : "● ";
        int w = 10, filled = static_cast<int>(ratio / 100.0 * w);
        std::string b;
        for (int i = 0; i < filled; ++i) b += "█";
        for (int i = filled; i < w; ++i) b += " ";
        auto row = hbox({text(pad + glyph + label) | flex,
                         text(b) | color(Color::Cyan),
                         text(" " + fixed(t, 2) + " ms (" + fixed(ratio, 1) + "%)") |
                             size(WIDTH, EQUAL, 20) | color(Color::Yellow)});
        if (path == selected)
            row = row | bgcolor(focused ? Color::Blue : Color::GrayDark) |
                  color(Color::White);
        rows.push_back(row);
    };

    add("Inference", "Inference loop", total, 0, true);
    if (is_open("Inference")) {
        add("embed_tokens", "embed_tokens", avg["embed_tokens"], 1, false);
        add("layers", "transformer blocks", layers_total, 1, true);
        if (is_open("layers")) {
            for (int i = 0; i < model.layers; ++i) {
                std::string p = "layers." + std::to_string(i);
                add(p, "layer " + std::to_string(i), sum_layer(i), 2, true);
                if (is_open(p)) {
                    add(p + ".input_layernorm", "input_layernorm",
                        avg[p + ".input_layernorm"], 3, false);
                    add(p + ".self_attn", "self_attn", avg[p + ".self_attn"], 3,
                        false);
                    add(p + ".post_attention_layernorm", "post_attn_layernorm",
                        avg[p + ".post_attention_layernorm"], 3, false);
                    add(p + ".mlp", "mlp", avg[p + ".mlp"], 3, false);
                }
            }
        }
        add("norm", "output_norm", avg["norm"], 1, false);
        add("lm_head", "lm_head", avg["lm_head"], 1, false);
    }

    if (total <= 0.0)
        rows = {text("No trace data yet — start the simulator (p).") | dim};

    return titled("CALL TREE / FLAME PROFILE", 1, focused,
                  vbox(std::move(rows)) | yframe | flex);
}

Element render_token_journey(const std::vector<TelemetryEvent>& events,
                             int token_idx, bool focused) {
    // Split the trace stream into per-token cycles delimited by embed_tokens.
    std::vector<std::vector<const TelemetryEvent*>> cycles;
    std::vector<const TelemetryEvent*> cur;
    for (const auto& e : events) {
        if (e.event_type != "layer_trace") continue;
        if (e.layer_name == "embed_tokens" && !cur.empty()) {
            cycles.push_back(cur);
            cur.clear();
        }
        cur.push_back(&e);
    }
    if (!cur.empty()) cycles.push_back(cur);

    if (cycles.empty()) {
        return titled("TOKEN ACTIVATION JOURNEY", 2, focused,
                      text("No token cycles logged yet.") | dim | center | flex);
    }
    int idx = std::clamp(token_idx, 0, static_cast<int>(cycles.size()) - 1);
    const auto& cycle = cycles[idx];

    Elements rows;
    rows.push_back(hbox({text("token cycle ") | bold,
                         text("#" + std::to_string(idx)) | color(Color::Yellow) |
                             bold,
                         text(" of " + std::to_string(cycles.size())) | dim,
                         text("  (j/k to step)") | dim}));
    rows.push_back(separator());
    int stage = 0;
    for (const auto* e : cycle) {
        const auto& s = e->activation_stats;
        rows.push_back(hbox({
            text("s" + std::to_string(stage++)) | size(WIDTH, EQUAL, 4) |
                color(Color::Cyan),
            text("→ " + e->layer_name) | size(WIDTH, EQUAL, 34),
            text("μ " + fixed(s.mean, 4)) | dim | size(WIDTH, EQUAL, 12),
            text("max " + fixed(s.max_val, 3)) | dim | size(WIDTH, EQUAL, 12),
            text("spars " + fixed(s.sparsity, 1) + "%") | color(Color::Yellow),
        }));
    }
    return titled("TOKEN ACTIVATION JOURNEY", 2, focused,
                  vbox(std::move(rows)) | yframe | flex);
}

}  // namespace scope::tui
