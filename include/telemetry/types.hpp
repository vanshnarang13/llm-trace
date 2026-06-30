// Core telemetry data model shared between the network/instrumentation layer
// and the TUI. Every value that crosses the TCP boundary is described here and
// (de)serialized with nlohmann::json. See docs/protocol.md for the wire format.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace scope {

// One-time description of the model under inspection.
struct ModelInfo {
    std::string name = "unknown";
    int layers = 0;
    int hidden_size = 0;
    int num_heads = 0;
    int vocab_size = 0;
    std::string quantization = "FP32";

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(ModelInfo, name, layers, hidden_size,
                                   num_heads, vocab_size, quantization)
};

// Shape / dtype / placement metadata for a single tensor.
struct TensorInfo {
    std::vector<int64_t> shape;
    std::string dtype = "float32";
    int64_t size_bytes = 0;
    std::string device = "CPU";

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(TensorInfo, shape, dtype, size_bytes, device)
};

// Cheap numerical summary of an activation tensor, computed sender-side so the
// raw tensor never has to leave the model process.
struct TensorStats {
    double mean = 0.0;
    double variance = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double sparsity = 0.0;  // percentage of ~zero entries, 0..100

    // The wire keys are "min"/"max"; we also accept "min_val"/"max_val" so
    // hand-written senders are forgiving.
    friend void to_json(nlohmann::json& j, const TensorStats& s) {
        j = nlohmann::json{{"mean", s.mean},
                           {"variance", s.variance},
                           {"min", s.min_val},
                           {"max", s.max_val},
                           {"sparsity", s.sparsity}};
    }
    friend void from_json(const nlohmann::json& j, TensorStats& s) {
        j.at("mean").get_to(s.mean);
        j.at("variance").get_to(s.variance);
        if (j.contains("min")) j.at("min").get_to(s.min_val);
        else if (j.contains("min_val")) j.at("min_val").get_to(s.min_val);
        if (j.contains("max")) j.at("max").get_to(s.max_val);
        else if (j.contains("max_val")) j.at("max_val").get_to(s.max_val);
        s.sparsity = j.value("sparsity", 0.0);
    }
};

// Attention probabilities for one submodule. matrices is [head][query][key];
// senders are expected to downsample large contexts before transmitting.
struct AttentionData {
    std::string layer_name;
    int num_heads = 0;
    int token_count = 0;
    std::vector<std::string> tokens;
    std::vector<std::vector<std::vector<float>>> matrices;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(AttentionData, layer_name, num_heads,
                                   token_count, tokens, matrices)
};

// A flagged numerical / system instability. Per the protocol the wire payload
// carries only severity/layer_name/description; timestamp comes from the event
// envelope, so it is optional on the way in.
struct AnomalyAlert {
    std::string timestamp;
    std::string severity = "WARNING";  // "WARNING" | "ERROR"
    std::string layer_name;
    std::string description;

    friend void to_json(nlohmann::json& j, const AnomalyAlert& a) {
        j = nlohmann::json{{"severity", a.severity},
                           {"layer_name", a.layer_name},
                           {"description", a.description}};
    }
    friend void from_json(const nlohmann::json& j, AnomalyAlert& a) {
        a.timestamp = j.value("timestamp", std::string{});
        a.severity = j.value("severity", std::string{"WARNING"});
        a.layer_name = j.value("layer_name", std::string{});
        a.description = j.value("description", std::string{});
    }
};

// The tagged union that travels over the wire. event_type selects which of the
// payload members is meaningful.
struct TelemetryEvent {
    std::string event_type;  // "model_info" | "layer_trace" | "attention_weights" | "anomaly"
    int64_t timestamp = 0;   // unix epoch milliseconds

    ModelInfo model_info;

    // layer_trace payload
    int64_t event_id = 0;
    std::string layer_name;
    std::string layer_type;
    std::string device = "CPU";
    double latency_ms = 0.0;
    TensorInfo input_tensor;
    TensorInfo output_tensor;
    TensorStats activation_stats;

    // attention_weights payload
    AttentionData attention;

    // anomaly payload
    AnomalyAlert anomaly;

    friend void to_json(nlohmann::json& j, const TelemetryEvent& e) {
        j = nlohmann::json{{"event_type", e.event_type},
                           {"timestamp", e.timestamp}};
        if (e.event_type == "model_info") {
            j["payload"] = e.model_info;
        } else if (e.event_type == "layer_trace") {
            j["payload"] = {{"event_id", e.event_id},
                            {"layer_name", e.layer_name},
                            {"layer_type", e.layer_type},
                            {"device", e.device},
                            {"latency_ms", e.latency_ms},
                            {"input", e.input_tensor},
                            {"output", e.output_tensor},
                            {"stats", e.activation_stats}};
        } else if (e.event_type == "attention_weights") {
            j["payload"] = e.attention;
        } else if (e.event_type == "anomaly") {
            j["payload"] = e.anomaly;
        }
    }

    friend void from_json(const nlohmann::json& j, TelemetryEvent& e) {
        j.at("event_type").get_to(e.event_type);
        e.timestamp = j.value("timestamp", int64_t{0});
        if (!j.contains("payload") || j["payload"].is_null()) return;
        const auto& p = j.at("payload");
        if (e.event_type == "model_info") {
            p.get_to(e.model_info);
        } else if (e.event_type == "layer_trace") {
            e.event_id = p.value("event_id", int64_t{0});
            e.layer_name = p.value("layer_name", std::string{});
            e.layer_type = p.value("layer_type", std::string{});
            e.device = p.value("device", std::string{"CPU"});
            e.latency_ms = p.value("latency_ms", 0.0);
            if (p.contains("input")) p.at("input").get_to(e.input_tensor);
            if (p.contains("output")) p.at("output").get_to(e.output_tensor);
            if (p.contains("stats")) p.at("stats").get_to(e.activation_stats);
        } else if (e.event_type == "attention_weights") {
            p.get_to(e.attention);
        } else if (e.event_type == "anomaly") {
            p.get_to(e.anomaly);
        }
    }
};

}  // namespace scope
