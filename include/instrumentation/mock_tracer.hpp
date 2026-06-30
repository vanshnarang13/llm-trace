// Synthetic telemetry source. Emulates a Llama-3-8B forward pass token by
// token, emitting model_info, layer_trace, attention_weights and the occasional
// anomaly so the TUI is fully exercised without a real model or Python.
#pragma once

#include <atomic>
#include <random>
#include <thread>

#include "instrumentation/device_monitor.hpp"
#include "telemetry/event_bus.hpp"

namespace scope {

class MockTracer {
public:
    MockTracer(EventBus& bus, DeviceMonitor& device);
    ~MockTracer();

    void start();
    void stop();
    bool is_running() const { return running_; }

private:
    void loop();
    void emit_token_pass(int token_index, int64_t& event_id);

    EventBus& bus_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::mt19937 rng_{std::random_device{}()};
};

}  // namespace scope
