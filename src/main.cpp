#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#include "anomaly/anomaly_detector.hpp"
#include "instrumentation/device_monitor.hpp"
#include "instrumentation/mock_tracer.hpp"
#include "instrumentation/telemetry_server.hpp"
#include "storage/ring_buffer.hpp"
#include "telemetry/aggregator.hpp"
#include "telemetry/event_bus.hpp"
#include "tui/dashboard.hpp"
#include "util/log.hpp"

namespace {

void usage() {
    std::cout <<
        "llm-trace — local transformer instrumentation, tracing & TUI replay\n"
        "\n"
        "Usage: llm-trace [options]\n"
        "  -h, --help        show this help\n"
        "      --sim         start the built-in mock tracer immediately\n"
        "      --headless    no TUI; print rolling stats (for piping/CI)\n"
        "      --snapshot    render one populated frame to stdout and exit\n"
        "      --port N      TCP telemetry port to listen on (default 5005)\n"
        "\n"
        "Connect a real model with examples/hook.py (streams over TCP).\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace scope;

    bool sim = false;
    bool headless = false;
    bool snapshot = false;
    int snapshot_tab = 0;
    int port = 5005;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { usage(); return 0; }
        else if (arg == "--sim") sim = true;
        else if (arg == "--headless") headless = true;
        else if (arg == "--snapshot") { snapshot = true; sim = true; }
        else if (arg == "--snapshot-tab" && i + 1 < argc) {
            snapshot = true; sim = true; snapshot_tab = std::atoi(argv[++i]);
        }
        else if (arg == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else { std::cerr << "unknown option: " << arg << "\n"; usage(); return 1; }
    }

    // In TUI mode the dashboard owns the terminal, so divert logs to a file.
    if (!headless) log::set_sink_file("llm-trace.log");
    log::info("starting llm-trace (port %d, sim %d, headless %d)", port,
              static_cast<int>(sim), static_cast<int>(headless));

    EventBus bus;
    RingBuffer ring(10000);
    TelemetryAggregator agg;
    AnomalyDetector anomaly(bus);
    auto device = DeviceMonitor::create();
    TelemetryServer server(bus, port);
    MockTracer tracer(bus, *device);

    server.start();
    // Subscribers must be attached before any telemetry source starts, since
    // model_info is sent only once: a late subscriber would miss the topology.

    if (headless) {
        // The dashboard normally wires these consumers; do it manually here.
        bus.subscribe("*", [&](const TelemetryEvent& e) {
            ring.push(e);
            agg.process_event(e);
            anomaly.process_event(e);
        });
        if (sim) tracer.start();
        std::cout << "headless mode, telemetry on tcp/" << port
                  << ", Ctrl+C to stop\n";
        while (server.is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::cout << "events " << agg.total_events() << " | "
                      << agg.tokens_per_sec() << " tok/s | avg "
                      << agg.avg_layer_latency() << " ms | anomalies "
                      << anomaly.alerts().size() << std::endl;
        }
    } else if (snapshot) {
        // Render one populated frame to stdout, then exit. For docs/CI.
        Dashboard dash(bus, ring, agg, anomaly, *device, server, tracer);
        if (sim) tracer.start();
        std::cout << dash.snapshot(snapshot_tab) << std::endl;
    } else {
        Dashboard dash(bus, ring, agg, anomaly, *device, server, tracer);
        if (sim) tracer.start();
        dash.run();
    }

    tracer.stop();
    server.stop();
    log::info("llm-trace shut down");
    return 0;
}
