// Listens on a TCP port for line-delimited JSON telemetry (see docs/protocol.md)
// and republishes each decoded packet onto the EventBus. Runs its accept/read
// loop on a background thread.
#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "telemetry/event_bus.hpp"

namespace scope {

class TelemetryServer {
public:
    TelemetryServer(EventBus& bus, int port = 5005);
    ~TelemetryServer();

    bool start();
    void stop();

    bool is_running() const { return running_; }
    bool client_connected() const { return client_connected_; }

private:
    void accept_loop();
    void serve_client(int client_fd);

    EventBus& bus_;
    int port_;
    int listen_fd_ = -1;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> client_connected_{false};
};

}  // namespace scope
