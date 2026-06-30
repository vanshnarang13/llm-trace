#include "instrumentation/telemetry_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>

#include <nlohmann/json.hpp>

#include "util/log.hpp"

namespace scope {

TelemetryServer::TelemetryServer(EventBus& bus, int port)
    : bus_(bus), port_(port) {}

TelemetryServer::~TelemetryServer() { stop(); }

bool TelemetryServer::start() {
    if (running_) return true;

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        log::error("telemetry: socket() failed: %s", std::strerror(errno));
        return false;
    }
    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
        0) {
        log::error("telemetry: bind(%d) failed: %s", port_,
                   std::strerror(errno));
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    if (::listen(listen_fd_, 1) < 0) {
        log::error("telemetry: listen() failed: %s", std::strerror(errno));
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    running_ = true;
    thread_ = std::thread(&TelemetryServer::accept_loop, this);
    log::info("telemetry: listening on tcp/%d", port_);
    return true;
}

void TelemetryServer::stop() {
    if (!running_.exchange(false)) return;
    // Shutting down the listening socket unblocks accept().
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (thread_.joinable()) thread_.join();
}

void TelemetryServer::accept_loop() {
    while (running_) {
        int client = ::accept(listen_fd_, nullptr, nullptr);
        if (client < 0) {
            if (!running_) break;
            continue;
        }
        client_connected_ = true;
        log::info("telemetry: client connected");
        serve_client(client);
        client_connected_ = false;
        log::info("telemetry: client disconnected");
        ::close(client);
    }
}

void TelemetryServer::serve_client(int client_fd) {
    std::string pending;
    char chunk[8192];
    while (running_) {
        ssize_t n = ::recv(client_fd, chunk, sizeof(chunk), 0);
        if (n <= 0) break;  // peer closed or error
        pending.append(chunk, static_cast<size_t>(n));

        // Each complete line is one JSON packet.
        std::size_t nl;
        while ((nl = pending.find('\n')) != std::string::npos) {
            std::string line = pending.substr(0, nl);
            pending.erase(0, nl + 1);
            if (line.empty()) continue;
            try {
                auto j = nlohmann::json::parse(line);
                bus_.publish(j.get<TelemetryEvent>());
            } catch (const std::exception& e) {
                std::string snippet = line.substr(0, 120);
                log::warn("telemetry: dropped malformed packet: %s | %s",
                          e.what(), snippet.c_str());
            }
        }
    }
}

}  // namespace scope
