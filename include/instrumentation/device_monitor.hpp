// Samples host hardware utilization for the performance dashboard. A factory
// returns the best monitor for the current platform; GPU stats fall back to a
// benign "not available" when no vendor library is present (e.g. on macOS).
#pragma once

#include <memory>
#include <string>

namespace scope {

struct SystemStats {
    double cpu_usage = 0.0;  // percent
    double ram_used_gb = 0.0;
    double ram_total_gb = 0.0;

    bool gpu_available = false;
    std::string gpu_name = "N/A";
    double gpu_utilization = 0.0;  // percent
    double vram_used_gb = 0.0;
    double vram_total_gb = 0.0;
};

class DeviceMonitor {
public:
    virtual ~DeviceMonitor() = default;
    virtual SystemStats get_stats() = 0;

    static std::unique_ptr<DeviceMonitor> create();
};

}  // namespace scope
