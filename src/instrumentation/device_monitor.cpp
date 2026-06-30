#include "instrumentation/device_monitor.hpp"

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <fstream>
#include <string>
#endif

namespace scope {

namespace {

#if defined(__APPLE__)
// CPU load is computed as the busy fraction between two host_statistics() ticks.
class AppleMonitor : public DeviceMonitor {
public:
    SystemStats get_stats() override {
        SystemStats s;

        int64_t mem_total = 0;
        size_t len = sizeof(mem_total);
        if (sysctlbyname("hw.memsize", &mem_total, &len, nullptr, 0) == 0) {
            s.ram_total_gb = mem_total / (1024.0 * 1024.0 * 1024.0);
        }

        // Used RAM = total - free - speculative pages.
        vm_size_t page_size = 0;
        host_page_size(mach_host_self(), &page_size);
        vm_statistics64_data_t vmstat;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                              reinterpret_cast<host_info64_t>(&vmstat),
                              &count) == KERN_SUCCESS) {
            double free_gb = (static_cast<double>(vmstat.free_count) +
                              vmstat.speculative_count) *
                             page_size / (1024.0 * 1024.0 * 1024.0);
            s.ram_used_gb = s.ram_total_gb - free_gb;
        }

        s.cpu_usage = sample_cpu();
        s.gpu_available = false;  // No portable GPU counter on macOS.
        s.gpu_name = "Apple GPU (counters unavailable)";
        return s;
    }

private:
    double sample_cpu() {
        host_cpu_load_info_data_t info;
        mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
        if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                            reinterpret_cast<host_info_t>(&info),
                            &count) != KERN_SUCCESS) {
            return 0.0;
        }
        uint64_t user = info.cpu_ticks[CPU_STATE_USER];
        uint64_t sys = info.cpu_ticks[CPU_STATE_SYSTEM];
        uint64_t nice = info.cpu_ticks[CPU_STATE_NICE];
        uint64_t idle = info.cpu_ticks[CPU_STATE_IDLE];
        uint64_t busy = user + sys + nice;
        uint64_t total = busy + idle;
        double pct = 0.0;
        if (have_prev_ && total > prev_total_) {
            uint64_t dbusy = busy - prev_busy_;
            uint64_t dtotal = total - prev_total_;
            pct = dtotal > 0 ? (100.0 * dbusy / dtotal) : 0.0;
        }
        prev_busy_ = busy;
        prev_total_ = total;
        have_prev_ = true;
        return pct;
    }

    uint64_t prev_busy_ = 0, prev_total_ = 0;
    bool have_prev_ = false;
};
#endif

#if defined(__linux__)
class LinuxMonitor : public DeviceMonitor {
public:
    SystemStats get_stats() override {
        SystemStats s;
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            double unit = si.mem_unit / (1024.0 * 1024.0 * 1024.0);
            s.ram_total_gb = si.totalram * unit;
            s.ram_used_gb = (si.totalram - si.freeram) * unit;
        }
        s.cpu_usage = sample_cpu();
        s.gpu_available = false;
        s.gpu_name = "GPU counters unavailable";
        return s;
    }

private:
    double sample_cpu() {
        std::ifstream f("/proc/stat");
        std::string cpu;
        uint64_t user, nice, sys, idle, iowait, irq, softirq;
        if (!(f >> cpu >> user >> nice >> sys >> idle >> iowait >> irq >>
              softirq)) {
            return 0.0;
        }
        uint64_t busy = user + nice + sys + irq + softirq;
        uint64_t total = busy + idle + iowait;
        double pct = 0.0;
        if (have_prev_ && total > prev_total_) {
            pct = 100.0 * (busy - prev_busy_) / (total - prev_total_);
        }
        prev_busy_ = busy;
        prev_total_ = total;
        have_prev_ = true;
        return pct;
    }
    uint64_t prev_busy_ = 0, prev_total_ = 0;
    bool have_prev_ = false;
};
#endif

// Used when no host counters are available.
class NullMonitor : public DeviceMonitor {
public:
    SystemStats get_stats() override {
        SystemStats s;
        s.ram_total_gb = 16.0;
        s.gpu_name = "N/A";
        return s;
    }
};

}  // namespace

std::unique_ptr<DeviceMonitor> DeviceMonitor::create() {
#if defined(__APPLE__)
    return std::make_unique<AppleMonitor>();
#elif defined(__linux__)
    return std::make_unique<LinuxMonitor>();
#else
    return std::make_unique<NullMonitor>();
#endif
}

}  // namespace scope
