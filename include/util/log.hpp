// Minimal thread-safe logger. The TUI owns the terminal once it starts, so by
// default logs are buffered to a file rather than stdout; call set_sink_file()
// before launching the dashboard. Headless mode logs to stderr.
#pragma once

#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>
#include <utility>

namespace scope::log {

namespace detail {
inline std::mutex& mutex() {
    static std::mutex m;
    return m;
}
inline std::FILE*& sink() {
    static std::FILE* f = stderr;
    return f;
}
inline std::string stamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    return buf;
}
template <typename... Args>
void emit(const char* level, const std::string& fmt, Args&&... args) {
    std::lock_guard<std::mutex> lk(mutex());
    std::FILE* out = sink();
    std::fprintf(out, "[%s] %-5s ", stamp().c_str(), level);
    // fmt is a printf format string supplied by trusted call sites.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
    std::fprintf(out, fmt.c_str(), std::forward<Args>(args)...);
#pragma GCC diagnostic pop
    std::fprintf(out, "\n");
    std::fflush(out);
}
}  // namespace detail

// Redirect all subsequent logs to a file (created/truncated). Returns false if
// the file could not be opened, leaving the previous sink intact.
inline bool set_sink_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::lock_guard<std::mutex> lk(detail::mutex());
    if (detail::sink() != stderr && detail::sink() != stdout) {
        std::fclose(detail::sink());
    }
    detail::sink() = f;
    return true;
}

template <typename... Args>
void info(const std::string& fmt, Args&&... args) {
    detail::emit("INFO", fmt, std::forward<Args>(args)...);
}
template <typename... Args>
void warn(const std::string& fmt, Args&&... args) {
    detail::emit("WARN", fmt, std::forward<Args>(args)...);
}
template <typename... Args>
void error(const std::string& fmt, Args&&... args) {
    detail::emit("ERROR", fmt, std::forward<Args>(args)...);
}

}  // namespace scope::log
