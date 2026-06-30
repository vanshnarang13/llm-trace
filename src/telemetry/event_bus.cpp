#include "telemetry/event_bus.hpp"

namespace scope {

void EventBus::subscribe(const std::string& event_type, Callback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    subscribers_[event_type].push_back(std::move(cb));
}

void EventBus::publish(const TelemetryEvent& event) {
    // Copy the relevant callbacks under the lock, then invoke them unlocked so a
    // subscriber is free to publish or subscribe without deadlocking.
    std::vector<Callback> targets;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (auto it = subscribers_.find(event.event_type); it != subscribers_.end()) {
            targets.insert(targets.end(), it->second.begin(), it->second.end());
        }
        if (auto it = subscribers_.find("*"); it != subscribers_.end()) {
            targets.insert(targets.end(), it->second.begin(), it->second.end());
        }
    }
    for (auto& cb : targets) cb(event);
}

}  // namespace scope
