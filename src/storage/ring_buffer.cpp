#include "storage/ring_buffer.hpp"

namespace scope {

RingBuffer::RingBuffer(std::size_t capacity)
    : capacity_(capacity == 0 ? 1 : capacity), buffer_(capacity_) {}

void RingBuffer::push(const TelemetryEvent& event) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (event.event_type == "attention_weights") {
        latest_attn_[event.attention.layer_name] = event.attention;
        return;
    }
    buffer_[write_ptr_] = event;
    write_ptr_ = (write_ptr_ + 1) % capacity_;
    if (size_ < capacity_) ++size_;
}

std::vector<TelemetryEvent> RingBuffer::get_all() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<TelemetryEvent> out;
    out.reserve(size_ + latest_attn_.size());
    // When full, the oldest element sits at write_ptr_; otherwise we filled
    // [0, size_) front to back.
    const std::size_t start = (size_ == capacity_) ? write_ptr_ : 0;
    for (std::size_t i = 0; i < size_; ++i) {
        out.push_back(buffer_[(start + i) % capacity_]);
    }
    for (const auto& [layer, attn] : latest_attn_) {
        TelemetryEvent ev;
        ev.event_type = "attention_weights";
        ev.attention = attn;
        out.push_back(ev);
    }
    return out;
}

std::optional<AttentionData> RingBuffer::latest_attention(
    const std::string& layer) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (auto it = latest_attn_.find(layer); it != latest_attn_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::size_t RingBuffer::size() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return size_;
}

void RingBuffer::clear() {
    std::lock_guard<std::mutex> lk(mutex_);
    write_ptr_ = 0;
    size_ = 0;
    latest_attn_.clear();
}

}  // namespace scope
