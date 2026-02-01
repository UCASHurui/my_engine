#include "Signal.h"

namespace MyEngine {

Signal::Signal() : _signal_id(0) {}

Signal::~Signal() {
    disconnect_all();
}

uint32_t Signal::connect(std::function<void(const Variant&, const Variant&)> callback) {
    uint32_t id = ++_signal_id;
    _connections.insert(id, callback);
    return id;
}

void Signal::disconnect(uint32_t connection_id) {
    _connections.erase(connection_id);
}

void Signal::disconnect_all() {
    _connections.clear();
}

bool Signal::is_connected(uint32_t connection_id) const {
    return _connections.find(connection_id) != _connections.end();
}

void Signal::emit(const Variant& arg1, const Variant& arg2) const {
    for (const auto& conn : _connections) {
        if (conn.value) {
            conn.value(arg1, arg2);
        }
    }
}

} // namespace MyEngine
