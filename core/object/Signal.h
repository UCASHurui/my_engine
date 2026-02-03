#pragma once

#include "containers/Vector.h"
#include "containers/HashMap.h"
#include "variant/Variant.h"
#include <functional>

namespace MyEngine {

// 简化版信号/事件系统
class Signal {
public:
    Signal();
    ~Signal();

    uint32_t connect(std::function<void(const Variant&, const Variant&)> callback);
    void disconnect(uint32_t connection_id);
    void disconnect_all();
    bool is_connected(uint32_t connection_id) const;

    void emit(const Variant& arg1, const Variant& arg2) const;

private:
    HashMap<uint32_t, std::function<void(const Variant&, const Variant&)>> _connections;
    mutable uint32_t _signal_id = 0;
};

} // namespace MyEngine
