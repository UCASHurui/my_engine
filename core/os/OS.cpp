#include "OS.h"
#include <thread>
#include <unistd.h>

namespace MyEngine {

bool OS::_initialized = false;

void OS::initialize() {
    if (_initialized) return;
    _initialized = true;
    // 平台特定初始化在子类中
}

void OS::shutdown() {
    if (!_initialized) return;
    _initialized = false;
}

uint64_t OS::get_ticks_usec() {
    using namespace std::chrono;
    return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

uint64_t OS::get_ticks_msec() {
    return get_ticks_usec() / 1000;
}

double OS::get_seconds() {
    return get_ticks_usec() / 1000000.0;
}

void OS::sleep(uint64_t msec) {
    std::this_thread::sleep_for(std::chrono::milliseconds(msec));
}

String OS::get_datetime_string() {
    time_t now = time(nullptr);
    tm* tm_now = localtime(&now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm_now);
    return String(buf);
}

String OS::get_date_string() {
    time_t now = time(nullptr);
    tm* tm_now = localtime(&now);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d", tm_now);
    return String(buf);
}

String OS::get_time_string() {
    time_t now = time(nullptr);
    tm* tm_now = localtime(&now);
    char buf[32];
    strftime(buf, sizeof(buf), "%H:%M:%S", tm_now);
    return String(buf);
}

String OS::get_user_data_dir() {
    return ".";
}

String OS::get_app_data_dir() {
    return ".";
}

String OS::get_current_dir() {
    char buf[256];
    getcwd(buf, sizeof(buf));
    return String(buf);
}

bool OS::set_current_dir(const String& path) {
    return chdir(path.c_str()) == 0;
}

String OS::get_clipboard() {
    return "";
}

void OS::set_clipboard(const String&) {}

String OS::get_os_name() {
#ifdef _WIN32
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "Unknown";
#endif
}

String OS::get_cpu_model() {
    return "Unknown";
}

size_t OS::get_memory_total() {
    return 0;
}

size_t OS::get_memory_available() {
    return 0;
}

void OS::print(const String& message) {
    printf("%s\n", message.c_str());
}

void OS::print_error(const String& message) {
    fprintf(stderr, "ERROR: %s\n", message.c_str());
}

void OS::print_warning(const String& message) {
    fprintf(stderr, "WARNING: %s\n", message.c_str());
}

void OS::request_quit() {
    // 退出请求
}

void OS::crash(const String& message) {
    fprintf(stderr, "CRASH: %s\n", message.c_str());
    abort();
}

// Time
Time::Timer::Timer() {
    reset();
}

void Time::Timer::start() {
    _start = std::chrono::high_resolution_clock::now();
    _running = true;
}

void Time::Timer::stop() {
    _end = std::chrono::high_resolution_clock::now();
    _running = false;
}

void Time::Timer::reset() {
    _start = std::chrono::high_resolution_clock::now();
    _running = false;
}

uint64_t Time::Timer::get_ticks_usec() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - _start).count();
}

uint64_t Time::Timer::get_ticks_msec() const {
    return get_ticks_usec() / 1000;
}

double Time::Timer::get_seconds() const {
    return get_ticks_usec() / 1000000.0;
}

} // namespace MyEngine
