#pragma once

#include "core/containers/String.h"
#include <chrono>
#include <ctime>

namespace MyEngine {

// 操作系统抽象层
class OS {
public:
    static void initialize();
    static void shutdown();

    // 时间
    static uint64_t get_ticks_usec();           // 微秒
    static uint64_t get_ticks_msec();           // 毫秒
    static double get_seconds();                // 秒
    static void sleep(uint64_t msec);           // 睡眠

    // 日期时间
    static String get_datetime_string();
    static String get_date_string();
    static String get_time_string();

    // 环境
    static String get_environment(const String& env);
    static bool set_environment(const String& env, const String& value);

    // 文件系统
    static String get_user_data_dir();
    static String get_app_data_dir();
    static String get_current_dir();
    static bool set_current_dir(const String& path);

    // 剪贴板
    static String get_clipboard();
    static void set_clipboard(const String& text);

    // 系统信息
    static String get_os_name();
    static String get_cpu_model();
    static size_t get_memory_total();
    static size_t get_memory_available();

    // 调试
    static void print(const String& message);
    static void print_error(const String& message);
    static void print_warning(const String& message);

    // 退出
    static void request_quit();
    static void crash(const String& message);

private:
    static bool _initialized;
};

// 时间工具
class Time {
public:
    struct DateTime {
        int year, month, day;
        int hour, minute, second;
        int day_of_week;
        int dst;
    };

    static DateTime get_datetime();
    static uint64_t get_unix_time();
    static double get_unix_time_as_double();

    // 计时器
    class Timer {
    public:
        Timer();
        void start();
        void stop();
        void reset();
        bool is_running() const { return _running; }

        uint64_t get_ticks_usec() const;
        uint64_t get_ticks_msec() const;
        double get_seconds() const;

    private:
        std::chrono::high_resolution_clock::time_point _start;
        std::chrono::high_resolution_clock::time_point _end;
        bool _running = false;
    };
};

// 性能监控
class Performance {
public:
    static double get_fps();
    static double get_frame_time();
    static uint64_t get_frame_count();
    static double get_average_frame_time();
};

} // namespace MyEngine
