#pragma once

#include "containers/String.h"
#include <cstdio>
#include <cstdint>
#include <vector>

namespace MyEngine {

// 文件访问类
class FileAccess {
public:
    enum Mode {
        READ = 1,
        WRITE = 2,
        READ_WRITE = READ | WRITE,
        APPEND = 4
    };

    static FileAccess* open(const String& path, Mode mode = READ);
    static void close(FileAccess* file);

    virtual ~FileAccess() = default;

    // 读取
    virtual size_t read(void* buffer, size_t size) = 0;
    virtual String read_line() = 0;
    virtual String read_to_end() = 0;
    virtual std::vector<uint8_t> get_buffer() = 0;

    // 写入
    virtual size_t write(const void* buffer, size_t size) = 0;
    virtual void write_string(const String& str) = 0;
    virtual void flush() = 0;

    // 位置
    virtual size_t get_position() const = 0;
    virtual bool seek(size_t pos) = 0;
    virtual bool seek_end(int offset = 0) = 0;
    virtual void rewind() = 0;

    // 属性
    virtual size_t get_size() const = 0;
    virtual bool is_open() const = 0;
    virtual bool eof() const = 0;
    virtual bool error() const = 0;

    // 文件系统操作
    static bool exists(const String& path);
    static bool remove(const String& path);
    static bool rename(const String& from, const String& to);
    static bool make_dir(const String& path);
    static bool dir_exists(const String& path);
    static String get_modification_time(const String& path);

protected:
    FileAccess() = default;
};

} // namespace MyEngine
