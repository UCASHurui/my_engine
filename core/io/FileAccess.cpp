#include "FileAccess.h"
#include <sys/stat.h>

namespace MyEngine {

FileAccess* FileAccess::open(const String& path, Mode mode) {
    // TODO: 平台特定实现
    (void)path;
    (void)mode;
    return nullptr;
}

void FileAccess::close(FileAccess* file) {
    delete file;
}

bool FileAccess::exists(const String& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

bool FileAccess::remove(const String& path) {
    return std::remove(path.c_str()) == 0;
}

bool FileAccess::rename(const String& from, const String& to) {
    return std::rename(from.c_str(), to.c_str()) == 0;
}

bool FileAccess::make_dir(const String& path) {
    return mkdir(path.c_str(), 0755) == 0;
}

bool FileAccess::dir_exists(const String& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

} // namespace MyEngine
