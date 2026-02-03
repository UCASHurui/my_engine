#include "ResourceLoader.h"
#include "object/Object.h"

namespace MyEngine {

ResourceLoader& ResourceLoader::get_instance() {
    static ResourceLoader instance;
    return instance;
}

void ResourceLoader::register_format(const String& extension,
                                      Resource* (*loader)(const String&),
                                      void (*saver)(Resource*, const String&)) {
    Format fmt;
    fmt.extension = extension;
    fmt.loader = loader;
    fmt.saver = saver;
    _formats.insert(extension, fmt);
}

Resource* ResourceLoader::load(const String& path, bool cache) {
    String ext = get_resource_extension(path);

    auto it = _formats.find(ext);
    if (it != _formats.end() && (*it).value.loader) {
        Resource* res = (*it).value.loader(path);
        if (cache && res) {
            cache_resource(res);
        }
        return res;
    }
    return nullptr;
}

Resource* ResourceLoader::load_interactive(const String& path) {
    return load(path, false);
}

bool ResourceLoader::save(const String& path, Resource* resource) {
    String ext = get_resource_extension(path);

    auto it = _formats.find(ext);
    if (it != _formats.end() && (*it).value.saver) {
        (*it).value.saver(resource, path);
        return true;
    }
    return false;
}

void ResourceLoader::cache_resource(Resource* resource) {
    (void)resource;
}

void ResourceLoader::remove_from_cache(const String& path) {
    _cache.erase(path);
}

bool ResourceLoader::has_cached_resource(const String& path) const {
    return _cache.find(path) != _cache.end();
}

Resource* ResourceLoader::get_cached_resource(const String& path) const {
    auto it = _cache.find(path);
    if (it != _cache.end()) {
        return (*it).value;
    }
    return nullptr;
}

void ResourceLoader::clear_cache() {
    _cache.clear();
}

String ResourceLoader::get_resource_path(const String& path) {
    return path;
}

String ResourceLoader::get_resource_dir(const String& path) {
    size_t pos = path.find_last_of("/");
    if (pos != String::npos) {
        return path.substr(0, pos);
    }
    return "";
}

String ResourceLoader::get_resource_basename(const String& path) {
    size_t pos1 = path.find_last_of("/");
    size_t pos2 = path.find_last_of(".");
    if (pos2 != String::npos && pos2 > pos1) {
        return path.substr(pos1 + 1, pos2 - pos1 - 1);
    }
    return path;
}

String ResourceLoader::get_resource_extension(const String& path) {
    size_t pos = path.find_last_of(".");
    if (pos != String::npos) {
        return path.substr(pos);
    }
    return "";
}

} // namespace MyEngine
