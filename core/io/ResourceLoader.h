#pragma once

#include "containers/String.h"
#include "containers/HashMap.h"

namespace MyEngine {

class Resource;

// 资源加载器
class ResourceLoader {
public:
    static ResourceLoader& get_instance();

    struct Format {
        String extension;
        Resource* (*loader)(const String&) = nullptr;
        void (*saver)(Resource*, const String&) = nullptr;
    };

    // 注册加载器
    void register_format(const String& extension,
                         Resource* (*loader)(const String&),
                         void (*saver)(Resource*, const String&));

    // 加载
    Resource* load(const String& path, bool cache = true);
    Resource* load_interactive(const String& path);

    // 保存
    bool save(const String& path, Resource* resource);

    // 缓存
    void cache_resource(Resource* resource);
    void remove_from_cache(const String& path);
    bool has_cached_resource(const String& path) const;
    Resource* get_cached_resource(const String& path) const;
    void clear_cache();

    // 路径
    static String get_resource_path(const String& path);
    static String get_resource_dir(const String& path);
    static String get_resource_basename(const String& path);
    static String get_resource_extension(const String& path);

private:
    ResourceLoader() = default;
    ~ResourceLoader() = default;

    HashMap<String, Format> _formats;
    HashMap<String, Resource*> _cache;
};

} // namespace MyEngine
