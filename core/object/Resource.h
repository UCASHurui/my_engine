#pragma once

#include "RefCounted.h"
#include <string>

namespace MyEngine {

// 资源类型
enum class ResourceType {
    NONE,
    TEXTURE,
    MESH,
    SHADER,
    MATERIAL,
    SCENE,
    FONT,
    AUDIO,
    ANIMATION,
    PARTICLE,
    MAX_TYPES
};

// 资源基类
class Resource : public RefCounted {
public:
    Resource();
    ~Resource() override;

    const char* get_class_name() const override { return "Resource"; }

    // 路径
    void set_path(const std::string& path) { _path = path; }
    std::string get_path() const { return _path; }

    // 名称（从路径派生）
    void set_name(const std::string& name) { _name = name; }
    std::string get_name() const { return _name.empty() ? _path : _name; }

    // 类型
    void set_type(ResourceType type) { _type = type; }
    ResourceType get_type() const { return _type; }

    // 是否已加载
    bool is_loaded() const { return _loaded; }
    void set_loaded(bool loaded) { _loaded = loaded; }

    // 本地到资源
    void set_local_to_scene(bool enabled) { _local_to_scene = enabled; }
    bool is_local_to_scene() const { return _local_to_scene; }

    // 虚拟化
    void set_virtual(bool virtualized) { _virtual = virtualized; }
    bool is_virtual() const { return _virtual; }

    // 导入相关
    void set_import_path(const std::string& path) { _import_path = path; }
    std::string get_import_path() const { return _import_path; }

    void set_import_md5(const std::string& md5) { _import_md5 = md5; }
    std::string get_import_md5() const { return _import_md5; }

    // 资源强度（用于流式加载）
    void set_resource_priority(int priority) { _resource_priority = priority; }
    int get_resource_priority() const { return _resource_priority; }

    // 引用计数
    int get_reference_count() const { return RefCounted::get_reference_count(); }

protected:
    // 加载/卸载
    virtual bool _load(const std::string& path);
    virtual void _unload();

    void _set_loaded(bool loaded) { _loaded = loaded; }

private:
    std::string _path;
    std::string _name;
    ResourceType _type = ResourceType::NONE;
    bool _loaded = false;
    bool _local_to_scene = false;
    bool _virtual = false;

    std::string _import_path;
    std::string _import_md5;
    int _resource_priority = 0;
};

} // namespace MyEngine
