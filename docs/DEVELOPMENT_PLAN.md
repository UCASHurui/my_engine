# 开发计划

本文档描述游戏引擎的完整开发计划，包括核心引擎开发和Agent集成。

## 目录

- [项目概览](#项目概览)
- [整体路线图](#整体路线图)
- [Phase 1: 核心基础 (已完成)](#phase-1-核心基础-已完成)
- [Phase 2: 渲染基础](#phase-2-渲染基础)
- [Phase 3: 场景系统](#phase-3-场景系统)
- [Phase 4: 风格化渲染](#phase-4-风格化渲染)
- [Phase 5: 脚本与工具](#phase-5-脚本与工具)
- [Phase 6: Agent集成 (新增)](#phase-6-agent集成-新增)
- [Phase 7: 编辑器与完善](#phase-7-编辑器与完善)
- [里程碑检查表](#里程碑检查表)

---

## 项目概览

### 目标

构建一个轻量化、可扩展的游戏引擎，支持：
- 2D/3D 游戏开发
- 风格化渲染
- Lua 脚本
- **AI Agent 协作开发** (新增)

### 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | C++17 |
| 脚本 | Lua 5.4+ |
| 渲染 | OpenGL / Vulkan |
| 构建 | CMake |
| Agent | OpenAI / Anthropic / DeepSeek API |

---

## 整体路线图

```
时间轴 (周)
─────────────────────────────────────────────────────────────────────────────
Phase 1     Phase 2     Phase 3     Phase 4     Phase 5     Phase 6     Phase 7
核心基础     渲染基础     场景系统     风格化      脚本工具    Agent集成    编辑器
(2-3周)     (3-4周)     (2-3周)     (3-4周)     (2-3周)     (4-5周)     (2-3周)
─────────────────────────────────────────────────────────────────────────────

[现有代码]───►[────────现有计划────────]───►[────新增────]───►[后续完善]
```

---

## Phase 1: 核心基础 (已完成)

### 完成项

- [x] Core 层 (Object, Variant, ClassDB)
- [x] 基础数学库 (Vector, Matrix, Transform)
- [x] 内存管理 (RefCounted, Object)
- [x] 基础容器 (Vector, HashMap, String)

### 文件清单

```
core/
├── object/
│   ├── Object.h/cpp          # 对象基类
│   ├── ClassDB.h/cpp         # 类注册与反射
│   ├── RefCounted.h/cpp      # 引用计数
│   └── Signal.h/cpp          # 信号机制
├── variant/
│   ├── Variant.h/cpp         # 变体类型
├── math/
│   ├── Vector2.h/cpp
│   ├── Vector3.h/cpp
│   ├── Vector4.h/cpp
│   └── Matrix4.h/cpp
├── containers/
│   ├── Vector.h
│   ├── HashMap.h
│   └── String.h
└── io/
    ├── FileAccess.h/cpp
    └── ResourceLoader.h/cpp
```

---

## Phase 2: 渲染基础

### 目标

实现渲染设备抽象和基础渲染能力。

### 任务清单

| 优先级 | 任务 | 描述 | 预计工作量 |
|--------|------|------|-----------|
| P0 | RenderDevice 抽象 | 定义渲染设备接口 | 3天 |
| P0 | OpenGL 后端 | OpenGL 渲染实现 | 1周 |
| P0 | Shader 系统 | Shader 编译和加载 | 3天 |
| P1 | Mesh 系统 | 网格资源管理 | 2天 |
| P1 | Texture 系统 | 纹理资源管理 | 2天 |
| P2 | Vulkan 后端 | Vulkan 渲染实现 | 1周 |

### 交付物

```
renderer/
├── RenderDevice.h/cpp       # 渲染设备抽象
├── OpenGLRenderDevice.cpp   # OpenGL 实现
├── VulkanRenderDevice.cpp   # Vulkan 实现
├── Shader.h/cpp             # Shader 资源
├── Mesh.h/cpp               # 网格资源
└── Texture.h/cpp            # 纹理资源
```

### 与现有代码集成

```cpp
// 在 Engine 中初始化渲染设备
class Engine {
    bool initialize(const EngineConfig& config) override {
        // ... 其他初始化

        _render_device = RenderDevice::create(config.render_backend);
        if (!_render_device->initialize()) {
            return false;
        }

        return true;
    }

private:
    Ref<RenderDevice> _render_device;
};
```

---

## Phase 3: 场景系统

### 目标

实现 Node/SceneTree 架构和基础节点。

### 任务清单

| 优先级 | 任务 | 描述 | 预计工作量 |
|--------|------|------|-----------|
| P0 | Node 基类 | 场景节点基类实现 | 2天 |
| P0 | SceneTree | 场景树管理器 | 2天 |
| P1 | Node2D | 2D 节点基类 | 2天 |
| P1 | Sprite2D | 2D 精灵节点 | 1天 |
| P1 | Camera2D | 2D 相机 | 1天 |
| P1 | Node3D | 3D 节点基类 | 2天 |
| P1 | MeshInstance3D | 3D 网格实例 | 2天 |
| P2 | Camera3D | 3D 相机 | 1天 |
| P2 | Light3D | 3D 光源 | 1天 |

### 交付物

```
scene/
├── Node.h/cpp               # 场景节点基类
├── SceneTree.h/cpp          # 场景树管理器
├── 2d/
│   ├── Node2D.h/cpp
│   ├── Sprite2D.h/cpp
│   └── Camera2D.h/cpp
└── 3d/
    ├── Node3D.h/cpp
    └── MeshInstance3D.h/cpp
```

### 序列化支持 (为 Agent 集成做准备)

```cpp
// 为 Agent 集成添加序列化方法
class Node : public Object {
public:
    // 序列化节点为 JSON
    virtual Dictionary serialize() const {
        Dictionary data;
        data["name"] = _name;
        data["type"] = get_class_name();
        data["children"] = _serialize_children();
        return data;
    }

    // 从 JSON 反序列化
    virtual void deserialize(const Dictionary& data) {
        _name = data.get("name", "");
        _deserialize_children(data.get("children", Array()));
    }

private:
    String _name;
    Vector<Ref<Node>> _children;
};
```

---

## Phase 4: 风格化渲染

### 目标

实现引擎特色：风格化渲染效果。

### 任务清单

| 优先级 | 任务 | 描述 | 预计工作量 |
|--------|------|------|-----------|
| P0 | PostProcess 系统 | 后处理框架 | 3天 |
| P1 | Cel Shading | 卡通渲染 | 3天 |
| P1 | Outline Pass | 描边效果 | 2天 |
| P2 | Rim Light | 边缘光 | 2天 |
| P2 | Gradient Ramp | 渐变映射 | 1天 |
| P3 | Pixel Art Filter | 像素化 | 2天 |
| P3 | Hatching | 影线风格 | 3天 |
| P3 | Halftone | 半调效果 | 2天 |

### 交付物

```
renderer/stylized/
├── PostProcess.h/cpp        # 后处理基类
├── CelShading.h/cpp         # 卡通渲染
├── OutlinePass.h/cpp        # 描边
├── RimLight.h/cpp           # 边缘光
├── GradientRamp.h/cpp       # 渐变映射
├── PixelArtFilter.h/cpp     # 像素化
└── HatchingRender.h/cpp     # 影线风格
```

### Shader 示例

```glsl
// cel_shading.glsl
shader_type spatial;

uniform vec3 base_color : source_color = vec3(1.0);
uniform sampler2D ramp_texture;
uniform int shade_levels = 3;

void fragment() {
    float NdotL = dot(NORMAL, LIGHT);
    float shade = floor(NdotL * float(shade_levels)) / float(shadeLevels);
    ALBEDO = base_color * (AMBIENT + DIFFUSE * shade);
}
```

---

## Phase 5: 脚本与工具

### 目标

完善 Lua 脚本支持和基础工具链。

### 任务清单

| 优先级 | 任务 | 描述 | 预计工作量 |
|--------|------|------|-----------|
| P0 | Lua 绑定完善 | 扩展 Lua API | 3天 |
| P1 | 场景序列化 | JSON 场景文件格式 | 2天 |
| P1 | 资源打包 | 资源打包工具 | 2天 |
| P2 | 场景编辑器UI | 简单的场景编辑界面 | 1周 |

### 交付物

```
scripts/
├── LuaBridge.h/cpp          # Lua 绑定
├── LuaVM.h/cpp              # Lua 虚拟机
└── api/
    ├── NodeAPI.cpp          # 节点 API
    ├── SceneAPI.cpp         # 场景 API
    └── RenderAPI.cpp        # 渲染 API

tools/
├── scene_exporter           # 场景导出工具
└── resource_packer          # 资源打包工具
```

---

## Phase 6: Agent集成 (新增)

### 目标

实现 AI Agent 与引擎的深度集成，支持人机协作开发。

### 架构总览

```
                    ┌─────────────────────┐
                    │   WorkFlowManager   │
                    │   (工作流管理器)     │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Human Only     │  │ Human + Agent   │  │  Sandbox        │
│  Mode           │  │  Mode           │  │  Manager        │
│  (纯人工模式)    │  │  (人机协作模式)  │  │  (沙箱管理)     │
└─────────────────┘  └────────┬────────┘  └─────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   SessionManager    │
                   │   (会话管理)         │
                   └──────────┬──────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  LLMClient      │  │  ToolRegistry   │  │  ContextManager │
│  (远程LLM通信)   │  │  (工具注册)      │  │  (上下文管理)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 任务清单

#### Step 6.1: 基础设施 (1周)

| 任务 | 描述 | 依赖 |
|------|------|------|
| 创建 agent 目录结构 | 建立模块组织 | 无 |
| 实现 LLMClient | 远程API通信 | 无 |
| 实现基础配置系统 | API Key管理等 | 无 |

```cpp
// agent/remote/LLMClient.h - 核心实现
class LLMClient : public RefCounted {
public:
    static Ref<LLMClient> create(const LLMProviderConfig& config);
    Result<LLMResponse> chat(const LLMRequest& request);

private:
    LLMProviderConfig _config;
    CURL* _curl = nullptr;
};
```

#### Step 6.2: 工具系统 (1周)

| 任务 | 描述 | 依赖 |
|------|------|------|
| ToolRegistry | 工具注册中心 | Step 6.1 |
| File Tools | read_file, write_file | Step 6.1 |
| Scene Tools | get_scene_state, create_node | Phase 3 |
| Render Tools | get_render_stats | Phase 2 |

```cpp
// agent/core/ToolRegistry.h
class ToolRegistry : public RefCounted {
public:
    void register_tool(Ref<Tool> tool);
    ToolResult execute_tool(const String& name, const Vector<Variant>& args);
    String generate_tools_description_json();
};
```

#### Step 6.3: 沙箱系统 (1周)

| 任务 | 描述 | 依赖 |
|------|------|------|
| SandboxManager | 沙箱管理器 | Step 6.1 |
| 隔离环境创建 | 独立目录/进程 | Step 6.2 |
| 文件同步 | 沙箱与主环境同步 | Step 6.2 |

```cpp
// agent/sandbox/SandboxManager.h
class SandboxManager : public RefCounted {
    String create_sandbox(const SandboxConfig& config);
    SandboxResult run_command(const String& instance_id, const String& cmd);
    void sync_to_sandbox(const String& instance_id, const String& path);
};
```

#### Step 6.4: 会话管理 (1周)

| 任务 | 描述 | 依赖 |
|------|------|------|
| SessionManager | 会话管理器 | Step 6.2 |
| Agent Role 定义 | 各类型Agent配置 | Step 6.2 |
| 上下文管理 | 记忆/历史管理 | Step 6.2 |

```cpp
// agent/core/SessionManager.h
struct AgentRole {
    String name;
    String system_prompt;
    Vector<String> allowed_tools;
    Vector<String> restricted_paths;
    bool require_approval;
    int max_iterations;
};

class AgentSession : public RefCounted {
    String execute(const String& instruction);
    void request_approval(const String& changes);
    void deploy_to_sandbox();
};
```

#### Step 6.5: 工作流与审批 (1周)

| 任务 | 描述 | 依赖 |
|------|------|------|
| WorkFlowManager | 工作流管理器 | Step 6.4 |
| 审批队列 | 审批请求管理 | Step 6.4 |
| 双模式切换 | Human/Human+Agent | Step 6.4 |

```cpp
// agent/workflow/WorkFlowManager.h
enum class WorkMode {
    HUMAN_ONLY,
    HUMAN_AGENT
};

class WorkFlowManager : public RefCounted {
    void set_work_mode(WorkMode mode);
    String submit_task(AgentType type, const String& instruction);
    void approve(const String& request_id);
    void reject(const String& request_id, const String& reason);

    Signal<void(const ApprovalRequest&)> on_approval_required;
};
```

### Agent 类型详细定义

| Agent | 系统提示词 | 工具集 | 审批要求 |
|-------|-----------|--------|---------|
| ENGINE_DEVELOPER | 引擎开发者角色 | file, compile, test | 总是需要 |
| SCENE_BUILDER | 场景构建者角色 | scene, file | 高风险需要 |
| ART_GENERATOR | 艺术创作者角色 | render, file | 中风险需要 |
| NARRATIVE_WRITER | 剧情编写者角色 | script, file | 低风险需要 |
| TESTER | 测试工程师角色 | test, file | 可选审批 |
| RENDER_SPECIALIST | 渲染专家角色 | render, test | 总是需要 |

### 工具 Schema 定义示例

```json
// api_spec/tool_schemas.json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read the contents of a file",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "Relative path from engine root"
          }
        },
        "required": ["path"]
      }
    },
    {
      "name": "get_scene_state",
      "description": "Get current scene tree as JSON",
      "parameters": {
        "type": "object",
        "properties": {}
      }
    },
    {
      "name": "run_test",
      "description": "Run test in sandbox",
      "parameters": {
        "type": "object",
        "properties": {
          "test_name": {
            "type": "string",
            "description": "Name of the test to run"
          }
        },
        "required": ["test_name"]
      }
    }
  ]
}
```

### 系统提示词模板

```text
# engine_dev_agent.txt

你是一个专业的游戏引擎开发者，专注于帮助用户迭代和优化游戏引擎代码。

## 你的职责
1. 理解用户的开发需求和技术问题
2. 编写高质量、符合引擎架构的C++代码
3. 遵循引擎现有的代码风格和规范
4. 确保代码通过测试后提交

## 工作流程
1. 先理解需求，必要时提出澄清问题
2. 分析现有代码结构和相关文件
3. 编写修改方案和代码
4. 在沙箱环境中编译测试
5. 生成变更摘要提交审批
6. 根据反馈进行调整

## 代码规范
- 使用引擎已有的类型系统 (Ref<T>, String, Vector 等)
- 遵循引擎的命名约定
- 添加适当的注释
- 保持代码简洁

## 当前引擎信息
- 语言: C++17
- 架构: Node/Component系统
- 脚本: Lua
- 渲染: OpenGL/Vulkan
```

### 交付物

```
agent/
├── core/
│   ├── SessionManager.h/cpp
│   ├── ToolRegistry.h/cpp
│   └── ContextManager.h/cpp
├── remote/
│   └── LLMClient.h/cpp
├── sandbox/
│   └── SandboxManager.h/cpp
├── workflow/
│   └── WorkFlowManager.h/cpp
└── tools/
    ├── FileTools.h/cpp
    ├── SceneTools.h/cpp
    ├── RenderTools.h/cpp
    └── TestTools.h/cpp

api_spec/
├── tool_schemas.json
└── system_prompts/
    ├── engine_dev_agent.txt
    ├── scene_builder_agent.txt
    ├── art_generator_agent.txt
    ├── narrative_agent.txt
    └── tester_agent.txt

docs/
└── AGENT_INTEGRATION.md
```

---

## Phase 7: 编辑器与完善

### 目标

完善编辑器功能，增强用户体验。

### 任务清单

| 优先级 | 任务 | 描述 | 预计工作量 |
|--------|------|------|-----------|
| P1 | Agent Panel | Agent 控制面板 | 3天 |
| P1 | 审批界面 | 审批请求 UI | 2天 |
| P1 | 会话历史 | 会话查看器 | 1天 |
| P2 | 实时监控 | 性能监控面板 | 2天 |
| P2 | 资源浏览器 | 资源管理 UI | 2天 |

### 交付物

```
editor/
├── AgentPanel.h/cpp        # Agent 控制面板
├── ApprovalDialog.h/cpp    # 审批对话框
└── SessionView.h/cpp       # 会话查看器
```

---

## 里程碑检查表

### 核心引擎里程碑

| 里程碑 | 检查项 | 状态 |
|--------|--------|------|
| M1: 核心基础完成 | Object/Variant 系统可用 | ✅ |
| M2: 渲染基础完成 | 可渲染基本图形 | ⬜ |
| M3: 场景系统完成 | 可创建和运行场景 | ⬜ |
| M4: 风格化完成 | Cel Shading 可用 | ⬜ |
| M5: 脚本完成 | Lua 脚本可运行 | ⬜ |

### Agent 集成里程碑

| 里程碑 | 检查项 | 状态 |
|--------|--------|------|
| A1: LLM 通信 | 可调用远程 API | ⬜ |
| A2: 工具系统 | Agent 可操作引擎 | ⬜ |
| A3: 沙箱可用 | 隔离测试环境可用 | ⬜ |
| A4: 审批流程 | 人机协作流程完整 | ⬜ |
| A5: 双模式 | 支持两种工作模式 | ⬜ |

### 功能完整性检查

| 功能 | 优先级 | 状态 |
|------|--------|------|
| 2D 渲染 | P0 | ⬜ |
| 3D 渲染 | P1 | ⬜ |
| 风格化效果 | P1 | ⬜ |
| Lua 脚本 | P0 | ⬜ |
| Agent 引擎开发 | P1 | ⬜ |
| Agent 场景构建 | P1 | ⬜ |
| Agent 艺术创作 | P2 | ⬜ |
| Agent 测试 | P2 | ⬜ |
| 审批系统 | P0 | ⬜ |
| 沙箱测试 | P0 | ⬜ |

---

## 依赖关系图

```
Phase 1 (核心)
    │
    ▼
Phase 2 (渲染) ─────┐
    │               │
    ▼               │
Phase 3 (场景) ─────┼──► Phase 6 (Agent)
    │               │
    ▼               │
Phase 4 (风格化) ───┤
    │               │
    ▼               │
Phase 5 (脚本) ─────┤
    │               │
    └───────────────┘
           │
           ▼
Phase 7 (编辑器)
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| LLM API 不稳定 | Agent 响应延迟 | 添加本地降级方案 |
| 沙箱资源占用 | 系统资源不足 | 限制并发沙箱数量 |
| 代码质量 | Agent 生成的代码有bug | 强制沙箱测试 + 审批 |
| 上下文长度 | 长对话丢失信息 | 实现上下文压缩 |

---

## 相关文档

- [README](../README.md) - 项目主文档
- [AG.md](AGENT_INTEGRATIONENT_INTEGRATION.md) - Agent 集成详细设计
- [API规范](../api_spec/) - 工具定义和提示词
