# Agent 集成架构设计

本文档描述如何将 AI Agent 集成到游戏引擎中，实现人与 Agent 共同开发。

## 目录

- [设计目标](#设计目标)
- [整体架构](#整体架构)
- [核心组件](#核心组件)
- [工作模式](#工作模式)
- [Agent 类型](#agent-类型)
- [工具系统](#工具系统)
- [沙箱系统](#沙箱系统)
- [审批流程](#审批流程)
- [上下文管理](#上下文管理)

---

## 设计目标

| 目标 | 描述 |
|------|------|
| **多Agent协作** | 支持引擎迭代、场景建模、艺术创作、剧情编写、测试、渲染等多种Agent |
| **双模式支持** | 纯人工模式 + 人机协作模式无缝切换 |
| **安全可控** | 沙箱测试 + 人工审批，确保代码安全 |
| **远程LLM** | 通过远程API（OpenAI/Anthropic/DeepSeek）调用大模型 |
| **引擎感知** | Agent能理解引擎API，进行有效开发 |

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              交互层 (Interaction Layer)                      │
│  ┌─────────────────────┐                      ┌─────────────────────────────┐│
│  │   Web IDE / CLI     │                      │   Desktop Dashboard         ││
│  │   (开发界面)         │                      │   (审批/监控面板)           ││
│  └──────────┬──────────┘                      └─────────────┬───────────────┘│
│             │                                           │                    │
│  ┌──────────▼──────────┐                      ┌──────────▼───────────────┐  │
│  │    Human Mode       │                      │   Human+Agent Mode        │  │
│  │   (纯人工模式)       │                      │   (人机协作模式)          │  │
│  └─────────────────────┘                      └───────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                        WorkFlow Manager (工作流管理器)                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │   Task Queue  │  Approval Queue  │  Sandbox Manager  │  Result Handler  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────────┤
│                          Agent Runtime (Agent运行时)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Remote LLM     │  │  Tool Executor  │  │   Context Manager           │  │
│  │  Client         │  │  (工具执行器)    │  │   (上下文/记忆管理)         │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │                    │                           │                  │
│  ┌────────▼────────────────────▼───────────────────────────▼──────────────┐  │
│  │                         Session Manager (会话管理)                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐   │  │
│  │  │ Role Define │  │ History     │  │ State       │  │ Prompt       │   │  │
│  │  │ (角色定义)   │  │ Manager     │  │ Machine     │  │ Generator    │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Tool Registry (工具注册中心)                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  File Tools  │  Scene Tools  │  Render Tools  │  Test Tools  │  ...    ││
│  │  • read_file │  • get_scene  │  • get_stats   │  • run_test  │         ││
│  │  • write_file│  • create_node│  • set_effect  │  • assert    │         ││
│  │  • list_dir  │  • modify     │  • compile     │  • benchmark │         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────────┤
│                         Sandbox Manager (沙箱管理器)                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  ││
│  │  │  Sandbox Env 1  │  │  Sandbox Env 2  │  │  Sandbox Env N          │  ││
│  │  │  (隔离环境)      │  │  (隔离环境)      │  │  (隔离环境)             │  ││
│  │  │  • 独立目录      │  │  • 独立目录      │  │  • 独立目录             │  ││
│  │  │  • 独立进程      │  │  • 独立进程      │  │  • 独立进程             │  ││
│  │  │  • 独立端口      │  │  • 独立端口      │  │  • 独立端口             │  ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────────┤
│                           Engine Core (引擎核心)                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  core/  │  scene/  │  renderer/  │  scripts/  │  editor/  │  servers/  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. Remote LLM Client (远程LLM客户端)

负责与远程大模型API通信。

```cpp
// agent/remote/LLMClient.h
class LLMClient : public RefCounted {
public:
    static Ref<LLMClient> create(const LLMProviderConfig& config);
    Result<LLMResponse> chat(const LLMRequest& request);
    void chat_stream(const LLMRequest& request,
                     std::function<void(const LLMResponse&)> on_chunk,
                     std::function<void(const Error&)> on_error);
};

struct LLMProviderConfig {
    enum class ProviderType { OPENAI, ANTHROPIC, DEEPSEEK, CUSTOM };
    ProviderType type;
    String api_url;
    String api_key;
    String default_model;
    float timeout_seconds = 60.0f;
};
```

### 2. Session Manager (会话管理器)

管理Agent会话，包括角色定义、历史记录、状态机等。

```cpp
// agent/core/SessionManager.h
struct AgentRole {
    String name;
    String system_prompt;           // 角色系统提示词
    Vector<String> allowed_tools;   // 允许使用的工具
    Vector<String> restricted_paths; // 禁止访问的路径
    bool require_approval = false;  // 是否需要审批
    int max_iterations = 10;        // 最大迭代次数
};

class AgentSession : public RefCounted {
    String session_id;
    AgentRole role;
    Ref<LLMClient> llm;

    String execute(const String& user_instruction);
    void request_approval(const String& changes_summary);
    void deploy_to_sandbox();
    SandboxResult run_in_sandbox(const String& command);
};

class SessionManager : public RefCounted {
    Ref<AgentSession> create_session(AgentType type);
    Ref<AgentSession> get_session(const String& session_id);
};
```

### 3. Sandbox Manager (沙箱管理器)

提供隔离的测试环境。

```cpp
// agent/sandbox/SandboxManager.h
struct SandboxConfig {
    String work_dir;           // 工作目录
    int memory_limit_mb;       // 内存限制
    int cpu_limit_percent;     // CPU限制
    int timeout_seconds;       // 超时时间
    Vector<int> allowed_ports; // 允许的端口
    bool enable_network;       // 是否允许网络
};

class SandboxManager : public RefCounted {
    String create_sandbox(const SandboxConfig& config);
    void destroy_sandbox(const String& instance_id);
    SandboxResult run_command(const String& instance_id, const String& cmd);
    void sync_to_sandbox(const String& instance_id, const String& local_path, const String& remote_path);
    String get_access_url(const String& instance_id);
};
```

### 4. WorkFlow Manager (工作流管理器)

协调任务分发、审批、结果处理。

```cpp
// agent/workflow/WorkFlowManager.h
enum class WorkMode {
    HUMAN_ONLY,       // 纯人工模式
    HUMAN_AGENT       // 人机协作模式
};

class WorkFlowManager : public RefCounted {
    void set_work_mode(WorkMode mode);
    String submit_task(AgentType agent_type, const String& instruction);
    String request_approval(const String& session_id, const String& task_id);
    void approve(const String& request_id, const String& approver = "human");
    void reject(const String& request_id, const String& reason);

    Signal<void(const String&)> on_task_complete;
    Signal<void(const ApprovalRequest&)> on_approval_required;
};
```

### 5. Tool Registry (工具注册表)

定义Agent可以使用的工具。

```cpp
// agent/core/ToolRegistry.h
struct ToolDef {
    String name;
    String description;
    Vector<ToolParam> params;
    String return_type;
    String category;
    bool require_sandbox = false;  // 是否需要在沙箱中执行
};

class ToolRegistry : public RefCounted {
    void register_tool(Ref<Tool> tool);
    ToolResult execute_tool(const String& tool_name, const Vector<Variant>& args,
                            bool use_sandbox = false);
    String generate_tools_description_json();
};
```

---

## 工作模式

### 模式1: Human Only (纯人工模式)

```
用户 ──直接操作──> 引擎/编辑器
```

- Agent功能完全禁用
- 用户直接操作引擎和编辑器
- 适合需要精细控制的场景

### 模式2: Human + Agent (人机协作模式)

```
用户指令
    │
    ▼
┌─────────────────────────────────────────┐
│           Agent Session                  │
│  1. 解析指令                             │
│  2. 执行工具调用                         │
│  3. 沙箱测试                             │
│  4. 生成变更摘要                         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           Approval Queue                │
│  • 等待用户审批                          │
│  • 显示变更摘要                          │
│  • 显示沙箱测试结果                      │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┐
    ▼                         ▼
┌──────────┐           ┌──────────┐
│  Approve │           │  Reject  │
│  (应用)  │           │  (拒绝)  │
└──────────┘           └──────────┘
```

---

## Agent 类型

| Agent类型 | 职责 | 典型工具 |
|----------|------|---------|
| `ENGINE_DEVELOPER` | 引擎迭代开发 | read_file, write_file, compile, run_test |
| `SCENE_BUILDER` | 场景建模 | get_scene_state, create_node, set_property |
| `ART_GENERATOR` | 艺术创作 | generate_texture, apply_material, render |
| `NARRATIVE_WRITER` | 剧情编写 | edit_script, create_dialogue, write_story |
| `TESTER` | 测试 | run_test, assert, benchmark, check_coverage |
| `RENDER_SPECIALIST` | 渲染优化 | get_render_stats, set_effect, profile |

---

## 工具系统

### 核心工具列表

#### 文件工具 (File Tools)
| 工具名 | 描述 | 参数 |
|-------|------|------|
| `read_file` | 读取文件 | `path: string` |
| `write_file` | 写入文件 | `path: string, content: string` |
| `list_dir` | 列出目录 | `path: string` |
| `search_files` | 搜索文件 | `pattern: string` |

#### 场景工具 (Scene Tools)
| 工具名 | 描述 | 参数 |
|-------|------|------|
| `get_scene_state` | 获取场景状态 | (无) |
| `create_node` | 创建节点 | `type: string, parent: string, name: string` |
| `set_property` | 设置属性 | `node_path: string, property: string, value: variant` |
| `get_property` | 获取属性 | `node_path: string, property: string` |

#### 渲染工具 (Render Tools)
| 工具名 | 描述 | 参数 |
|-------|------|------|
| `get_render_stats` | 获取渲染统计 | (无) |
| `set_effect` | 设置特效 | `effect_name: string, params: object` |
| `capture_frame` | 捕获帧画面 | `path: string` |

#### 测试工具 (Test Tools)
| 工具名 | 描述 | 参数 |
|-------|------|------|
| `run_test` | 运行测试 | `test_name: string, timeout: int` |
| `run_test_suite` | 运行测试套件 | `suite_name: string` |
| `assert` | 断言检查 | `condition: bool, message: string` |

---

## 沙箱系统

### 沙箱配置示例

```json
{
  "sandbox": {
    "memory_limit_mb": 4096,
    "cpu_limit_percent": 50,
    "timeout_seconds": 300,
    "allowed_ports": [18000, 18001, 18002],
    "enable_network": false,
    "work_dir": "./sandbox_environments"
  }
}
```

### 沙箱隔离级别

| 级别 | 隔离程度 | 适用场景 |
|------|---------|---------|
| L1 | 独立目录 | 文件操作测试 |
| L2 | 独立进程 | 代码编译测试 |
| L3 | 独立容器 | 完整集成测试 |
| L4 | 独立虚拟机 | 高风险操作 |

---

## 审批流程

### 审批请求结构

```json
{
  "request_id": "req_001",
  "session_id": "session_abc",
  "task_id": "task_123",
  "agent_type": "ENGINE_DEVELOPER",
  "changes_summary": "Added new ParticleSystem class with 3 methods",
  "file_changes": [
    {
      "path": "core/particle/ParticleSystem.h",
      "type": "CREATE",
      "diff_summary": "+120 lines"
    },
    {
      "path": "core/particle/ParticleSystem.cpp",
      "type": "CREATE",
      "diff_summary": "+350 lines"
    }
  ],
  "sandbox_result": {
    "success": true,
    "test_passed": 15,
    "test_failed": 0
  },
  "requested_at": "2026-02-01T10:00:00Z",
  "status": "pending"
}
```

### 审批UI展示

```
┌─────────────────────────────────────────────────────┐
│               Approval Required                      │
├─────────────────────────────────────────────────────┤
│  Agent: ENGINE_DEVELOPER                            │
│  Task: Add ParticleSystem class                     │
│                                                     │
│  Changed Files:                                     │
│    [+]: core/particle/ParticleSystem.h              │
│    [+]: core/particle/ParticleSystem.cpp            │
│                                                     │
│  Sandbox Test: PASSED (15/15)                       │
│                                                     │
│  [Approve]  [Reject]  [View Details]                │
└─────────────────────────────────────────────────────┘
```

---

## 上下文管理

### 上下文组成

```
┌─────────────────────────────────────────────────────┐
│                   Agent Context                      │
├─────────────────────────────────────────────────────┤
│  1. System Prompt                                   │
│     - Agent role definition                         │
│     - Capabilities and limitations                  │
│     - Working guidelines                            │
│                                                     │
│  2. Engine State                                    │
│     - Current scene tree (serialized)               │
│     - Loaded resources                              │
│     - Active nodes and their properties             │
│                                                     │
│  3. Tool Definitions                                │
│     - Available tools with schemas                  │
│     - Usage examples                                │
│                                                     │
│  4. Session History                                 │
│     - Previous messages                             │
│     - Executed commands                             │
│     - Results received                              │
│                                                     │
│  5. Recent Changes Summary                          │
│     - Files modified recently                       │
│     - Test results                                  │
│     - Any pending approvals                         │
└─────────────────────────────────────────────────────┘
```

### 上下文压缩策略

当上下文过长时，自动进行摘要：
1. 压缩历史消息为摘要
2. 保留关键决策点
3. 移除冗长的代码输出
4. 使用向量检索相关历史

---

## 目录结构

```
my_engine/
├── agent/                         # Agent集成层
│   ├── core/                      # 核心组件
│   │   ├── SessionManager.h/cpp
│   │   ├── ToolRegistry.h/cpp
│   │   ├── ContextManager.h/cpp
│   │   └── ToolBase.h
│   ├── remote/                    # 远程通信
│   │   └── LLMClient.h/cpp
│   ├── sandbox/                   # 沙箱系统
│   │   └── SandboxManager.h/cpp
│   ├── workflow/                  # 工作流
│   │   └── WorkFlowManager.h/cpp
│   └── tools/                     # 工具实现
│       ├── FileTools.h/cpp
│       ├── SceneTools.h/cpp
│       ├── RenderTools.h/cpp
│       └── TestTools.h/cpp
├── api_spec/                      # API规范
│   ├── tool_schemas.json          # 工具JSON Schema
│   └── system_prompts/            # 系统提示词
│       ├── engine_dev_agent.txt
│       ├── scene_builder_agent.txt
│       ├── art_generator_agent.txt
│       ├── narrative_agent.txt
│       └── tester_agent.txt
├── core/
│   └── ...
├── scene/
│   └── ...
├── renderer/
│   └── ...
└── scripts/
    └── ...
```

---

## 与引擎的集成点

| 模块 | 集成方式 |
|------|---------|
| `scene/Node.h/cpp` | 添加 `serialize()` / `deserialize()` 方法 |
| `scene/SceneTree.h/cpp` | 添加 `get_serialized_state()` 方法 |
| `renderer/RenderDevice.h/cpp` | 添加 `get_capabilities()` 方法 |
| `scripts/LuaVM.h/cpp` | 添加 `execute_string()` 供Agent执行脚本 |
| `editor/` | 添加 `AgentPanel` 用于可视化操作 |

---

## 下一步

- 查看 [开发计划](../DEVELOPMENT_PLAN.md) 了解实现顺序
- 查看 [API规范](../api_spec/) 了解工具定义格式
