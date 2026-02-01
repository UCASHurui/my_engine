# Lightweight Game Engine (名称待定)

基于 C++ 的轻量化 2D/3D 游戏引擎，专注于风格化渲染、AI Agent 协作开发与便捷使用。

## 目录

- [架构概览](#架构概览)
- [核心模块设计](#核心模块设计)
- [风格化渲染](#风格化渲染)
- [Agent 集成](#agent-集成)
- [开发路线](#开发路线)
- [与 Godot 的差异](#与-godot-的差异)
- [Lua 脚本接口](#lua-脚本接口)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 (Application)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Editor    │  │    Game     │  │   Agent Layer           │  │
│  │  (编辑器)   │  │   (运行时)  │  │   (AI Agent 协作)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      场景层 (Scene Layer)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Node/SceneTree  |  Resource System  |  Animation System │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      服务层 (Server Layer)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │Renderer  │ │ Physics  │ │  Audio   │ │   Input/Window   │  │
│  │Server    │ │ Server   │ │ Server   │ │   Server         │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      渲染后端 (Render Backend)                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RenderDevice (Vulkan/OpenGL/Metal)  |  Shader Compiler  │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      平台层 (Platform)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│  │  Windows │ │  Linux   │ │ macOS    │                      │
│  └──────────┘ └──────────┘ └──────────┘                      │
├─────────────────────────────────────────────────────────────────┤
│                      核心层 (Core)                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Object | Variant | ClassDB | Math | String | Containers  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心模块设计

### 核心层 (Core)

提供类型系统、内存管理、基础数据结构。

```
core/
├── object/              # 对象系统基类
│   ├── Object.h/cpp           # 所有对象的基类
│   ├── ClassDB.h/cpp          # 类注册与反射系统
│   ├── RefCounted.h/cpp       # 引用计数基类
│   └── Signal.h/cpp           # 信号机制
├── variant/             # 变体类型系统
│   ├── Variant.h              # Variant 联合体
│   ├── TypeInfo.h             # 类型信息
│   └── Array/Dictionary.cpp   # 容器类型
├── math/                # 数学库
│   ├── Vector2.h/cpp          # 2D 向量
│   ├── Vector3.h/cpp          # 3D 向量
│   ├── Matrix4.h/cpp          # 4x4 矩阵
│   ├── Transform2D.h/cpp      # 2D 变换
│   ├── Transform3D.h/cpp      # 3D 变换
│   └── Math.h/cpp             # 数学工具函数
├── containers/          # 容器库
│   ├── Vector.h              # 动态数组
│   ├── List.h                # 双向链表
│   ├── HashMap.h             # 哈希表
│   └── String.h/cpp          # 字符串
├── io/                  # 输入输出
│   ├── ResourceLoader.h/cpp  # 资源加载
│   └── FileAccess.h/cpp      # 文件访问
└── os/                  # 系统抽象
    ├── OS.h                  # 操作系统接口
    └── Thread.h/cpp          # 线程
```

### 渲染系统 (Renderer)

```
renderer/
├── RenderDevice.h/cpp       # 渲染设备抽象接口
├── RenderPass.h/cpp         # 渲染通道
├── FrameBuffer.h/cpp        # 帧缓冲
├── Pipeline.h/cpp           # 图形管线
├── Shader.h/cpp             # Shader 资源
├── Texture.h/cpp            # 纹理资源
├── Mesh.h/cpp               # 网格资源
└── StyleRenderer.h/cpp      # 风格化渲染器
```

### 场景系统 (Scene)

```
scene/
├── Node.h/cpp               # 场景节点基类
├── SceneTree.h/cpp          # 场景树管理器
├── Viewport.h/cpp           # 视口/相机系统
├── 2d/                      # 2D 节点
│   ├── Node2D.h/cpp
│   ├── Sprite2D.h/cpp
│   ├── AnimatedSprite2D.h/cpp
│   ├── Camera2D.h/cpp
│   ├── Light2D.h/cpp
│   └── TileMap.h/cpp
├── 3d/                      # 3D 节点
│   ├── Node3D.h/cpp
│   ├── MeshInstance3D.h/cpp
│   ├── Camera3D.h/cpp
│   ├── Light3D.h/cpp
│   └── DirectionalLight3D.h/cpp
├── gui/                     # UI 系统
│   ├── Control.h/cpp
│   ├── Label.h/cpp
│   ├── Button.h/cpp
│   └── Panel.h/cpp
└── resources/               # 资源定义
    ├── Resource.h/cpp
    ├── Mesh.h/cpp
    ├── Texture.h/cpp
    ├── Material.h/cpp
    ├── Shader.h/cpp
    └── Animation.h/cpp
```

---

## 风格化渲染

### 风格化模块 (重点)

```
renderer/stylized/
├── ToonShading.h/cpp        # 卡通渲染 (Cel/Toon Shading)
├── OutlinePass.h/cpp        # 描边效果 (Inverted Hull / Post-process)
├── HatchingRender.h/cpp     # 手绘风格渲染
├── GradientRamp.h/cpp       # 渐变映射
├── RimLight.h/cpp           # 边缘光效果
├── Dithering.h/cpp          # 抖动效果 (复古感)
├── PixelArtFilter.h/cpp     # 像素化滤镜
├── Halftone.h/cpp           # 半调图案
├── CrossHatching.h/cpp      # 十字影线
├── WaterEffect.h/cpp        # 水彩效果
└── PostProcess.h/cpp        # 后处理系统
```

### 预设风格化 Shader

```glsl
// 卡通渲染 Shader
shader_type spatial;

uniform vec3 baseColor : source_color = vec3(1.0);
uniform sampler2D rampTexture;  // 渐变映射表
uniform int shadeLevels = 3;

void fragment() {
    float NdotL = dot(NORMAL, LIGHT);
    float shade = floor(NdotL * float(shadeLevels)) / float(shadeLevels);
    ALBEDO = baseColor * (ambient + diffuse * shade);
}

// 描边 Shader (Inverted Hull)
shader_type spatial;
render_mode cull_disabled;

uniform float outlineWidth = 0.02;
uniform vec3 outlineColor : source_color = vec3(0.0);

void vertex() {
    POSITION += NORMAL * outlineWidth;
}

void fragment() {
    ALBEDO = outlineColor;
}
```

---

## Agent 集成

引擎支持 AI Agent 协作开发，实现人与 Agent 共同迭代引擎。

### 设计目标

| 目标 | 描述 |
|------|------|
| **多Agent协作** | 支持引擎迭代、场景建模、艺术创作、测试等多种Agent |
| **双模式支持** | 纯人工模式 + 人机协作模式无缝切换 |
| **安全可控** | 沙箱测试 + 人工审批，确保代码安全 |
| **远程LLM** | 通过 OpenAI/Anthropic/DeepSeek API 调用大模型 |

### Agent 类型

| Agent | 职责 |
|-------|------|
| ENGINE_DEVELOPER | 引擎迭代开发 |
| SCENE_BUILDER | 场景建模 |
| ART_GENERATOR | 艺术创作 |
| NARRATIVE_WRITER | 剧情编写 |
| TESTER | 测试 |
| RENDER_SPECIALIST | 渲染优化 |

### 工作流程

```
用户指令 → Agent解析 → 工具调用 → 沙箱测试 → 审批 → 应用
```

### 核心组件

```
agent/
├── core/              # SessionManager, ToolRegistry, ContextManager
├── remote/            # LLMClient (远程API通信)
├── sandbox/           # SandboxManager (隔离测试)
├── workflow/          # WorkFlowManager (工作流审批)
└── tools/             # FileTools, SceneTools, RenderTools
```

详见 [Agent 集成文档](docs/AGENT_INTEGRATION.md)

---

## 开发路线

### Phase 1: 核心基础 (2-3 周)
- [ ] 实现 Core 层 (Object, Variant, ClassDB)
- [ ] 实现基础数学库 (Vector, Matrix, Transform)
- [ ] 实现内存管理 (RefCounted, Object)

### Phase 2: 渲染基础 (3-4 周)
- [ ] 实现 RenderDevice 抽象 (OpenGL/Vulkan)
- [ ] 实现 Shader 编译和加载
- [ ] 实现基础 Mesh/Texture 系统
- [ ] 实现 2D 渲染 (Canvas 模式)

### Phase 3: 场景系统 (2-3 周)
- [ ] 实现 Node/SceneTree
- [ ] 实现 2D 节点 (Sprite, Camera2D, Light2D)
- [ ] 实现 3D 节点基础 (Camera3D, MeshInstance3D)

### Phase 4: 风格化渲染 (3-4 周)
- [ ] 实现卡通渲染 (Cel Shading)
- [ ] 实现描边效果 (Outline)
- [ ] 实现后处理系统
- [ ] 预设风格化材质库

### Phase 5: 脚本与工具 (2-3 周)
- [ ] Lua 脚本绑定
- [ ] JSON 场景文件格式
- [ ] 基础工具链

### Phase 6: Agent 集成 (4-5 周) - 新增
- [ ] 实现 LLMClient 远程API通信
- [ ] 实现 ToolRegistry 工具注册中心
- [ ] 实现 SandboxManager 沙箱管理器
- [ ] 实现 SessionManager 会话管理
- [ ] 实现 WorkFlowManager 工作流与审批
- [ ] 实现 Agent Panel 可视化面板

详见 [Agent 集成文档](docs/AGENT_INTEGRATION.md) 和 [开发计划](docs/DEVELOPMENT_PLAN.md)

---

## 与 Godot 的差异

| 特性 | Godot | 你的引擎 |
|------|-------|---------|
| 规模 | 完整引擎 (大) | 轻量化 (小) |
| 脚本 | GDScript (自研) | Lua (成熟) |
| 编辑器 | 完整 IDE | 后期/简单 |
| 渲染后端 | Vulkan/Metal/D3D | OpenGL/Vulkan |
| 风格化 | 通用 | **重点优化** |
| 物理 | Jolt/Bullet | Box2D/PhysX (可选) |

---

## Lua 脚本接口

### 为什么选择 Lua？

#### 1. 快速迭代与热重载

| 特性 | C++ | Lua |
|------|-----|-----|
| 编译时间 | 需要编译，秒级~分钟级 | 无需编译，毫秒级 |
| 热重载 | 需重启程序 | 运行时热重载 |
| 开发效率 | 编写-编译-运行循环 | 编写-运行循环 |

```lua
-- 修改后立即生效，无需重启游戏
function on_hit(enemy, damage)
    enemy:take_damage(damage)
    if enemy.hp <= 0 then
        enemy:die()
    end
end
```

#### 2. 学习曲线低

```
C++ 学习曲线:
    ████████████████████████████████████  (6-12个月+)

Lua 学习曲线:
    ████████  (1-2周)
```

- 语法简洁，接近 Python/JavaScript
- 无需理解指针、内存管理、模板等复杂概念
- 适合策划、美术等非程序员使用

#### 3. 嵌入成本低

- Lua 虚拟机仅约 200KB
- 可直接嵌入到 C++ 程序中
- 与 C++ 通过 C API 无缝交互

#### 4. 游戏行业广泛验证

| 游戏引擎 | 脚本语言 |
|---------|---------|
| Roblox | Lua |
| CryEngine | Lua |
| Corona SDK | Lua |
| LÖVE | Lua |
| WoW/FFXIV | Lua |

#### 5. 职责分离

```
┌─────────────────────────────────────┐
│           C++ 引擎核心              │
│  ┌─────────────────────────────────┐│
│  │  • 渲染 (高性能)                ││
│  │  • 物理 (C++ 实现)              ││
│  │  • 内存管理                     ││
│  │  • 平台抽象                     ││
│  └─────────────────────────────────┘│
│                  ↓                  │
│           Lua 脚本层                │
│  ┌─────────────────────────────────┐│
│  │  • 游戏逻辑                     ││
│  │  • 关卡设计                     ││
│  │  • AI 行为                     ││
│  │  • UI 交互                     ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

### C++ vs Lua 使用场景对比

| 场景 | 推荐使用 C++ | 推荐使用 Lua |
|------|-------------|-------------|
| 核心渲染管线 | ✅ | ❌ |
| 物理引擎 | ✅ | ❌ |
| 游戏逻辑/AI | ❌ | ✅ |
| 关卡配置 | ❌ | ✅ |
| 工具脚本 | ❌ | ✅ |
| 性能关键代码 | ✅ | ❌ |
| 快速原型 | ❌ | ✅ |

### 示例：C++ 与 Lua 交互

```cpp
// C++ 端：注册函数到 Lua
extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

int l_player_take_damage(lua_State* L) {
    int hp = (int)lua_tointeger(L, 1);
    int damage = (int)lua_tointeger(L, 2);

    // C++ 引擎内部处理
    Player* player = get_current_player();
    player->take_damage(damage);

    lua_pushinteger(L, player->get_hp());
    return 1;
}

// 注册到 Lua 环境
lua_register(L, "take_damage", l_player_take_damage);
```

```lua
-- Lua 端：使用 C++ 提供的功能
function on_click()
    take_damage(100)  -- 调用 C++ 实现
    print("玩家受伤！剩余血量: " .. get_player_hp())
end
```

### 总结

选择 Lua 的核心理由：

1. **开发效率**：热重载、快速迭代
2. **团队协作**：降低非程序员的上手门槛
3. **资源占用**：轻量级虚拟机
4. **行业验证**：大量成功案例
5. **职责分离**：C++ 专注性能，Lua 专注灵活

> **最佳实践**：核心模块用 C++ 实现确保性能，游戏逻辑用 Lua 编写确保灵活。
