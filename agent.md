# Agent 开发日志

记录 Claude Code 协助开发 ChemSpectra Agent 的完整过程。

---

## 2026-06-19: 多工具自主路由 Agent

### 背景

项目此前是固定 5 步流水线（解析意图 → 谱库搜索 → 化学验证 → 人类确认 → 报告生成），始终只调用一个 API 端点 `/ftir/analyze_spectrum`。从黑客松评审角度看，这更像 API wrapper 而非 Autopilot Agent。

分析 FTIR.fun 后端后发现共有 5 个 REST 端点可用，且全部共享相同的请求/响应模型 (`CanonicalFtirRequest` → `SearchResponse`)：

| 端点 | goal | 用途 |
|------|------|------|
| `/ftir/identify_material` | identification | 材料鉴定 |
| `/ftir/explain_peaks` | explanation | 峰位解释 |
| `/ftir/assign_functional_groups` | functional_groups | 官能团归属 |
| `/ftir/match_library_topk` | matching | Top-K 快速匹配 |
| `/ftir/analyze_spectrum` | 动态 | 全流程分析 |

另有 MCP 端点 `search`（端口 18081）可搜索公开分析结果。

### 实施过程

#### 1. tools.py — 扩展 API 客户端

- 提取公共方法 `_build_body()` 构建 `CanonicalFtirRequest` 请求体、`_post()` 统一发送请求
- 在原有 `analyze_spectrum()` 基础上新增 4 个方法：`identify_material()`、`explain_peaks()`、`assign_functional_groups()`、`match_library_topk()`
- 每个方法对应一个 REST 端点，通过 `goal` 和 `sample_type` 参数区分分析目标
- `search_public_results()` 已存在，保留不变

#### 2. agent.py — 从固定流水线重构为 ReAct 循环

核心改动：用 Qwen Function Calling 替代硬编码的步骤序列。

**新增常量：**
- `AGENT_TOOLS` — 5 个工具的 Function Calling schema 定义，包含中英文描述和适用场景说明

**新增方法：**
- `_call_qwen_with_tools()` — 带 `tools` 参数调用 dashscope `Generation.call()`，返回完整 choice 对象（含 `tool_calls`）
- `run_tool_loop()` — ReAct 循环核心：发送用户请求 + 工具定义 → Qwen 返回 tool_calls → 执行工具 → 将结果回传 → Qwen 决定继续调用或生成最终综合分析。最多 6 轮迭代
- `_execute_tool()` — 根据工具名分发到 `FtirfunClient` 对应方法，光谱数据从 Session 中获取
- `_format_tool_result_for_llm()` — 将 API 返回的 JSON 格式化为 LLM 可读的文本摘要
- `extract_verification()` — 从多工具综合分析中提取结构化判定（verdict/confidence/flags）

**删除方法：**
- `parse_intent()` — 不再需要单独意图解析，Qwen 在 ReAct 循环中通过 Function Calling 直接表达意图
- `search_library()` — 不再直接调用单一搜索端点，由 `_execute_tool()` 按需分发
- `verify_results()` — 被 `extract_verification()` 替代

**修改方法：**
- `run_pipeline()` — 流程改为：存储光谱到 Session → `run_tool_loop()` → `extract_verification()` → `build_confirmation_payload()`
- `build_confirmation_payload()` — 返回数据增加 `tools_called` 和 `synthesis` 字段
- `handle_followup()` — 追问上下文包含工具使用记录和综合分析
- `generate_report()` — 报告 prompt 包含工具列表、峰位解释、官能团证据等多工具信息

**Session 新增字段：**
- `file_base64`、`filename`、`peaks` — 光谱数据存到 Session 供多次工具调用复用
- `tool_calls_log` — 工具调用日志（工具名、参数、成功/失败、匹配数）
- `tool_results` — 各工具的完整返回结果（按工具名索引）
- `synthesis` — Qwen 的多工具综合分析文本

#### 3. server.py — 前端展示多工具信息

- Pipeline 步骤文案更新：Reasoning → Executing → Synthesizing → Confirmation → Report
- 结果页新增 "Tools Used by Agent" 标签区，用蓝色圆角标签显示 Agent 选择了哪些工具
- 结果页新增 "Agent Synthesis" 区块，展示 Qwen 的多工具综合分析文本
- `renderResults()` JS 函数更新，从 `confirmation.tools_called` 和 `confirmation.synthesis` 读取数据

#### 4. 文档更新

- `README.md` — 架构图改为 ReAct Multi-Tool Agent 流程，Features 增加多工具描述，Project Structure 更新各文件职责
- `ARCHITECTURE.md` — 完整重写，包含 ReAct 循环流程图、工具选择策略表、技术栈更新

### 端到端测试结果

测试在启动服务器后通过 curl 执行，验证 Qwen 的工具选择行为：

| 测试场景 | 用户输入 | Qwen 选择的工具 | 结果 |
|----------|----------|-----------------|------|
| PE 鉴定 | peaks=2920,2850,1460,720 + "suspected polyethylene" | `identify_material` + `explain_peaks` | HDPE, score=0.4938 |
| 官能团查询 | peaks=1730,1250,1100 + "what functional groups?" | `assign_functional_groups` + `explain_peaks` | C=O, C-O 等 |
| 文件上传全鉴定 | test.csv + "full identification" | `identify_material` + `explain_peaks` + `assign_functional_groups` | 4-nitrobenzaldehyde, score=0.7092 |
| 多轮追问 | "Why is the confidence score so low for HDPE?" | — | 正常回答，引用工具结果 |
| 报告生成 | confirm → accept | — | 报告包含多工具分析方法和证据 |

### 遇到的问题

1. **API 401 错误** — 环境变量名不匹配：fastapi.env 中是 `FTIRFUN_API_KEYS`（复数），代码读的是 `FTIRFUN_API_KEY`（单数）。启动服务器时需手动 `export FTIRFUN_API_KEY=...`。

### 未改动的文件

- `report.py` — 独立报告模板生成器，当前未被主流程使用
- `requirements.txt` — 依赖无变化
- `PROOF_ALIBABA_CLOUD.md` — 阿里云证明无变化

---

## 2026-06-15 ~ 2026-06-18: 初始版本 + 功能补全（上一轮对话）

### 初始搭建

- 创建项目结构：agent.py / tools.py / server.py / report.py
- tools.py 最初使用 MCP JSON-RPC 调用 `https://ftir.fun/mcp`，遇到 401 后改为 REST API `http://127.0.0.1:18080` + `X-API-Key`
- agent.py 实现固定 5 步流水线
- server.py 实现 FastAPI + 嵌入式暗色主题前端

### 四项功能补全

用户选择同时修复四个功能缺失：

1. **文件上传模式** — server.py 的 `/api/analyze` 接收 multipart form，base64 编码后传给 agent
2. **并发安全** — 引入 `Session` dataclass，每请求独立 UUID，消除全局状态
3. **多轮对话** — `/api/followup` 端点 + 前端聊天框，Session 中维护 conversation 历史
4. **报告导出** — `/api/confirm` 生成报告 + `/api/report/{session_id}` 下载 Markdown 文件

### 遇到的问题

- MCP 401 → 改用 REST API + X-API-Key
- Cloudflare 403 → 改用本地端点 127.0.0.1:18080
- REST 422 → 请求体需要嵌套 `CanonicalFtirRequest` 格式
- `RESULT_REVIEW_MODE` 导入错误 → 重启 FastAPI 服务（stale `sys.modules` 缓存）
- Qwen 返回 JSON 被 markdown 代码块包裹 → 添加 `_extract_json()` 正则提取
- 端口 8080 被占用 → `lsof -ti :8080 | xargs kill`
- 全局 agent 状态被并发覆盖 → 引入 Session dataclass
