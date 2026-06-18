# Qwen Cloud Hackathon — 参赛要求与待办

> Track 4: Autopilot Agent | 截止: 2026-07-09 | 奖金: $45,000

---

## 一、赛事提交清单（7 项必须 + 1 项可选）

从 [官方规则](https://qwencloud-hackathon.devpost.com/rules) 逐条提取：

| # | 必须 | 要求 | 落实文件 |
|---|------|------|----------|
| 1 | ✅ | **公开代码仓库 + 开源许可证** | 仓库 `chemspectra-agent` 已创建，含 `LICENSE` (MIT) |
| 2 | ✅ | **文字描述** — 解释项目功能 | 见下方「文字描述模板」 |
| 3 | ✅ | **阿里云部署证明** — 代码中有阿里云服务调用的文件链接 | `agent.py` 中 `dashscope` SDK 调用视为证明 |
| 4 | ✅ | **架构图** — 系统组件关系图 | `ARCHITECTURE.md` |
| 5 | ✅ | **演示视频 ≤3 分钟** — YouTube/Vimeo/Youku 公开 | 见下方「录屏指南」 |
| 6 | ✅ | **指定赛道** — 提交时选 Track 4 | Devpost 提交表单 |
| 7 | ✅ | **使用 Qwen 模型** — 项目须调用 Qwen Cloud API | `agent.py:104` — `dashscope.Generation.call(model="qwen-max")` |
| 8 | ⭐ | **（可选）博客文章** — 分享构建过程，争取 Blog Post 奖（$500×10） | 发布后链接填入 Devpost |

---

## 二、评分标准

| 维度 | 权重 | 意思 |
|------|------|------|
| Technical Depth & Engineering | 30% | 代码质量、架构复杂度、是否善用 Qwen Cloud API（MCP 集成加分）、性能优化 |
| Innovation & AI Creativity | 30% | 创意、是否用了非平凡逻辑、技术栈是否先进 |
| Problem Value & Impact | 25% | 是否解决真实痛点、可规模化、可开源社区采纳 |
| Presentation & Documentation | 15% | Demo 清晰、文档完善、架构图可读 |

---

## 三、资格要求

- ✅ 年满 18 岁
- ✅ 中国不在排除国家之列
- ✅ 用 Qwen Cloud 上的模型（你的通义 key 直接可用）
- ✅ 后端须展示阿里云服务调用（dashscope SDK 即可，无需买 ECS）

---

## 四、文字描述模板

以下可直接贴到 Devpost 提交页（等代码跑通后再微调具体数值）：

```
ChemSpectra Agent is an AI autopilot for FTIR spectral analysis.
Track 4: Autopilot Agent.

WORKFLOW:
1. User uploads spectrum file (28+ formats supported) + sample description
2. Qwen-Max parses intent and extracts analysis requirements
3. Agent invokes FTIR.fun API to search 130,000 reference spectra
4. Qwen-Max performs chemical reasoning: verifies functional group consistency,
   checks for peak shifts, detects potential mixtures
5. Human-in-the-loop checkpoint: agent presents findings and requires explicit
   confirmation before proceeding
6. After confirmation, agent generates structured report with DOI-cited evidence

TECH STACK:
- Qwen-Max via Alibaba Cloud dashscope SDK
- FTIR.fun spectral library API (130K spectra, 28+ formats)
- FastAPI web server with built-in UI
- Custom 5-step agent loop

Traditional manual analysis: 30-60 min per sample.
ChemSpectra Agent: <2 min per sample.
```

---

## 五、当前状态 & 下一步

### 已完成

- [x] GitHub 仓库创建并推送: https://github.com/jxbaoxiaodong/chemspectra-agent
- [x] Devpost 注册
- [x] 通义 API Key 验证通过（`sk-f3770...` — `qwen-plus` 返回正常）
- [x] 代码骨架完成（agent.py / tools.py / server.py / report.py）

### 待做

| 步骤 | 具体操作 | 依赖 |
|------|----------|------|
| **1. 修复 tools.py** | 当前 MCP 客户端直接调 `/mcp` 走不通。需改为用 `X-API-Key` 头调 REST API（`/v1/search`），或者直接调 Django 内部 `search_spectrum()` | 需要 FTIR API key（`runtime/private/fastapi.env` 里有） |
| **2. 端到端测试** | 用真实光谱文件跑通完整 5 步 pipeline（上传→解析→搜索→AI 复核→确认→报告） | 步骤 1 完成 |
| **3. 准备测试光谱文件** | 需要一个 `.csv` 或 `.spc` 格式的 FTIR 光谱文件用于 demo | — |
| **4. 完善 Web UI** | 确保录屏时界面看起来专业 | 步骤 2 完成 |
| **5. 录制视频** | 3 分钟，上传 YouTube/Youku | 步骤 4 完成 |
| **6. Devpost 提交** | 贴链接 + 描述 + 架构图 + 视频链 | 全部完成 |

---

## 六、录屏指南

### 视频内容（3 分钟）

1. **(0:00-0:30)** 开场：你是谁、项目叫什么、Track 4
2. **(0:30-0:50)** 展示 Web UI → 上传光谱文件 → 输入样品描述
3. **(0:50-1:30)** Agent 工作过程：展示终端日志或 UI 状态变化（Intent parse → Library search → Chemical reasoning）
4. **(1:30-2:00)** Human-in-the-loop：展示 AI 给出结果但等待人工确认
5. **(2:00-2:30)** 确认后生成报告
6. **(2:30-3:00)** 总结 + GitHub 链接

### 要点

- 录屏时切换一次终端，展示 `env | grep DASHSCOPE` 证明用了阿里云 key
- 视频上传 YouTube（设公开）或 Youku
- 视频里不能有第三方版权音乐

---

## 七、可用资源

| 资源 | 路径 |
|------|------|
| 通义 API Key | `ftirfun/settings/ftir.py:21` → `TONGYI_KEY` |
| FTIR API Key | `runtime/private/fastapi.env` → `FTIRFUN_API_KEYS` |
| 代码仓库 | https://github.com/jxbaoxiaodong/chemspectra-agent |
| GitHub Token | `/home/bob/projects/ftirfun/.env` → `GITHUB_TOKEN` |
| 推送命令 | 见 README.md 底部 |
| Devpost 页面 | https://qwencloud-hackathon.devpost.com/ |

---

## 八、关于不需要买服务器

评审要求的是"使用了阿里云服务"，不是"服务器托管在阿里云"。`dashscope` SDK 调 Qwen API 就是阿里云服务。录屏时展示一下阿里云控制台或 key 即可。不需要花一分钱。
