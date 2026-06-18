# ChemSpectra Agent — AI Autopilot for Chemical Spectral Analysis

> **Track 4: Autopilot Agent** — Global AI Hackathon Series with Qwen Cloud  
> Automates real-world material identification workflows end-to-end.

## What It Does

ChemSpectra Agent is a production-grade AI autopilot for FTIR (Fourier Transform Infrared) spectral analysis. A chemist uploads a spectrum file — the Agent handles parsing, 130K-library search, AI chemical verification, human-in-the-loop review, and report generation.

**Traditional workflow:** 30-60 min per sample (manual peak interpretation → library search → literature cross-reference → report writing)  
**ChemSpectra Agent:** <2 min per sample (automated end-to-end)

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Local / Cloud (FastAPI)                        │
│                                                                   │
│  ┌──────────┐     ┌─────────────────┐     ┌────────────────────┐ │
│  │  Web UI  │────▶│   Qwen Agent    │────▶│  FTIR.fun MCP      │ │
│  │ (FastAPI)│     │  (dashscope     │     │  Server            │ │
│  │          │     │   Qwen-Max)     │     │  (ftir.fun:18081)  │ │
│  └──────────┘     │                 │     │                    │ │
│                   │ • Intent parse  │     │ • Parse spectrum   │ │
│                   │ • Chem reasoning│     │ • Search 130K lib  │ │
│                   │ • Human-in-loop │     │ • AI verification  │ │
│                   │ • Report gen    │     │ • DOI citations    │ │
│                   └─────────────────┘     └────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Workflow

1. **Upload** → User provides spectrum file (.spc/.csv/.jdx/.opus + 25 formats) + sample context
2. **Parse** → Qwen Agent extracts user intent, sample metadata, analysis requirements
3. **Search** → MCP Tool `analyze_ftir_spectrum` → 130K spectral library → ranked matches
4. **Verify** → Qwen performs chemical reasoning: functional group consistency, peak shift analysis, mixture detection
5. **Confirm** → Human-in-the-loop checkpoint: Agent presents findings, asks for confirmation
6. **Report** → Structured analysis report with DOI-cited evidence generated upon confirmation

## Scoring Alignment

| Criterion (Weight) | How We Address It |
|--------------------|--------------------|
| Technical Depth (30%) | MCP tool integration, multi-step agentic workflow, real 130K spectral database |
| Innovation (30%) | First LLM + spectral analysis agent; chemical reasoning chain; regulated-industry HITL |
| Impact (25%) | Solves real pain in pharma QC, polymer mfg, forensics, environmental testing |
| Presentation (15%) | Architecture diagram, 3-min demo video, open-source MIT license |

## Quick Start

```bash
# Prerequisites
pip install -r requirements.txt

# Set API keys (get from https://ftir.fun and Alibaba Cloud)
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
export FTIRFUN_API_KEY="ftir-xxxxxxxxxxxxxxxx"

# Run
python server.py
# → http://localhost:8080
```

## Proof of Alibaba Cloud

See [`PROOF_ALIBABA_CLOUD.md`](PROOF_ALIBABA_CLOUD.md).  
Key Alibaba Cloud service used: **dashscope SDK** (Qwen-Max model) — the agent's reasoning engine runs on Alibaba Cloud infrastructure via API calls. Server can run locally or on any cloud; Alibaba Cloud proof is the dashscope SDK integration.

## Project Structure

```
├── agent.py                     # Core Agent: Qwen + MCP orchestration
├── tools.py                     # FTIR.fun MCP client wrapper
├── server.py                    # FastAPI web server (local / any cloud)
├── report.py                    # Analysis report generator
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT
├── ARCHITECTURE.md              # Detailed architecture
├── PROOF_ALIBABA_CLOUD.md        # Alibaba Cloud deployment proof
└── README.md                    # This file
```

## 参赛待办清单

- [ ] **注册 Devpost**：[qwencloud-hackathon.devpost.com](https://qwencloud-hackathon.devpost.com/) → 点 "Join Hackathon"
- [ ] **注册 Qwen Cloud**：[qwencloud.com](https://qwencloud.com) → 获取免费额度 + API Key → 填入 `DASHSCOPE_API_KEY`
- [ ] **本地运行测试**：`python server.py` → http://localhost:8080，录屏时展示阿里云控制台 + dashscope API Key 作为部署证明
- [ ] **获取 FTIR.fun API Key**：[ftir.fun](https://ftir.fun) → Personal → API Keys
- [ ] **端到端测试**：用真实光谱文件跑通完整 5 步 pipeline
- [ ] **录制演示视频**：3 分钟，按 `DEVPOST_SUBMISSION.md` 里的脚本，上传 YouTube/Youku
- [ ] **提交 Devpost**：贴 README 内容 + 架构图 + 视频链接 + GitHub 仓库链接
- [ ] **（可选）写博客**：分享构建过程，争取 Blog Post 奖金（$500 × 10）

## Development

### GitHub Token

本仓库推送使用 ftirfun 项目的 GitHub Token，位于：

```
/home/bob/projects/ftirfun/.env → GITHUB_TOKEN=ghp_xxx
```

### 修改 & 推送

```bash
cd /home/bob/projects/qwen-hackathon

# 修改代码后
git add -A
git commit -m "描述你的改动"

# 推送（token 自动从 .env 读取）
bash -c 'TOKEN=$(grep GITHUB_TOKEN /home/bob/projects/ftirfun/.env | cut -d= -f2-); git push "https://x-access-token:${TOKEN}@github.com/jxbaoxiaodong/chemspectra-agent.git" main'
```

> Token 过期时，去 GitHub Settings → Developer settings → Personal access tokens 重新生成，更新 `/home/bob/projects/ftirfun/.env` 中的 `GITHUB_TOKEN` 即可。

## License

MIT
