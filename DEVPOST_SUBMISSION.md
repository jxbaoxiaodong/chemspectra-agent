# Devpost Submission — ChemSpectra Agent

> Global AI Hackathon Series with Qwen Cloud — Track 4: Autopilot Agent

---

## 项目名称（建议）

**ChemSpectra Agent — AI Autopilot for FTIR Spectral Analysis**

---

## 一句话描述（Tagline）

One-click FTIR spectral analysis autopilot: upload a spectrum, get a verified material identification report with DOI-cited evidence — powered by Qwen + 130K spectral library.

---

## 正文描述（Text Description）

### What it does

ChemSpectra Agent is a production-grade AI autopilot that automates the entire FTIR (Fourier Transform Infrared) spectral analysis pipeline end-to-end. A chemist or QC engineer uploads a spectrum file in any of 28+ instrument formats — the Agent handles everything:

1. **Intent Parsing** (Qwen): Extracts analysis type, sample context, and special concerns from natural language
2. **Spectral Library Search** (MCP Tool): Searches 130,000+ reference spectra via FTIR.fun's domain engine
3. **AI Chemical Verification** (Qwen): Reviews matches for functional group consistency, peak shift analysis, and mixture detection
4. **Human-in-the-Loop**: Presents findings with evidence and explicitly asks for confirmation before proceeding — a critical requirement for regulated industries (pharma, forensics)
5. **Report Generation**: Produces a structured analysis report with DOI-cited peak assignments

Traditional manual analysis takes 30-60 minutes per sample. ChemSpectra Agent reduces this to under 2 minutes while maintaining the human analyst's authority as the final decision-maker.

### How we built it

- **Agent Core**: Custom Python agent loop orchestrating the 5-step pipeline
- **Qwen Integration**: dashscope SDK (Alibaba Cloud) with Qwen-Max for chemical reasoning, intent parsing, and report generation
- **External Tools via MCP**: FTIR.fun MCP Server provides `analyze_ftir_spectrum` tool — 130K spectral library search, 28+ file format parsers, peak-to-functional-group mapping with DOI citations
- **Web Server**: FastAPI with built-in Web UI for file upload and human-in-the-loop confirmation (Alibaba Cloud proof: dashscope SDK for Qwen-Max)
- **Human-in-the-Loop**: Deliberate checkpoint design where the agent cannot finalize a report without human confirmation

### Why Qwen Cloud

Qwen-Max's strong reasoning capabilities are essential for the chemical verification step — it's not enough to return a similarity score; the agent must explain WHY a match is chemically plausible. Qwen's ability to reason about functional groups, peak assignments, and potential interferences makes it the ideal brain for this domain-specific autopilot.

### Challenges we ran into

- Designing the human-in-the-loop checkpoint to be both non-disruptive and genuinely useful
- Handling the 28+ spectrum file formats through the MCP interface without pre-processing
- Ensuring the agent NEVER fabricates chemical names or CAS numbers (domain constraint)

### What we learned

MCP (Model Context Protocol) is a powerful pattern for connecting LLMs to domain-specific tools. By separating "chemical reasoning" (Qwen) from "spectral matching" (FTIR.fun MCP), we got the best of both worlds without either system needing to understand the other's internals.

### What's next

- Multi-spectrum batch analysis for QC labs
- Integration with LIMS (Laboratory Information Management Systems)
- Edge deployment on handheld FTIR spectrometers via Qwen-EdgeAgent

---

## 评分维度自述（用于提交页面）

| 维度 | 自述 |
|------|------|
| Technical Depth (30%) | MCP tool integration, multi-step agent loop, real 130K spectral database, 28+ file format support, dashscope SDK integration |
| Innovation (30%) | First LLM agent for chemical spectral analysis; combines MCP tools with reasoning chain; regulated-industry HITL design |
| Impact (25%) | Pharma QC, polymer manufacturing, forensics, environmental testing — multi-billion dollar industries |
| Presentation (15%) | Architecture diagram in README, 3-min demo video, open-source MIT codebase |

---

## 演示视频脚本（3分钟）

### 0:00-0:20 — 开场
"Hi, I'm building ChemSpectra Agent for the Qwen Cloud Hackathon, Track 4: Autopilot Agent. This is an AI autopilot for chemical spectral analysis. Let me show you how it turns a 30-minute manual workflow into a 2-minute automated pipeline."

### 0:20-0:50 — 上传光谱
"Here's my web interface. I'll upload a spectrum file — this is a .csv from an FTIR instrument. I'll add some context: 'Polymer film from production, suspected PE contamination.' Select 'Identify unknown material' and hit Analyze. The agent runs Qwen-Max via Alibaba Cloud dashscope for all reasoning."

### 0:50-1:30 — Agent 工作
"The agent is now running. Behind the scenes, Qwen-Max parses my intent, then calls the FTIR.fun MCP tool to search 130,000 reference spectra. Within seconds, we get ranked matches with similarity scores. But the agent doesn't stop there — Qwen now performs chemical reasoning, checking if the functional groups are consistent with the proposed material."

### 1:30-2:10 — Human-in-the-Loop
"Here's the key part — the agent presents findings but DOES NOT finalize. It says: 'Best match is Polyethylene. Confidence 94%. Chemical reasoning confirms C-H stretching at 2918 and 2848 cm⁻¹. No unexplained peaks.' Now I, as the human analyst, must confirm. This human-in-the-loop checkpoint is critical for regulated industries."

### 2:10-2:40 — 报告生成
"I confirm, and the agent generates a structured report with DOI-cited evidence for each peak assignment. The report includes PCA quality control notes and my confirmation stamp."

### 2:40-3:00 — 总结
"ChemSpectra Agent: Production-grade chemical analysis autopilot, powered by Qwen and FTIR.fun. Built on Alibaba Cloud. Open source on GitHub. Thank you!"

---

## 图床链接占位（架构图已内嵌于GitHub README）

架构图使用 ASCII art 直接在 README.md 和 ARCHITECTURE.md 中展示，无需外部图床。
