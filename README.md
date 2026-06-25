# ChemSpectra Agent

**AI Autopilot for FTIR Spectral Analysis — Multi-Round Self-Verifying ReAct Agent**

> Global AI Hackathon Series with Qwen Cloud · Track 4: Autopilot Agent  
> Built by a materials science graduate student who taught himself Python in 2024 to solve a real industry pain point.

---

## The Problem

FTIR (Fourier Transform Infrared) spectroscopy is the gold standard for identifying unknown materials in polymer manufacturing, pharma QC, forensics, and environmental testing. But the analysis workflow is painfully manual:

- **30–60 minutes per sample** — reading peaks, searching reference libraries, cross-checking literature, writing reports
- **High error rate** — single-pass library matching misses mixture components and flags false positives
- **No audit trail** — manual analysis lacks reproducibility for regulated industries

## The Solution

ChemSpectra Agent automates this workflow end-to-end using **Qwen-3.7-Max via Alibaba Cloud dashscope SDK**. The agent doesn't just call an API once — it runs a **multi-round reasoning loop** with autonomous self-verification, evidence cross-validation, and self-repair.

**Result: under 2 minutes per sample, with a verifiable confidence trace.**

---

## Architecture

### Four-Phase Agent Loop

```
User: spectrum file or peak positions + natural language request
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1 — ReAct Reasoning Loop                             │
│                                                             │
│  Qwen-3.7-Max (dashscope, enable_thinking=True)             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Think   │ →  │   Act    │ →  │ Observe  │ → repeat     │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
│  Qwen autonomously selects which tools to call:             │
│    identify_material · explain_peaks                        │
│    assign_functional_groups · match_library_topk            │
│    search_public_results                                    │
│  Intent "identify" → 3 tools                               │
│  Intent "explain peaks" → 2 tools                          │
│  Intent "deformulate" → all 5 tools                        │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2 — Evidence Cross-Validation                        │
│                                                             │
│  Automated consistency check across tool results:           │
│  • Does the top material match expected functional groups?  │
│  • Are the top-2 candidates too close in score? (ambiguous) │
│  • Estimate overall confidence (0.0 – 1.0)                  │
│                                                             │
│  Confidence ≥ 0.75 → fast path                             │
│  Confidence < 0.75 or conflict detected → Phase 3          │
└────────────────────────────┬────────────────────────────────┘
                             ▼ (if issues found)
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3 — Autonomous Self-Verification Round               │
│                                                             │
│  Agent is told exactly which conflicts were detected.       │
│  Qwen autonomously calls additional tools to resolve them.  │
│  Confidence is recalculated after verification.             │
│                                                             │
│  Example: confidence 0.72 → runs verification               │
│           → explain_peaks confirms ester linkage            │
│           → confidence rises to 0.94 (+30%)                │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4 — Self-Repair                                      │
│                                                             │
│  If any LLM output fails JSON parsing, the error and        │
│  original output are sent back to Qwen with context.        │
│  The model corrects its own output — no silent failures.    │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Human-in-the-Loop Checkpoint                               │
│                                                             │
│  Agent presents findings + confidence trace + tools used.   │
│  User can ask follow-up questions (Qwen-3.7-Max answers).   │
│  Agent DOES NOT generate final report without confirmation. │
│  Required by pharma, forensics, and materials QC standards. │
└────────────────────────────┬────────────────────────────────┘
                             ▼ (after user confirms)
                    Structured Markdown Report
                    (downloadable, multi-tool evidence, audit trail)
```

### System Components

```
chemspectra-agent/
├── agent.py        # Core agent: ReAct loop, cross-validation,
│                   # self-verification, self-repair, HITL, report
├── tools.py        # FTIR.fun API client (5 tools, REST + MCP)
├── server.py       # FastAPI server + embedded dark-theme web UI
├── report.py       # Standalone report template generator
├── requirements.txt
├── LICENSE         # MIT
├── ARCHITECTURE.md # Detailed architecture + flow diagrams
└── video/          # Demo video + voiceover script
```

### Technology Stack

| Layer | Technology | Provider |
|-------|-----------|----------|
| LLM Reasoning | Qwen-3.7-Max (thinking mode) | **Alibaba Cloud** dashscope SDK |
| Tool Selection | Function Calling (5 tools) | **Alibaba Cloud** dashscope SDK |
| Verification & Repair | Multi-round Qwen calls | **Alibaba Cloud** dashscope SDK |
| Web Server | FastAPI + Uvicorn | — |
| Spectral Analysis API | FTIR.fun REST + MCP | FTIR.fun |
| Spectral Database | 130,000+ reference spectra | FTIR.fun |
| MCP Integration | `/mcp` JSON-RPC endpoint | FTIR.fun |

All LLM calls — tool selection, synthesis, verification, self-repair, follow-up chat, report generation — run through Alibaba Cloud's dashscope SDK.

---

## Quantitative Benefits

| Metric | Typical API Wrapper | ChemSpectra Agent |
|--------|--------------------|--------------------|
| LLM calls per analysis | 1–2 | 3–6 (adaptive) |
| Confidence calibration | None | Tracked per phase |
| Evidence conflict detection | None | Automated |
| Output failure handling | Silent error | Self-repair with retry |
| Low-confidence handling | Returns result as-is | Triggers verification round |
| Demo example | — | 0.72 → 0.94 confidence (+30%) |

---

## The Five Analysis Tools

| Tool | Endpoint | Purpose |
|------|----------|---------|
| `identify_material` | `POST /ftir/identify_material` | Match spectrum against 130K reference library, return ranked candidates |
| `explain_peaks` | `POST /ftir/explain_peaks` | Explain chemical bond vibrations for each peak (cm⁻¹ → bond assignment) |
| `assign_functional_groups` | `POST /ftir/assign_functional_groups` | Map peaks to functional groups (C=O, O-H, N-H, C-O, Si-O, etc.) |
| `match_library_topk` | `POST /ftir/match_library_topk` | Rapid top-K screening without deep analysis |
| `search_public_results` | MCP `/mcp` | Search publicly shared analysis cases |

Qwen decides which tools to call based on user intent. Same model, same spectrum, different questions → different tool combinations.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Alibaba Cloud dashscope API key (get from [modelstudio.console.aliyun.com](https://modelstudio.console.aliyun.com))
- FTIR.fun API key (get from [ftir.fun](https://ftir.fun))

### Installation

```bash
git clone https://github.com/jxbaoxiaodong/chemspectra-agent
cd chemspectra-agent
pip install -r requirements.txt
```

### Configuration

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export FTIRFUN_API_KEY="your-ftirfun-api-key"

# Optional overrides
export QWEN_MODEL="qwen3.7-max"          # default
export FTIRFUN_API_URL="http://127.0.0.1:18080"  # default
```

### Run

```bash
python server.py
# Open http://localhost:8080
```

### Usage

1. Upload a spectrum file **or** enter peak positions manually (comma-separated cm⁻¹ values)
2. Optionally describe the sample (e.g. "polymer film from packaging")
3. Select analysis type: Identify / Explain Peaks / Deformulate / Quick Screen
4. The agent autonomously selects tools, runs multi-round analysis, and returns findings
5. Review results — ask follow-up questions if needed
6. Confirm to generate a downloadable structured report

### Supported Spectrum Input

- **File upload**: SPC, CSV, JDX, JCAMP-DX, OPUS, SPA, XLSX, JSON, TXT, and 20+ more formats
- **Manual peaks**: enter comma-separated wavenumbers, e.g. `2920, 1720, 1230`

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/api/analyze` | Submit spectrum for analysis (multipart/form-data) |
| `POST` | `/api/followup` | Ask follow-up question about current results |
| `POST` | `/api/confirm` | Accept results, generate final report |
| `GET` | `/api/report/{session_id}` | Download Markdown report |
| `GET` | `/health` | Health check |

### `POST /api/analyze`

```
Content-Type: multipart/form-data

file          (optional) — spectrum file
peaks         (optional) — comma-separated cm⁻¹, e.g. "2920,1720,1230"
context       (optional) — sample description
analysis_type (optional) — "identify" | "explain" | "deformulate" | "screen"
```

At least one of `file` or `peaks` is required.

### Response

```json
{
  "step": "awaiting_confirmation",
  "session_id": "abc123",
  "tools_called": ["identify_material", "explain_peaks", "assign_functional_groups"],
  "n_tools": 3,
  "agent_metrics": {
    "react_iterations": 2,
    "verification_rounds": 1,
    "repair_count": 0,
    "evidence_conflicts": 1,
    "confidence_trace": [0.72, 0.94],
    "total_llm_calls": 4
  },
  "confirmation": {
    "best_match": {"name": "Polyethylene Terephthalate", "cas": "25038-59-9", "score": 0.942},
    "verdict": "confirmed",
    "confidence": 0.94,
    "reasoning": "...",
    "candidates": [...],
    "tools_called": [...]
  }
}
```

---

## Background: Why This Project Exists

I'm a materials science graduate student. In my lab, every unknown sample requires FTIR analysis — identifying the material, assigning functional groups, cross-referencing literature. Doing this manually takes 30–60 minutes per sample, and errors in library matching are common.

In 2024, I started teaching myself Python to build a better solution. Not to follow the AI trend — to solve a problem I was living every day.

The result is [FTIR.fun](https://ftir.fun): a production platform with 130,000+ reference spectra, 28 file formats, users in 52 countries. ChemSpectra Agent is the intelligent automation layer on top of that platform — a Qwen-powered agent that does what I used to do manually, with multi-round verification to catch the errors that single-pass matching misses.

---

## About FTIR.fun (the Backend)

ChemSpectra Agent calls the FTIR.fun production API. This is not a toy database:

- **130,000+** reference spectra (polymers, pharmaceuticals, minerals, chemicals, additives)
- **28+ file formats** supported (SPC, SPA, OPUS, JDX, CSV, XLSX, and more)
- **52 countries** — active users worldwide
- **Paying customers** — production service since 2025
- **MCP endpoint** — exposed at `/mcp` for AI tool interoperability

---

## Alibaba Cloud Integration

This project uses Alibaba Cloud exclusively for all LLM capabilities:

- **Model**: `qwen3.7-max` via `dashscope` SDK
- **Thinking mode**: `enable_thinking=True` on reasoning calls — full chain-of-thought visible
- **Function Calling**: 5 structured tool schemas, Qwen autonomously decides which to invoke
- **Multi-round**: 3–6 Qwen API calls per analysis (ReAct loop + verification + synthesis)
- **API endpoint**: `https://dashscope.aliyuncs.com/compatible-mode/v1`

See [`PROOF_ALIBABA_CLOUD.md`](PROOF_ALIBABA_CLOUD.md) for API call evidence.

---

## License

MIT — see [`LICENSE`](LICENSE)

---

## Links

- **Live platform**: [ftir.fun](https://ftir.fun)
- **GitHub**: [github.com/jxbaoxiaodong/chemspectra-agent](https://github.com/jxbaoxiaodong/chemspectra-agent)
- **Hackathon**: [qwencloud-hackathon.devpost.com](https://qwencloud-hackathon.devpost.com)
