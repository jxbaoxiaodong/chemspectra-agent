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
│                     Alibaba Cloud ECS                             │
│                                                                   │
│  ┌──────────┐     ┌─────────────────┐     ┌────────────────────┐ │
│  │  Web UI  │────▶│   Qwen Agent    │────▶│  FTIR.fun MCP      │ │
│  │ (FastAPI │     │  (dashscope     │     │  Server            │ │
│  │  on ECS) │     │   Qwen-Max)     │     │  (ftir.fun:18081)  │ │
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
Key Alibaba Cloud services used: **dashscope SDK** (Qwen-Max model), **ECS** (FastAPI hosting).

## Project Structure

```
├── agent.py                     # Core Agent: Qwen + MCP orchestration
├── tools.py                     # FTIR.fun MCP client wrapper
├── server.py                    # FastAPI web server (Alibaba Cloud ECS)
├── report.py                    # Analysis report generator
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT
├── ARCHITECTURE.md              # Detailed architecture
├── PROOF_ALIBABA_CLOUD.md        # Alibaba Cloud deployment proof
└── README.md                    # This file
```

## License

MIT
