# ChemSpectra Agent — Architecture Documentation

## System Overview

ChemSpectra Agent is an AI-powered autopilot for FTIR spectral analysis,
built for **Track 4: Autopilot Agent** of the Qwen Cloud Hackathon.

```
┌──────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Web UI (FastAPI + vanilla HTML/JS)                            │ │
│  │  • File upload (.spc, .csv, .jdx, .opus, .xlsx … 25+ formats) │ │
│  │  • Sample context input                                        │ │
│  │  • Analysis type selection                                     │ │
│  │  • Human-in-the-loop confirmation buttons                      │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                      │
│                    FastAPI (local or any cloud)                      │
└───────────────────────────────┼──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      QWEN AGENT (dashscope)                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Step 1: Parse Intent                                          │ │
│  │  • Extract sample description, analysis type, special concerns │ │
│  │  • Identify peak positions if mentioned in natural language    │ │
│  │  • Model: Qwen-Max via Alibaba Cloud ModelStudio               │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │  Step 2: MCP Tool Call (spectral search)                       │ │
│  │  • Call FTIR.fun MCP Server → analyze_ftir_spectrum            │ │
│  │  • 130,000+ reference spectra library                          │ │
│  │  • Returns ranked matches + peak explanations + DOI citations  │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │  Step 3: Chemical Reasoning (AI verification)                  │ │
│  │  • Qwen reviews match results for chemical consistency         │ │
│  │  • Checks functional group assignments against IR tables       │ │
│  │  • Detects potential mixtures and anomalous peaks              │ │
│  │  • Computes adjusted confidence score                          │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │  Step 4: Human-in-the-Loop Checkpoint                          │ │
│  │  • Agent presents findings with evidence                       │ │
│  │  • Asks human to: Accept / Request Alternative / Add Context   │ │
│  │  • This is a REGULATED INDUSTRY requirement                    │ │
│  │  • Agent does NOT proceed without human confirmation           │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │  Step 5: Report Generation (after human confirmation)          │ │
│  │  • Structured Markdown + JSON report                           │ │
│  │  • DOI-cited evidence for each peak assignment                 │ │
│  │  • PCA quality control notes                                   │ │
│  │  • Human analyst confirmation stamp                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘

                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   FTIR.FUN MCP SERVER (External)                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Tools:                                                        │ │
│  │  • analyze_ftir_spectrum — Full-spectrum + peak search         │ │
│  │  • search — Public result search                               │ │
│  │  • fetch — Retrieve specific result by ID                      │ │
│  │                                                                │ │
│  │  Backend:                                                      │ │
│  │  • 130K+ IR reference spectra (SQLite speclib.db)              │ │
│  │  • Chemical knowledge graph (IrEntity, peak-group evidence)    │ │
│  │  • LLM RAG for qualitative reasoning                           │ │
│  │  • 28+ spectrum file format parsers                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Layer | Technology | Provider |
|-------|-----------|----------|
| LLM Reasoning | Qwen-Max via dashscope SDK | **Alibaba Cloud** ModelStudio |
| Web Server | FastAPI + Uvicorn | Local / any cloud |
| Agent Framework | Custom Agent loop (Python) | Self-built |
| External Tools | MCP (Model Context Protocol) | FTIR.fun MCP Server |
| Spectral Database | SQLite speclib.db (130K spectra) | FTIR.fun |
| Storage (optional) | OSS | **Alibaba Cloud** |

## Why This Architecture

1. **Separation of concerns**: Qwen handles reasoning; FTIR.fun handles domain-specific spectral matching. Neither needs to know the other's internals.

2. **MCP as standard protocol**: Using MCP means the agent can swap in any MCP-compatible tool server — not just FTIR.fun.

3. **Human-in-the-loop by design**: In regulated industries (pharma, forensics), AI cannot make final decisions. The agent deliberately pauses at critical checkpoints.

4. **Alibaba Cloud via dashscope**: Qwen-Max via dashscope SDK is the core Alibaba Cloud dependency. The FastAPI server runs locally; Alibaba Cloud proof is the API integration itself.

## Data Flow

1. User uploads spectrum file → FastAPI receives multipart form
2. File bytes + context → Agent.parse_intent() → Qwen extracts structured data
3. base64(file) + parameters → MCP Client → FTIR.fun MCP Server
4. Spectral matches ← MCP response → Agent.verify_results() → Qwen reviews
5. Confirmation prompt → User → accepts/rejects → Agent.generate_report()
6. Final report → JSON response → rendered in Web UI

## Error Handling

- **MCP timeout**: Retry once, then return partial results with warning
- **Qwen API error**: Fall back to basic regex-based intent parsing
- **No spectral matches**: Agent explicitly reports "No matches found" (not a fabricated answer)
- **Low confidence (<0.8)**: Automatically flagged for human review regardless of top match score
