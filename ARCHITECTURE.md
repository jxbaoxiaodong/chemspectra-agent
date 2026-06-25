# ChemSpectra Agent — Architecture Documentation

## System Overview

ChemSpectra Agent is a **multi-round self-verifying ReAct agent** for FTIR spectral analysis,
built for **Track 4: Autopilot Agent** of the Qwen Cloud Hackathon.

The agent uses **Qwen Function Calling** with autonomous reasoning, evidence cross-validation,
self-verification, and self-repair — not a fixed pipeline.

```
┌──────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Web UI (FastAPI + vanilla HTML/JS)                            │ │
│  │  • File upload (.spc, .csv, .jdx, .opus, .xlsx … 28+ formats) │ │
│  │  • Peak position input (manual cm⁻¹ values)                   │ │
│  │  • Sample context / analysis type selection                   │ │
│  │  • Tool selection visualization (which tools Agent chose)      │ │
│  │  • Human-in-the-loop confirmation + chat follow-up            │ │
│  │  • Report viewing + Markdown download                         │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                    FastAPI (localhost:8080)                          │
└───────────────────────────────┼──────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                                   ▼
┌──────────────────────────┐    ┌──────────────────────────────────────┐
│   QWEN AGENT (dashscope) │    │   FTIR.FUN TOOL SET (5 tools)        │
│  ┌────────────────────┐  │    │                                      │
│  │ ReAct Loop:        │──┼───►│  ① identify_material                 │
│  │ • Reason about     │  │    │     POST /ftir/identify_material      │
│  │   user intent      │  │    │     → Ranked material matches         │
│  │ • Select tools     │  │    │                                      │
│  │   (Function Call)  │  │    │  ② explain_peaks                     │
│  │ • Synthesize       │  │    │     POST /ftir/explain_peaks          │
│  │   multi-tool       │  │    │     → Chemical bond interpretations   │
│  │   results          │  │    │                                      │
│  ├────────────────────┤  │    │  ③ assign_functional_groups           │
│  │ Follow-up Chat     │  │    │     POST /ftir/assign_functional_groups│
│  │ (Qwen-Max)         │  │    │     → Functional group mapping        │
│  ├────────────────────┤  │    │                                      │
│  │ Report Generation  │  │    │  ④ match_library_topk                 │
│  │ (Qwen-Max)         │  │    │     POST /ftir/match_library_topk     │
│  └────────────────────┘  │    │     → Rapid top-K screening           │
│                          │    │                                      │
│                          │    │  ⑤ search_public_results              │
│                          │    │     MCP /mcp (JSON-RPC)               │
│                          │    │     → Public analysis lookup           │
│                          │    └──────────────────────────────────────┘
└──────────────────────────┘

All tools share the same CanonicalFtirRequest schema.
Auth: X-API-Key header (REST) / JSON-RPC (MCP).
```

## Multi-Round ReAct Agent Loop

The core architecture: Qwen autonomously selects tools, then the agent
cross-validates evidence and self-verifies before committing to a conclusion.

```
User request + spectrum data
        │
        ▼
   ┌─────────────────────────────┐
   │  PHASE 1: ReAct Reasoning   │
   │  Qwen reasons about intent  │
   │  and selects tool(s) via    │◄──┐
   │  Function Calling           │   │
   └────────────┬────────────────┘   │
                │ tool_calls         │
                ▼                    │
   ┌─────────────────────────────┐   │
   │  Execute selected tool(s)   │   │
   │  against FTIR.fun API       │   │
   └────────────┬────────────────┘   │
                │ tool_results       │
                ▼                    │
   ┌─────────────────────────────┐   │
   │  More tools needed?         │───┘ Yes → loop back
   │  (Qwen decides)             │
   └────────────┬────────────────┘
                │ No → initial synthesis
                ▼
   ┌─────────────────────────────┐
   │  PHASE 2: Cross-Validation  │
   │  • Estimate confidence      │
   │  • Detect evidence conflicts│
   │    (functional group ↔      │
   │     material mismatch,      │
   │     ambiguous top matches)  │
   └────────────┬────────────────┘
                │
          ┌─────┴─────┐
          │ Issues?    │
          └─────┬─────┘
          Yes   │   No → commit synthesis
                ▼
   ┌─────────────────────────────┐
   │  PHASE 3: Self-Verification │
   │  Agent told about specific  │
   │  conflicts, autonomously    │
   │  calls more tools to verify │
   │  → updated synthesis        │
   └────────────┬────────────────┘
                │
                ▼
   ┌─────────────────────────────┐
   │  PHASE 4: Self-Repair       │
   │  If any LLM output is       │
   │  malformed JSON, error sent  │
   │  back to Qwen with context   │
   │  for automatic retry         │
   └─────────────────────────────┘
```

### Tool Selection Examples

| User Intent | Tools Selected | Why |
|-------------|---------------|-----|
| "Identify this material" | identify_material + explain_peaks | Match library + explain supporting peaks |
| "What are these peaks?" | explain_peaks + assign_functional_groups | Peak interpretation + structural info |
| "Deformulate this sample" | identify_material + explain_peaks + assign_functional_groups | Full reverse engineering |
| "Quick screening" | match_library_topk | Fast top-K matches only |
| "QC batch check" | identify_material + match_library_topk | Confirm identity + check consistency |

## Concurrency Model

Each analysis request creates an independent `Session` object. Multiple users can
analyze concurrently without state corruption — sessions are identified by UUID and
stored in-memory on the server.

## Technology Stack

| Layer | Technology | Provider |
|-------|-----------|----------|
| LLM Reasoning + Tool Selection | Qwen-Max via dashscope SDK (Function Calling) | **Alibaba Cloud** ModelStudio |
| Web Server | FastAPI + Uvicorn | Local / any cloud |
| Agent Framework | Custom ReAct agent with multi-tool routing (Python) | Self-built |
| Spectral Analysis (5 tools) | REST API + MCP (X-API-Key auth) | FTIR.fun |
| Spectral Database | SQLite speclib.db (130K+ spectra) | FTIR.fun |

## Key Design Decisions

1. **Multi-round ReAct with self-verification** — The agent doesn't just pick tools and
   synthesize. After the initial ReAct loop, it estimates confidence and checks for evidence
   conflicts. Low confidence or contradictions trigger an autonomous verification round
   where the agent calls additional tools to resolve the issues.

2. **Evidence cross-validation** — When multiple tools return results, the agent checks
   for logical consistency (e.g., if identify_material says "PET" but functional groups
   don't include ester groups, that's a conflict). This catches errors that single-pass
   synthesis would miss.

3. **Self-repair** — When Qwen's structured JSON output fails to parse, the error message
   and original output are sent back to Qwen for retry. The LLM sees exactly what went
   wrong and fixes it — no silent failures.

4. **Confidence-aware routing** — The agent tracks confidence through the analysis pipeline.
   High confidence (>0.75) → fast path. Low confidence → deep verification with additional
   tool calls. Confidence trace is logged for transparency.

5. **Qwen Function Calling** — Tools are defined as structured function schemas, and Qwen
   decides which to invoke. This is real autonomous decision-making, not hard-coded routing.

6. **Human-in-the-loop by design** — In regulated industries (pharma, forensics, materials QC),
   AI cannot make final decisions alone. The agent deliberately pauses and requires
   explicit human confirmation. Users can also ask follow-up questions before confirming.

7. **Session isolation** — Each request gets its own Session with independent state.
   No global mutable state means concurrent requests never interfere.

8. **Alibaba Cloud via dashscope** — Qwen-Max via dashscope SDK is the core Alibaba Cloud
   dependency. All LLM reasoning (tool selection, synthesis, verification, self-repair,
   follow-up chat, report generation) goes through this single provider.

## Quantitative Benefits

| Metric | Single-Pass Agent | With Self-Verification |
|--------|------------------|----------------------|
| Confidence calibration | None | Tracked per-step |
| Evidence conflict detection | None | Automated (functional group ↔ material) |
| Malformed output handling | Silent failure | Self-repair with error context |
| Analysis depth (LLM calls) | 1-2 | 3-6 (adaptive) |
| Ambiguous result handling | Accept top match | Autonomous investigation |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| POST | `/api/analyze` | Run multi-tool analysis pipeline (returns session_id) |
| POST | `/api/followup` | Ask follow-up questions about results |
| POST | `/api/confirm` | Accept results and generate report |
| GET | `/api/report/{session_id}` | Download report as Markdown |
| GET | `/health` | Health check |

## Error Handling

- **API timeout**: FTIR.fun search has a 120s timeout; returns error message on failure
- **Qwen API error**: Raises RuntimeError with status code and message
- **Tool execution error**: Logged and returned to Qwen for synthesis (agent may try alternative tools)
- **No spectral matches**: Agent explicitly reports "No matches found" (never fabricates)
- **Max iterations**: ReAct loop capped at 6 iterations to prevent infinite loops
- **Invalid input**: Returns 400 with clear error message
- **Session not found**: Returns 404 (sessions are in-memory, lost on restart)
