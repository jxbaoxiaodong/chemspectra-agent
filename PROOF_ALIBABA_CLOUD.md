# Alibaba Cloud Integration Proof

This document verifies that ChemSpectra Agent uses Alibaba Cloud infrastructure
for all LLM reasoning, as required by the Qwen Cloud Hackathon rules.

## Service Used: Alibaba Cloud ModelStudio (dashscope SDK)

**Model:** `qwen3.7-max` — Alibaba Cloud's latest flagship agent-optimized model  
**API:** `https://dashscope.aliyuncs.com/compatible-mode/v1`  
**SDK:** `dashscope` Python package (official Alibaba Cloud SDK)

### Code Evidence

```python
# agent.py — dashscope SDK import
import dashscope
from dashscope import Generation

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.7-max")

# Qwen call with thinking mode enabled
def _call_qwen(self, messages, thinking=True, **kwargs):
    response = Generation.call(
        api_key=DASHSCOPE_API_KEY,
        model=QWEN_MODEL,           # qwen3.7-max
        messages=messages,
        result_format="message",
        enable_thinking=thinking,   # Alibaba Cloud thinking mode
        **kwargs,
    )

# Function Calling — Qwen autonomously selects tools
def _call_qwen_with_tools(self, messages):
    response = Generation.call(
        api_key=DASHSCOPE_API_KEY,
        model=QWEN_MODEL,
        messages=messages,
        tools=AGENT_TOOLS,          # 5 FTIR analysis tools
        result_format="message",
    )
```

### Where Qwen is Called (6 call sites)

| Call site | Purpose | Thinking |
|-----------|---------|---------|
| `_call_qwen_with_tools` | ReAct reasoning + tool selection (Function Calling) | No (FC mode) |
| `_call_qwen` in verification round | Autonomous verification after conflict detected | Yes |
| `_call_qwen` in self-repair | Retry with error context after JSON parse failure | Yes |
| `extract_verification` | Structured verdict extraction | Yes |
| `handle_followup` | Answer user follow-up questions | Yes |
| `generate_report` | Final professional report generation | Yes |

All 6 call sites go through `dashscope` → Alibaba Cloud infrastructure.

## Live Verification

```bash
# Test model availability
curl -s https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.7-max","messages":[{"role":"user","content":"ping"}],"max_tokens":5}'
```

## Health Check

```bash
python server.py
curl http://localhost:8080/health
```

```json
{"status": "ok", "service": "chemspectra-agent", "alibaba_cloud": "dashscope SDK (qwen3.7-max)"}
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DASHSCOPE_API_KEY` | Alibaba Cloud ModelStudio API key |
| `QWEN_MODEL` | Model name (default: `qwen3.7-max`) |
| `FTIRFUN_API_KEY` | FTIR.fun spectral library API key |
| `FTIRFUN_API_URL` | FTIR.fun API base URL (default: `http://127.0.0.1:18080`) |
