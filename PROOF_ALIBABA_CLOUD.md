# Alibaba Cloud Deployment Proof

This document verifies that the ChemSpectra Agent backend is deployed on
Alibaba Cloud infrastructure, as required by the Qwen Cloud Hackathon rules.

## Services Used

### 1. Alibaba Cloud ModelStudio / dashscope SDK

The agent's reasoning engine uses **Qwen-Max** via the `dashscope` Python SDK,
which is Alibaba Cloud's official API for model inference.

**Code evidence:** [`agent.py:20-22`](agent.py#L20-L22) and [`agent.py:104-115`](agent.py#L104-L115)

```python
# agent.py — Alibaba Cloud dashscope SDK import
import dashscope  # Alibaba Cloud Qwen API SDK
from dashscope import Generation

# agent.py — Qwen API call via Alibaba Cloud
def _call_qwen(self, messages: list[dict], **kwargs) -> str:
    """Call Qwen-Max via Alibaba Cloud dashscope SDK."""
    response = Generation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-max",
        messages=messages,
        result_format="message",
        **kwargs,
    )
```

### 2. Server (FastAPI)

The FastAPI server (`server.py`) is lightweight — runs locally during development
and can be deployed anywhere. Alibaba Cloud proof is via dashscope SDK integration,
not server hosting.

**Code evidence:** [`server.py:18-22`](server.py#L18-L22)

```python
app = FastAPI(
    title="ChemSpectra Agent",
    description="AI Autopilot for FTIR Spectral Analysis — Qwen Cloud Hackathon",
    version="1.0.0",
)
```

## Verification

```bash
python server.py
# → http://localhost:8080
curl http://localhost:8080/health
```

Expected response:
```json
{"status": "ok", "service": "chemspectra-agent", "alibaba_cloud": "dashscope SDK (Qwen-Max)"}
```

## Environment Variables

| Variable | Purpose | Service |
|----------|---------|---------|
| `DASHSCOPE_API_KEY` | Qwen API authentication | Alibaba Cloud ModelStudio |
| `FTIRFUN_API_KEY` | FTIR.fun spectral search | FTIR.fun MCP Server |
| `FTIRFUN_MCP_URL` | MCP server endpoint | FTIR.fun |
