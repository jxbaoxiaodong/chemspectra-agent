"""
ChemSpectra Agent — FastAPI Web Server.
Deployed on Alibaba Cloud ECS.

Provides REST API and simple Web UI for the FTIR spectral analysis agent.
"""

from __future__ import annotations

import base64
import json
import logging
import os

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent import ChemSpectraAgent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChemSpectra Agent",
    description="AI Autopilot for FTIR Spectral Analysis — Qwen Cloud Hackathon",
    version="1.0.0",
)

agent = ChemSpectraAgent()

# ── Web UI ──
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChemSpectra Agent — FTIR Analysis Autopilot</title>
<style>
:root{--bg:#0a0e17;--blue:#42c8f0;--text:#e6edf7;--gray:#8b95a5;--card:rgba(20,28,45,.85)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;align-items:center}
header{text-align:center;padding:40px 20px 20px}
h1{font-size:28px;background:linear-gradient(135deg,#42c8f0,#78d8f8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:var(--gray);font-size:14px;margin-top:8px}
.container{width:100%;max-width:720px;padding:0 20px 40px}
.card{background:var(--card);border:1px solid rgba(66,200,240,.12);border-radius:10px;padding:24px;margin-bottom:16px}
.card h2{font-size:16px;color:var(--blue);margin-bottom:12px}
label{display:block;font-size:13px;color:var(--gray);margin-bottom:4px}
input,textarea,select{width:100%;padding:10px 12px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.1);border-radius:6px;color:var(--text);font-size:14px;margin-bottom:12px}
button{background:linear-gradient(135deg,#42c8f0,#2a90b8);color:#fff;border:none;padding:12px 32px;border-radius:8px;font-size:15px;font-weight:600;cursor:pointer;width:100%}
button:hover{opacity:.9}
#result{white-space:pre-wrap;font-size:13px;line-height:1.7;color:var(--gray)}
#status{text-align:center;padding:12px;color:var(--blue);font-size:13px}
.spinner{display:none;text-align:center;padding:20px}
.step{font-size:12px;color:var(--gray);padding:4px 0}
</style>
</head>
<body>
<header>
<h1>ChemSpectra Agent</h1>
<div class="sub">AI Autopilot for FTIR Spectral Analysis · Powered by Qwen + FTIR.fun</div>
</header>
<div class="container">
<div class="card">
<h2>📤 Upload Spectrum</h2>
<form id="uploadForm" enctype="multipart/form-data">
<label>Spectrum File (.spc .csv .jdx .opus .xlsx …)</label>
<input type="file" name="file" accept=".spc,.csv,.jdx,.opus,.xlsx,.spa,.txt,.json" required>
<label>Sample Description</label>
<textarea name="context" rows="2" placeholder="e.g. Polymer film from production lot #452, suspected PE/PP blend"></textarea>
<label>Analysis Type</label>
<select name="analysis_type">
<option value="identify">Identify unknown material</option>
<option value="qc_check">QC batch consistency check</option>
<option value="deformulate">Deformulate / reverse engineer</option>
<option value="compare">Compare with reference</option>
</select>
<button type="submit">🔬 Analyze Spectrum</button>
</form>
</div>
<div class="spinner" id="spinner"><div class="step">⏳ Agent analyzing spectrum…</div></div>
<div class="card" id="resultCard" style="display:none">
<h2>📊 Analysis Results</h2>
<div id="status"></div>
<div id="result"></div>
<div id="confirmation" style="margin-top:16px">
<button onclick="confirmResult()" style="background:linear-gradient(135deg,#27ae60,#1e8449);margin-bottom:8px">✅ Accept & Generate Report</button>
<button onclick="requestRerun()" style="background:linear-gradient(135deg,#e67e22,#d35400);margin-bottom:8px">🔄 Request Alternative</button>
<button onclick="addContext()">📝 Add More Context</button>
</div>
</div>
</div>
<script>
document.getElementById('uploadForm').addEventListener('submit',async e=>{
e.preventDefault();
document.getElementById('spinner').style.display='block';
document.getElementById('resultCard').style.display='none';
const fd=new FormData(e.target);
try{
const r=await fetch('/api/analyze',{method:'POST',body:fd});
const d=await r.json();
document.getElementById('spinner').style.display='none';
document.getElementById('resultCard').style.display='block';
document.getElementById('result').textContent=d.confirmation_prompt||JSON.stringify(d,null,2);
document.getElementById('status').textContent='Step: '+d.state;
}catch(err){
document.getElementById('spinner').style.display='none';
document.getElementById('result').textContent='Error: '+err.message;
}
});
async function confirmResult(){
document.getElementById('status').textContent='⏳ Generating report…';
const r=await fetch('/api/confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'accept'})});
const d=await r.json();
document.getElementById('result').textContent=d.report||JSON.stringify(d,null,2);
document.getElementById('status').textContent='✅ Report generated';
document.getElementById('confirmation').style.display='none';
}
async function requestRerun(){location.reload();}
function addContext(){prompt('Additional sample context:','');}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


# ── API Endpoints ──

@app.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(None),
    context: str = Form(""),
    analysis_type: str = Form("identify"),
):
    """Run the full spectral analysis pipeline."""
    file_b64 = None
    filename = "spectrum.0"

    if file:
        content = await file.read()
        file_b64 = base64.b64encode(content).decode("ascii")
        filename = file.filename or "spectrum.0"

    user_input = f"Analysis type: {analysis_type}. {context}"

    try:
        result = agent.run_pipeline(
            user_input=user_input,
            file_base64=file_b64,
            filename=filename,
            sample_context=context,
        )
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Analysis failed")
        return JSONResponse(
            {"success": False, "error": str(e)}, status_code=500
        )


@app.post("/api/confirm")
async def confirm(request: Request):
    """Human-in-the-loop: confirm or reject the analysis result."""
    body = await request.json()
    action = body.get("action", "accept")

    if action == "accept":
        report = agent.generate_report()
        return JSONResponse({"success": True, "report": report})
    else:
        agent.state.human_confirmed = False
        return JSONResponse({"success": True, "action": "rejected"})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chemspectra-agent", "cloud": "Alibaba Cloud ECS"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
