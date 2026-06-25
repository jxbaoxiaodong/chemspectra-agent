"""
ChemSpectra Agent — FastAPI Web Server.

Provides REST API and Web UI for the FTIR spectral analysis agent.
Alibaba Cloud integration: dashscope SDK (qwen3.7-max) via API calls.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.responses import StreamingResponse

from agent import ChemSpectraAgent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChemSpectra Agent",
    description="AI Autopilot for FTIR Spectral Analysis — Qwen Cloud Hackathon",
    version="1.0.0",
)

agent = ChemSpectraAgent()


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(None),
    context: str = Form(""),
    peaks: str = Form(""),
    analysis_type: str = Form("identify"),
):
    """Run the analysis pipeline up to the human confirmation checkpoint."""
    file_b64 = None
    filename = "spectrum.0"

    if file and file.filename:
        content = await file.read()
        if len(content) > 0:
            file_b64 = base64.b64encode(content).decode("ascii")
            filename = file.filename

    peak_list = None
    if peaks and peaks.strip():
        try:
            peak_list = [float(p.strip()) for p in peaks.split(",") if p.strip()]
        except ValueError:
            return JSONResponse({"error": "Invalid peak format. Use comma-separated numbers."}, status_code=400)

    if not file_b64 and not peak_list:
        return JSONResponse({"error": "Please upload a spectrum file or enter peak positions."}, status_code=400)

    user_input = f"{analysis_type}: {context}" if context else analysis_type
    session = agent.new_session()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent.run_pipeline(
                session,
                user_input=user_input,
                file_base64=file_b64,
                filename=filename,
                peaks=peak_list,
                sample_context=context,
            ),
        )
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Analysis failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/analyze/stream")
async def analyze_stream(
    request: Request,
    file: UploadFile | None = File(None),
    context: str = Form(""),
    peaks: str = Form(""),
    analysis_type: str = Form("identify"),
):
    """流式分析——SSE 实时推送 Qwen 思考链、工具调用、结果。"""
    file_b64 = None
    filename = "spectrum.0"

    if file and file.filename:
        content = await file.read()
        if len(content) > 0:
            file_b64 = base64.b64encode(content).decode("ascii")
            filename = file.filename

    peak_list = None
    if peaks and peaks.strip():
        try:
            peak_list = [float(p.strip()) for p in peaks.split(",") if p.strip()]
        except ValueError:
            async def err():
                yield f"data: {json.dumps({'type':'error','data':{'message':'Invalid peak format'}})}\n\n"
            return StreamingResponse(err(), media_type="text/event-stream")

    if not file_b64 and not peak_list:
        async def err():
            yield f"data: {json.dumps({'type':'error','data':{'message':'No spectrum provided'}})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    user_input = f"{analysis_type}: {context}" if context else analysis_type
    session = agent.new_session()

    # 在后台线程运行 pipeline（dashscope SDK 是同步的），SSE 从 event_queue 读取
    def run_pipeline_thread():
        try:
            result = agent.run_pipeline(
                session,
                user_input=user_input,
                file_base64=file_b64,
                filename=filename,
                peaks=peak_list,
                sample_context=context,
            )
            session.event_queue.put({"type": "done", "data": result})
        except Exception as e:
            session.event_queue.put({"type": "error", "data": {"message": str(e)}})

    thread = threading.Thread(target=run_pipeline_thread, daemon=True)
    thread.start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                event = await loop.run_in_executor(
                    None, lambda: session.event_queue.get(timeout=120)
                )
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in ("done", "error"):
                    break
            except Exception:
                yield f"data: {json.dumps({'type':'error','data':{'message':'timeout'}})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/followup")
async def followup(request: Request):
    """Handle user follow-up questions during the confirmation checkpoint."""
    body = await request.json()
    session_id = body.get("session_id")
    message = body.get("message", "")

    if not session_id or not message:
        return JSONResponse({"error": "session_id and message are required"}, status_code=400)

    session = agent.get_session(session_id)
    if not session:
        return JSONResponse({"error": "Session not found. Please start a new analysis."}, status_code=404)

    try:
        result = agent.handle_followup(session, message)
        return JSONResponse({
            "success": True,
            "response": result.get("response", ""),
            "action": result.get("action", "none"),
        })
    except Exception as e:
        logger.exception("Followup failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/confirm")
async def confirm(request: Request):
    """Human-in-the-loop: confirm and generate the final report."""
    body = await request.json()
    session_id = body.get("session_id")
    action = body.get("action", "accept")

    if not session_id:
        return JSONResponse({"error": "session_id is required"}, status_code=400)

    session = agent.get_session(session_id)
    if not session:
        return JSONResponse({"error": "Session not found."}, status_code=404)

    if action == "accept":
        try:
            report = agent.generate_report(session)
            return JSONResponse({"success": True, "report": report, "session_id": session_id})
        except Exception as e:
            logger.exception("Report generation failed")
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    else:
        session.human_confirmed = False
        return JSONResponse({"success": True, "action": "rejected"})


@app.get("/api/report/{session_id}")
async def download_report(session_id: str, fmt: str = "md"):
    """Download the analysis report as Markdown."""
    session = agent.get_session(session_id)
    if not session or not session.final_report:
        return JSONResponse({"error": "No report found."}, status_code=404)

    content = session.final_report
    return Response(
        content=content.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="ChemSpectra_Report_{session_id}.md"'},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "chemspectra-agent",
        "alibaba_cloud": "dashscope SDK (qwen3.7-max)",
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChemSpectra Agent — FTIR Analysis Autopilot</title>
<style>
:root{--bg:#0a0e17;--surface:#111827;--border:rgba(66,200,240,.15);--blue:#42c8f0;--blue2:#2a90b8;
--green:#27ae60;--orange:#e67e22;--red:#e74c3c;--text:#e6edf7;--dim:#8b95a5;--card:rgba(17,24,39,.92)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Inter","PingFang SC","Microsoft YaHei",system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.app{max-width:960px;margin:0 auto;padding:24px 20px 60px}
header{text-align:center;padding:32px 0 24px}
h1{font-size:26px;font-weight:700;background:linear-gradient(135deg,#42c8f0,#78d8f8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.tagline{color:var(--dim);font-size:13px;margin-top:6px}
.badge{display:inline-block;background:rgba(66,200,240,.12);color:var(--blue);font-size:11px;padding:3px 10px;border-radius:20px;margin-top:8px}

.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin-bottom:16px}
.card h2{font-size:15px;color:var(--blue);margin-bottom:14px;font-weight:600}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
input[type=file],textarea,select{width:100%;padding:10px 12px;background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.08);border-radius:6px;color:var(--text);font-size:14px;margin-bottom:14px;transition:border .2s}
input[type=file]:focus,textarea:focus,select:focus{outline:none;border-color:var(--blue)}
textarea{resize:vertical;min-height:56px}
select{cursor:pointer}
.btn{border:none;padding:12px 28px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:all .15s;width:100%}
.btn-primary{background:linear-gradient(135deg,var(--blue),var(--blue2));color:#fff}
.btn-success{background:linear-gradient(135deg,var(--green),#1e8449);color:#fff}
.btn-warn{background:linear-gradient(135deg,var(--orange),#d35400);color:#fff}
.btn-sm{padding:8px 16px;font-size:13px;width:auto}
.btn:hover{opacity:.88;transform:translateY(-1px)}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none}

.pipeline{margin:20px 0}
.step{display:flex;align-items:center;gap:12px;padding:10px 16px;border-left:3px solid rgba(255,255,255,.06);margin-left:12px;font-size:13px;color:var(--dim);transition:all .3s}
.step.active{border-color:var(--blue);color:var(--blue)}
.step.done{border-color:var(--green);color:var(--green)}
.step.error{border-color:var(--red);color:var(--red)}
.step-icon{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0}
.step.pending .step-icon{background:rgba(255,255,255,.06)}
.step.active .step-icon{background:rgba(66,200,240,.15);animation:pulse 1.5s infinite}
.step.done .step-icon{background:rgba(39,174,96,.15)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

.result-card{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.result-header{padding:16px 20px;border-bottom:1px solid var(--border)}
.result-body{padding:20px}
.match-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.match-row:last-child{border:none}
.match-name{font-weight:500;font-size:14px}
.match-cas{color:var(--dim);font-size:12px}
.match-score{font-size:13px;font-weight:600;color:var(--blue)}

.verdict{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;text-transform:uppercase}
.verdict-confirmed{background:rgba(39,174,96,.15);color:var(--green)}
.verdict-needs_review{background:rgba(230,126,34,.15);color:var(--orange)}
.verdict-rejected{background:rgba(231,76,60,.15);color:var(--red)}
.verdict-no_results{background:rgba(231,76,60,.15);color:var(--red)}

.reasoning{background:rgba(0,0,0,.2);border-radius:8px;padding:14px 16px;margin:12px 0;font-size:13px;line-height:1.7;color:var(--dim)}

/* ── Thinking Panel ── */
.thinking-panel{background:rgba(0,0,0,.35);border:1px solid rgba(66,200,240,.2);border-radius:10px;margin:16px 0;overflow:hidden}
.thinking-header{display:flex;align-items:center;gap:8px;padding:10px 16px;border-bottom:1px solid rgba(66,200,240,.1);font-size:12px;color:var(--blue);font-weight:600;letter-spacing:.5px;text-transform:uppercase}
.thinking-dot{width:7px;height:7px;border-radius:50%;background:var(--blue);animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.thinking-body{padding:12px 16px;max-height:220px;overflow-y:auto;font-family:"JetBrains Mono","Courier New",monospace;font-size:11px;line-height:1.7;color:#6ee7fa;white-space:pre-wrap;word-break:break-word}
.tool-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(39,174,96,.12);color:#27ae60;border:1px solid rgba(39,174,96,.25);padding:4px 10px;border-radius:6px;font-size:11px;font-weight:600;margin:3px 3px 3px 0}
.tool-badge.verify{background:rgba(230,126,34,.12);color:var(--orange);border-color:rgba(230,126,34,.25)}
.verify-banner{background:rgba(230,126,34,.08);border:1px solid rgba(230,126,34,.3);border-radius:8px;padding:10px 14px;margin:8px 0;font-size:12px;color:var(--orange)}
.verify-done{background:rgba(39,174,96,.08);border:1px solid rgba(39,174,96,.3);border-radius:8px;padding:10px 14px;margin:8px 0;font-size:12px;color:var(--green)}

.chat-box{margin-top:16px;border-top:1px solid var(--border);padding-top:16px}
.chat-messages{max-height:200px;overflow-y:auto;margin-bottom:10px}
.chat-msg{padding:6px 12px;margin:4px 0;border-radius:8px;font-size:13px}
.chat-msg.user{background:rgba(66,200,240,.1);text-align:right}
.chat-msg.agent{background:rgba(255,255,255,.05)}
.chat-input{display:flex;gap:8px}
.chat-input input{flex:1;padding:8px 12px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.1);border-radius:6px;color:var(--text);font-size:13px}

.actions{display:flex;gap:10px;margin-top:16px}
.actions .btn{flex:1}

.report{background:rgba(0,0,0,.25);border-radius:8px;padding:20px;font-size:13px;line-height:1.8;white-space:pre-wrap;color:var(--dim)}

.hidden{display:none}
.fade-in{animation:fadeIn .3s ease-in}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

footer{text-align:center;padding:24px;color:var(--dim);font-size:11px}
footer a{color:var(--blue);text-decoration:none}
</style>
</head>
<body>
<div class="app">

<header>
  <h1>ChemSpectra Agent</h1>
  <div class="tagline">AI Autopilot for FTIR Spectral Analysis</div>
  <div class="badge">Powered by Qwen-3.7-Max + FTIR.fun (130K spectra)</div>
</header>

<!-- Upload Card -->
<div class="card" id="uploadCard">
  <h2>Upload Spectrum</h2>
  <form id="uploadForm" enctype="multipart/form-data">
    <label>Spectrum File</label>
    <input type="file" name="file" id="specFile"
           accept=".spc,.csv,.jdx,.opus,.xlsx,.spa,.txt,.json,.dx,.asp,.0,.1,.2,.3,.4,.5">

    <label>Sample Description</label>
    <textarea name="context" id="sampleCtx" rows="2"
              placeholder="e.g. Polymer film from production line, suspected PE/PP blend"></textarea>

    <label>Or Enter Peak Positions (cm-1, comma-separated)</label>
    <textarea name="peaks" id="peakInput" rows="1"
              placeholder="e.g. 2920, 2850, 1460, 720"></textarea>

    <label>Analysis Type</label>
    <select name="analysis_type" id="analysisType">
      <option value="identify">Identify unknown material</option>
      <option value="qc_check">QC batch consistency check</option>
      <option value="deformulate">Deformulate / reverse engineer</option>
      <option value="compare">Compare with reference</option>
    </select>

    <button type="submit" class="btn btn-primary" id="submitBtn">Analyze Spectrum</button>
  </form>
</div>

<!-- Pipeline Progress -->
<div class="pipeline hidden" id="pipeline">
  <div class="step pending" id="s1"><span class="step-icon">1</span> Qwen reasoning: selecting analysis tools...</div>
  <div class="step pending" id="s2"><span class="step-icon">2</span> Executing FTIR.fun API tools...</div>
  <div class="step pending" id="s3"><span class="step-icon">3</span> Synthesizing multi-tool results...</div>
  <div class="step pending" id="s4"><span class="step-icon">4</span> Awaiting human confirmation</div>
  <div class="step pending" id="s5"><span class="step-icon">5</span> Generating analysis report</div>
</div>

<!-- Thinking Panel -->
<div class="thinking-panel hidden" id="thinkingPanel">
  <div class="thinking-header">
    <span class="thinking-dot" id="thinkingDot"></span>
    <span id="thinkingLabel">Qwen-3.7-Max is reasoning...</span>
  </div>
  <div id="thinkingActivity" style="padding:8px 16px;font-size:12px;color:var(--dim);border-bottom:1px solid rgba(66,200,240,.08)"></div>
  <div class="thinking-body" id="thinkingBody"></div>
</div>

<!-- Results Card -->
<div class="hidden" id="resultsSection">
  <div class="result-card fade-in">
    <div class="result-header">
      <h2 style="margin:0;color:var(--blue)">Analysis Results</h2>
      <div style="margin-top:6px">
        <span class="verdict" id="verdictBadge"></span>
        <span style="color:var(--dim);font-size:12px;margin-left:8px" id="searchMode"></span>
      </div>
    </div>
    <div class="result-body">
      <div id="bestMatch" style="margin-bottom:16px"></div>
      <div class="reasoning" id="reasoningText"></div>

      <h3 style="font-size:13px;color:var(--dim);margin:16px 0 8px">Top Candidates</h3>
      <div id="candidateList"></div>

      <div id="toolsSection" class="hidden" style="margin-top:12px">
        <h3 style="font-size:13px;color:var(--blue);margin-bottom:6px">Tools Used by Agent</h3>
        <div id="toolsList" style="font-size:12px;color:var(--dim);display:flex;gap:6px;flex-wrap:wrap"></div>
      </div>

      <div id="synthesisSection" class="hidden" style="margin-top:12px">
        <h3 style="font-size:13px;color:var(--dim);margin-bottom:6px">Agent Synthesis</h3>
        <div class="reasoning" id="synthesisText"></div>
      </div>

      <div id="flagsSection" class="hidden" style="margin-top:12px">
        <h3 style="font-size:13px;color:var(--orange);margin-bottom:6px">Flags</h3>
        <ul id="flagsList" style="font-size:12px;color:var(--dim);padding-left:20px"></ul>
      </div>

      <!-- Chat / Follow-up -->
      <div class="chat-box" id="chatBox">
        <div class="chat-messages" id="chatMessages"></div>
        <div class="chat-input">
          <input type="text" id="chatInput" placeholder="Ask a question about the results..." onkeydown="if(event.key==='Enter')sendChat()">
          <button class="btn btn-primary btn-sm" onclick="sendChat()">Send</button>
        </div>
      </div>

      <div class="actions" id="confirmActions">
        <button class="btn btn-success" onclick="confirmResult()">Accept & Generate Report</button>
        <button class="btn btn-warn" onclick="resetForm()">New Analysis</button>
      </div>
    </div>
  </div>
</div>

<!-- Report Section -->
<div class="hidden" id="reportSection">
  <div class="card fade-in">
    <h2>Analysis Report</h2>
    <div class="report" id="reportContent"></div>
    <div style="margin-top:16px;display:flex;gap:10px">
      <button class="btn btn-primary" style="flex:1" onclick="downloadReport()">Download Report (.md)</button>
      <button class="btn btn-warn" style="flex:1" onclick="resetForm()">New Analysis</button>
    </div>
  </div>
</div>

<footer>
  <a href="https://github.com/jxbaoxiaodong/chemspectra-agent" target="_blank">GitHub</a>
  &nbsp;&middot;&nbsp; Qwen Cloud Hackathon Track 4
  &nbsp;&middot;&nbsp; <a href="https://ftir.fun" target="_blank">FTIR.fun</a>
</footer>

</div>

<script>
const $ = id => document.getElementById(id);
let currentSessionId = null;

function setStep(n, status) {
  for (let i = 1; i <= 5; i++) {
    const el = $('s' + i);
    el.className = 'step ' + (i < n ? 'done' : i === n ? status : 'pending');
    el.querySelector('.step-icon').textContent = i < n ? '\\u2713' : i;
  }
}

const TOOL_LABELS = {
  identify_material: 'Material ID',
  explain_peaks: 'Peak Explain',
  assign_functional_groups: 'Func Groups',
  match_library_topk: 'Library Match',
  search_public_results: 'Public Search',
};

function appendThinking(text) {
  const el = $('thinkingBody');
  el.textContent += text;
  el.scrollTop = el.scrollHeight;
}

function setThinkingLabel(label) {
  $('thinkingLabel').textContent = label;
}

function addActivityBadge(tool, phase) {
  const act = $('thinkingActivity');
  const cls = phase === 'verification' ? 'tool-badge verify' : 'tool-badge';
  const label = TOOL_LABELS[tool] || tool;
  act.innerHTML += `<span class="${cls}">&#8594; ${esc(label)}</span>`;
}

function showVerifyBanner(data) {
  const act = $('thinkingActivity');
  act.innerHTML += `<div class="verify-banner">&#9888; ${esc(data.label || 'Verification triggered')}</div>`;
}

function showVerifyDone(data) {
  const act = $('thinkingActivity');
  act.innerHTML += `<div class="verify-done">&#10003; ${esc(data.label || 'Verification complete')}</div>`;
}

$('uploadForm').addEventListener('submit', async e => {
  e.preventDefault();
  $('submitBtn').disabled = true;
  $('pipeline').classList.remove('hidden');
  $('thinkingPanel').classList.remove('hidden');
  $('thinkingBody').textContent = '';
  $('thinkingActivity').innerHTML = '';
  $('resultsSection').classList.add('hidden');
  $('reportSection').classList.add('hidden');
  $('chatMessages').innerHTML = '';
  setStep(1, 'active');
  setThinkingLabel('Qwen-3.7-Max is reasoning...');

  const fd = new FormData(e.target);

  try {
    const resp = await fetch('/api/analyze/stream', {method: 'POST', body: fd});
    if (!resp.ok) {
      throw new Error('HTTP ' + resp.status);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let finalData = null;

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let evt;
        try { evt = JSON.parse(line.slice(6)); } catch { continue; }

        switch (evt.type) {
          case 'phase':
            setThinkingLabel(evt.data.label || evt.data.phase);
            if (evt.data.phase === 'ReAct') setStep(1, 'active');
            else if (evt.data.phase === 'Synthesis') setStep(3, 'active');
            else if (evt.data.phase === 'Verification synthesis') setStep(3, 'active');
            break;
          case 'thinking':
            appendThinking(evt.data.text);
            break;
          case 'synthesis_chunk':
            // synthesis text coming in — just show phase label
            break;
          case 'tool_call':
            setStep(2, 'active');
            addActivityBadge(evt.data.tool, evt.data.phase);
            break;
          case 'tool_result':
            // badge already shown on tool_call
            break;
          case 'verification_triggered':
            showVerifyBanner(evt.data);
            setThinkingLabel('Self-verification in progress...');
            break;
          case 'verification_done':
            showVerifyDone(evt.data);
            break;
          case 'done':
            finalData = evt.data;
            break;
          case 'error':
            throw new Error(evt.data.message || 'Stream error');
        }
      }
      if (finalData) break;
    }

    if (!finalData) throw new Error('Stream ended without result');

    currentSessionId = finalData.session_id;
    $('thinkingDot').style.animation = 'none';
    $('thinkingDot').style.background = 'var(--green)';
    setThinkingLabel('Analysis complete');

    if (finalData.step === 'needs_clarification') {
      setStep(1, 'done');
      addChatMsg('agent', finalData.question || 'Could you provide more details?');
      $('resultsSection').classList.remove('hidden');
      $('bestMatch').innerHTML = '<div style="color:var(--orange)">Agent needs more information</div>';
      $('reasoningText').textContent = finalData.question || '';
      $('verdictBadge').textContent = 'clarification needed';
      $('verdictBadge').className = 'verdict verdict-needs_review';
      $('candidateList').innerHTML = '';
      $('confirmActions').classList.add('hidden');
    } else {
      setStep(4, 'active');
      renderResults(finalData);
      $('resultsSection').classList.remove('hidden');
      $('confirmActions').classList.remove('hidden');
    }

  } catch (err) {
    setStep(1, 'error');
    alert('Error: ' + err.message);
  }
  $('submitBtn').disabled = false;
});

function renderResults(data) {
  const c = data.confirmation || {};
  const best = c.best_match || {};

  const vb = $('verdictBadge');
  vb.textContent = (c.verdict || 'needs_review').replace(/_/g, ' ');
  vb.className = 'verdict verdict-' + (c.verdict || 'needs_review');

  $('searchMode').textContent = data.search_summary || '';

  const score = (best.score || 0);
  const pct = score < 1 ? (score * 100).toFixed(1) + '%' : score.toFixed(1) + '%';
  $('bestMatch').innerHTML =
    '<div style="font-size:20px;font-weight:700;color:var(--text)">' + esc(best.name || 'Unknown') + '</div>' +
    '<div style="color:var(--dim);font-size:13px;margin-top:4px">CAS: ' + esc(best.cas || 'N/A') +
    ' &middot; Score: <span style="color:var(--blue);font-weight:600">' + pct + '</span></div>';

  $('reasoningText').textContent = c.reasoning || 'Chemical verification completed.';

  const cl = $('candidateList');
  cl.innerHTML = '';
  (c.candidates || []).forEach(m => {
    const s = m.score < 1 ? (m.score * 100).toFixed(1) + '%' : m.score.toFixed(1) + '%';
    cl.innerHTML += '<div class="match-row">' +
      '<div><span class="match-name">' + esc(m.name) + '</span><br><span class="match-cas">CAS: ' + esc(m.cas || 'N/A') + '</span></div>' +
      '<span class="match-score">' + s + '</span></div>';
  });

  const toolsCalled = c.tools_called || data.tools_called || [];
  if (toolsCalled.length > 0) {
    $('toolsSection').classList.remove('hidden');
    const toolLabels = {
      identify_material: 'Material ID',
      explain_peaks: 'Peak Explain',
      assign_functional_groups: 'Func Groups',
      match_library_topk: 'Library Match',
      search_public_results: 'Public Search',
    };
    $('toolsList').innerHTML = toolsCalled.map(t =>
      '<span style="background:rgba(66,200,240,.12);color:var(--blue);padding:3px 10px;border-radius:12px;font-size:11px">' +
      esc(toolLabels[t] || t) + '</span>'
    ).join('');
  } else {
    $('toolsSection').classList.add('hidden');
  }

  const synthesis = c.synthesis || '';
  if (synthesis) {
    $('synthesisSection').classList.remove('hidden');
    $('synthesisText').textContent = synthesis;
  } else {
    $('synthesisSection').classList.add('hidden');
  }

  const flags = c.flags || [];
  if (flags.length > 0) {
    $('flagsSection').classList.remove('hidden');
    $('flagsList').innerHTML = flags.map(f => '<li>' + esc(f) + '</li>').join('');
  } else {
    $('flagsSection').classList.add('hidden');
  }
}

function addChatMsg(role, text) {
  const el = document.createElement('div');
  el.className = 'chat-msg ' + role;
  el.textContent = text;
  $('chatMessages').appendChild(el);
  $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
}

async function sendChat() {
  const input = $('chatInput');
  const msg = input.value.trim();
  if (!msg || !currentSessionId) return;
  input.value = '';
  addChatMsg('user', msg);

  try {
    const r = await fetch('/api/followup', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: currentSessionId, message: msg}),
    });
    const d = await r.json();
    addChatMsg('agent', d.response || d.error || 'No response');
  } catch (err) {
    addChatMsg('agent', 'Error: ' + err.message);
  }
}

async function confirmResult() {
  if (!currentSessionId) return;
  setStep(5, 'active');
  $('confirmActions').innerHTML = '<div style="color:var(--blue);font-size:13px">Generating report...</div>';

  try {
    const r = await fetch('/api/confirm', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: currentSessionId, action: 'accept'}),
    });
    const d = await r.json();
    setStep(5, 'done');
    $('reportContent').textContent = d.report || JSON.stringify(d, null, 2);
    $('reportSection').classList.remove('hidden');
    $('resultsSection').classList.add('hidden');
  } catch (err) {
    setStep(5, 'error');
    alert('Report generation failed: ' + err.message);
  }
}

function downloadReport() {
  if (!currentSessionId) return;
  window.open('/api/report/' + currentSessionId + '?fmt=md', '_blank');
}

function resetForm() {
  currentSessionId = null;
  $('uploadForm').reset();
  $('pipeline').classList.add('hidden');
  $('resultsSection').classList.add('hidden');
  $('reportSection').classList.add('hidden');
  $('chatMessages').innerHTML = '';
  setStep(1, 'pending');
  $('confirmActions').innerHTML =
    '<button class="btn btn-success" onclick="confirmResult()">Accept & Generate Report</button>' +
    '<button class="btn btn-warn" onclick="resetForm()">New Analysis</button>';
  $('confirmActions').classList.remove('hidden');
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
