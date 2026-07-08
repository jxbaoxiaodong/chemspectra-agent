"""
ChemSpectra Agent — FastAPI Web Server.

Provides REST API and Web UI for the FTIR spectral analysis agent.
Alibaba Cloud integration: dashscope SDK (qwen3.7-max) via API calls.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import threading
from datetime import datetime

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from collections import defaultdict
import time as _time

from agent import ChemSpectraAgent
from audit import audit_logger

logger = logging.getLogger(__name__)

import pathlib as _pathlib

# IP-based rate limiter: max 1 analysis per 15s per IP
_rate_limits: dict[str, float] = defaultdict(float)
RATE_LIMIT_SECONDS = 15


def _check_rate_limit(request: Request) -> str | None:
    ip = request.client.host if request.client else "unknown"
    now = _time.time()
    last = _rate_limits[ip]
    if now - last < RATE_LIMIT_SECONDS:
        wait = int(RATE_LIMIT_SECONDS - (now - last)) + 1
        return f"Rate limited. Please wait {wait}s before trying again."
    _rate_limits[ip] = now
    return None
app = FastAPI(
    title="ChemSpectra Agent",
    description="AI Autopilot for FTIR Spectral Analysis — Qwen Cloud Hackathon",
    version="1.0.0",
)
_static_dir = _pathlib.Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
_samples_dir = _pathlib.Path(__file__).parent / "samples"
_samples_dir.mkdir(exist_ok=True)
app.mount("/samples", StaticFiles(directory=str(_samples_dir)), name="samples")

agent = ChemSpectraAgent()

SESSION_LOG_DIR = _pathlib.Path(__file__).parent / "session_logs"
SESSION_LOG_DIR.mkdir(exist_ok=True)


def _request_meta(request: Request) -> dict:
    return {
        "method": request.method,
        "path": request.url.path,
        "query": str(request.url.query),
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", ""),
        "referer": request.headers.get("referer", ""),
    }


def _save_session_log(session, result: dict, *, stage: str, extra: dict | None = None):
    """将完整会话数据保存到 JSON 文件，供赛事材料引用真实数据。"""
    try:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "session_id": session.session_id,
            "user_input": session.user_input,
            "sample_context": session.sample_context,
            "filename": session.filename,
            "file_sha256": session.file_sha256,
            "file_size_bytes": session.file_size_bytes,
            "peaks": session.peaks,
            "tools_called": [t.get("tool") for t in session.tool_calls_log],
            "tool_calls_log": session.tool_calls_log,
            "tool_results": session.tool_results,
            "synthesis": session.synthesis,
            "verification": session.verification,
            "verification_plan": session.verification_plan,
            "confidence_trace": session.confidence_trace,
            "evidence_conflicts": session.evidence_conflicts,
            "react_iterations": session.react_iterations,
            "resolved_level": session.resolved_level,
            "decision_status": session.decision_status,
            "review_required": session.review_required,
            "repair_count": session.repair_count,
            "conversation": session.conversation,
            "final_report": session.final_report,
            "api_result": result,
            "extra": extra or {},
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{ts}_{session.session_id}_{stage}.json"
        path = SESSION_LOG_DIR / fname
        path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        audit_path = audit_logger.write_event(
            category="session_snapshot",
            action=stage,
            session_id=session.session_id,
            payload=log_data,
        )
        logger.info("Session log saved: %s", path)
        logger.info("Audit snapshot saved: %s", audit_path)
    except Exception:
        logger.exception("Failed to save session log")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    audit_logger.write_event(
        category="access",
        action="page_view",
        payload=_request_meta(request),
    )
    index_path = _pathlib.Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return INDEX_HTML  # fallback


@app.get("/api/samples")
async def list_samples(request: Request):
    """Return available demo samples for one-click evaluation."""
    samples = []
    for f in sorted(_samples_dir.iterdir()):
        if f.is_file() and f.suffix in (".csv", ".dx", ".jdx", ".txt", ".json"):
            samples.append({"filename": f.name, "url": f"/samples/{f.name}"})
    audit_logger.write_event(
        category="api",
        action="samples_listed",
        payload={
            "request": _request_meta(request),
            "sample_count": len(samples),
            "samples": samples,
        },
    )
    return JSONResponse(samples)


@app.post("/api/audit/frontend")
async def frontend_audit(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    audit_logger.write_event(
        category="frontend",
        action=body.get("event", "unknown"),
        session_id=session_id,
        payload={
            "request": _request_meta(request),
            "payload": body.get("payload", {}),
        },
    )
    return JSONResponse({"success": True})


@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: UploadFile | None = File(None),
    context: str = Form(""),
    peaks: str = Form(""),
    analysis_type: str = Form("identify"),
):
    """Run the analysis pipeline up to the human confirmation checkpoint."""
    started = _time.perf_counter()
    err = _check_rate_limit(request)
    if err:
        audit_logger.write_event(
            category="api",
            action="analyze_rate_limited",
            payload={
                "request": _request_meta(request),
                "error": err,
            },
        )
        return JSONResponse({"error": err}, status_code=429)
    file_b64 = None
    filename = "spectrum.0"
    file_size_bytes = 0
    file_sha256 = ""

    if file and file.filename:
        content = await file.read()
        if len(content) > 0:
            file_b64 = base64.b64encode(content).decode("ascii")
            filename = file.filename
            file_size_bytes = len(content)
            file_sha256 = hashlib.sha256(content).hexdigest()

    peak_list = None
    if peaks and peaks.strip():
        try:
            peak_list = [float(p.strip()) for p in peaks.split(",") if p.strip()]
        except ValueError:
            return JSONResponse({"error": "Invalid peak format. Use comma-separated numbers."}, status_code=400)

    user_input = f"{analysis_type}: {context}" if context else analysis_type
    session = agent.new_session()
    session.file_size_bytes = file_size_bytes
    session.file_sha256 = file_sha256

    audit_logger.write_event(
        category="api",
        action="analyze_request",
        session_id=session.session_id,
        payload={
            "request": _request_meta(request),
            "analysis_type": analysis_type,
            "context": context,
            "filename": filename,
            "file_size_bytes": file_size_bytes,
            "file_sha256": file_sha256,
            "peak_count": len(peak_list or []),
            "user_input": user_input,
        },
    )

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
        _save_session_log(
            session,
            result,
            stage="analyze",
            extra={
                "request": _request_meta(request),
                "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
            },
        )
        audit_logger.write_event(
            category="api",
            action="analyze_response",
            session_id=session.session_id,
            payload={
                "request": _request_meta(request),
                "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                "response": result,
            },
        )
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Analysis failed")
        audit_logger.write_event(
            category="api",
            action="analyze_error",
            session_id=session.session_id,
            payload={
                "request": _request_meta(request),
                "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                "error": str(e),
            },
        )
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
    started = _time.perf_counter()
    err = _check_rate_limit(request)
    if err:
        audit_logger.write_event(
            category="api",
            action="analyze_stream_rate_limited",
            payload={
                "request": _request_meta(request),
                "error": err,
            },
        )
        return JSONResponse({"error": err}, status_code=429)
    file_b64 = None
    filename = "spectrum.0"
    file_size_bytes = 0
    file_sha256 = ""

    if file and file.filename:
        content = await file.read()
        if len(content) > 0:
            file_b64 = base64.b64encode(content).decode("ascii")
            filename = file.filename
            file_size_bytes = len(content)
            file_sha256 = hashlib.sha256(content).hexdigest()

    peak_list = None
    if peaks and peaks.strip():
        try:
            peak_list = [float(p.strip()) for p in peaks.split(",") if p.strip()]
        except ValueError:
            async def err():
                yield f"data: {json.dumps({'type':'error','data':{'message':'Invalid peak format'}})}\n\n"
            return StreamingResponse(err(), media_type="text/event-stream")

    user_input = f"{analysis_type}: {context}" if context else analysis_type
    session = agent.new_session()
    session.file_size_bytes = file_size_bytes
    session.file_sha256 = file_sha256

    audit_logger.write_event(
        category="api",
        action="analyze_stream_request",
        session_id=session.session_id,
        payload={
            "request": _request_meta(request),
            "analysis_type": analysis_type,
            "context": context,
            "filename": filename,
            "file_size_bytes": file_size_bytes,
            "file_sha256": file_sha256,
            "peak_count": len(peak_list or []),
            "user_input": user_input,
        },
    )

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
            _save_session_log(
                session,
                result,
                stage="analyze_stream",
                extra={
                    "request": _request_meta(request),
                    "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                },
            )
            audit_logger.write_event(
                category="api",
                action="analyze_stream_completed",
                session_id=session.session_id,
                payload={
                    "request": _request_meta(request),
                    "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                    "response": result,
                },
            )
            session.event_queue.put({"type": "done", "data": result})
        except Exception as e:
            audit_logger.write_event(
                category="api",
                action="analyze_stream_error",
                session_id=session.session_id,
                payload={
                    "request": _request_meta(request),
                    "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                    "error": str(e),
                },
            )
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
    started = _time.perf_counter()
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
        _save_session_log(
            session,
            result,
            stage="followup",
            extra={
                "request": _request_meta(request),
                "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                "message": message,
            },
        )
        audit_logger.write_event(
            category="api",
            action="followup_response",
            session_id=session_id,
            payload={
                "request": _request_meta(request),
                "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                "message": message,
                "response": result,
            },
        )
        return JSONResponse({
            "success": True,
            "response": result.get("response", ""),
            "action": result.get("action", "none"),
            "tools_used": result.get("tools_used", []),
        })
    except Exception as e:
        logger.exception("Followup failed")
        audit_logger.write_event(
            category="api",
            action="followup_error",
            session_id=session_id,
            payload={
                "request": _request_meta(request),
                "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                "message": message,
                "error": str(e),
            },
        )
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/confirm")
async def confirm(request: Request):
    """Human-in-the-loop: confirm and generate the final report."""
    started = _time.perf_counter()
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
            _save_session_log(
                session,
                {"success": True, "report_length": len(report)},
                stage="confirm_accept",
                extra={
                    "request": _request_meta(request),
                    "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                },
            )
            audit_logger.write_event(
                category="api",
                action="confirm_accept_response",
                session_id=session_id,
                payload={
                    "request": _request_meta(request),
                    "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                    "report_length": len(report),
                },
            )
            return JSONResponse({"success": True, "report": report, "session_id": session_id})
        except Exception as e:
            logger.exception("Report generation failed")
            audit_logger.write_event(
                category="api",
                action="confirm_accept_error",
                session_id=session_id,
                payload={
                    "request": _request_meta(request),
                    "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
                    "error": str(e),
                },
            )
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    else:
        session.human_confirmed = False
        _save_session_log(
            session,
            {"success": True, "action": "rejected"},
            stage="confirm_reject",
            extra={
                "request": _request_meta(request),
                "duration_ms": round((_time.perf_counter() - started) * 1000, 2),
            },
        )
        return JSONResponse({"success": True, "action": "rejected"})


@app.get("/api/report/{session_id}")
async def download_report(session_id: str, request: Request, fmt: str = "md"):
    """Download the analysis report as Markdown."""
    session = agent.get_session(session_id)
    if not session or not session.final_report:
        return JSONResponse({"error": "No report found."}, status_code=404)

    audit_logger.write_event(
        category="access",
        action="report_download",
        session_id=session_id,
        payload={
            "request": _request_meta(request),
            "format": fmt,
            "report_length": len(session.final_report),
        },
    )
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
:root{--bg:#0a0e14;--surface:rgba(10,14,20,.92);--border:rgba(58,124,165,.15);--blue:#3a7ca5;--blue2:#2c5f80;
--green:#3a7ca5;--orange:#f59e0b;--red:#ef4444;--text:#e8ecf0;--dim:#8a9baa;--card:rgba(10,14,20,.75);
--glow:rgba(58,124,165,.08);--radius:14px}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{font-family:"Inter","PingFang SC","Microsoft YaHei",system-ui,sans-serif;background:var(--bg);color:var(--text)}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(58,124,165,.04),transparent),radial-gradient(ellipse 60% 40% at 80% 100%,rgba(34,197,94,.03),transparent);pointer-events:none}

.layout{display:grid;grid-template-columns:420px 1fr;grid-template-rows:auto 1fr;height:100vh;gap:0}
.top-bar{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;padding:12px 28px;border-bottom:1px solid var(--border);background:var(--surface);backdrop-filter:blur(16px)}
.top-bar h1{font-size:20px;font-weight:700;color:#3a7ca5}
.top-bar .meta{display:flex;align-items:center;gap:12px;font-size:11px;color:var(--dim)}
.top-bar .badge{background:rgba(58,124,165,.1);color:var(--blue);padding:4px 12px;border-radius:20px;font-weight:600}

.left-panel{overflow-y:auto;padding:20px;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:16px}
.right-panel{overflow:hidden;display:flex;flex-direction:column;background:rgba(3,7,18,.6)}

.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;backdrop-filter:blur(12px)}
.card h2{font-size:13px;color:var(--blue);margin-bottom:12px;font-weight:700;text-transform:uppercase;letter-spacing:.8px}
.card p{font-size:12px;color:var(--dim);line-height:1.5;margin-bottom:12px}

label{display:block;font-size:11px;color:var(--dim);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px;font-weight:600}
input[type=file],textarea,select{width:100%;padding:10px 12px;background:rgba(0,0,0,.4);border:1px solid rgba(255,255,255,.06);border-radius:8px;color:var(--text);font-size:13px;margin-bottom:12px;transition:border .2s}
input[type=file]:focus,textarea:focus,select:focus{outline:none;border-color:var(--blue)}
textarea{resize:vertical;min-height:48px}
select{cursor:pointer}

.btn{border:none;padding:11px 24px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all .15s;width:100%}
.btn-primary{background:#3a7ca5;color:#fff;box-shadow:0 4px 16px rgba(58,124,165,.25)}
.btn-success{background:#3a7ca5;color:#fff}
.btn-warn{background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.3);color:var(--orange)}
.btn-sm{padding:8px 14px;font-size:12px;width:auto}
.btn:hover{opacity:.9;transform:translateY(-1px)}
.btn:disabled{opacity:.35;cursor:not-allowed;transform:none}

.sample-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.btn-sample{display:flex;align-items:center;gap:10px;padding:12px 14px;border-radius:10px;background:rgba(58,124,165,.04);border:1px solid rgba(58,124,165,.15);color:var(--text);cursor:pointer;transition:all .2s;width:100%;text-align:left}
.btn-sample:hover{background:rgba(58,124,165,.12);border-color:var(--blue);transform:translateY(-1px);box-shadow:0 6px 20px rgba(58,124,165,.1)}
.btn-sample .sample-icon{font-size:20px;flex-shrink:0}
.btn-sample .sample-info{display:flex;flex-direction:column;gap:2px;overflow:hidden}
.btn-sample .sample-name{font-size:12px;font-weight:600;color:var(--text)}
.btn-sample .sample-file{font-size:10px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

.pipeline{margin:0;padding:0}
.step{display:flex;align-items:center;gap:10px;padding:8px 12px;border-left:2px solid rgba(255,255,255,.06);margin-left:8px;font-size:12px;color:var(--dim);transition:all .3s}
.step.active{border-color:var(--blue);color:var(--blue)}
.step.done{border-color:var(--green);color:var(--green)}
.step.error{border-color:var(--red);color:var(--red)}
.step-icon{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0;font-weight:700}
.step.pending .step-icon{background:rgba(255,255,255,.06)}
.step.active .step-icon{background:rgba(58,124,165,.15);animation:pulse 1.5s infinite}
.step.done .step-icon{background:rgba(34,197,94,.15)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* === Right Panel: Thinking Stream === */
.thinking-header{display:flex;align-items:center;gap:10px;padding:14px 24px;border-bottom:1px solid var(--border);background:rgba(6,9,15,.8);flex-shrink:0}
.thinking-header .label{font-size:12px;color:var(--blue);font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.thinking-header .status{font-size:11px;color:var(--dim);margin-left:auto}
.thinking-dot{width:8px;height:8px;border-radius:50%;background:var(--blue);animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.thinking-activity{padding:10px 24px;font-size:11px;color:var(--dim);border-bottom:1px solid rgba(58,124,165,.06);display:flex;gap:6px;flex-wrap:wrap;flex-shrink:0}
.thinking-body{flex:1;overflow-y:auto;padding:16px 24px;font-family:"JetBrains Mono","SF Mono","Cascadia Code",monospace;font-size:12.5px;line-height:1.7;color:#c8dbe6;background:linear-gradient(180deg,rgba(5,8,12,.3),rgba(5,8,12,.6))}
.thinking-body::-webkit-scrollbar{width:6px}
.thinking-body::-webkit-scrollbar-track{background:transparent}
.thinking-body::-webkit-scrollbar-thumb{background:rgba(58,124,165,.2);border-radius:3px}
.log-line{display:grid;grid-template-columns:52px 72px 1fr;gap:8px;align-items:start;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.03)}
.log-line:last-child{border-bottom:none}
.log-time{color:#475569;font-size:10px}
.log-level{font-size:10px;text-transform:uppercase;letter-spacing:.5px;font-weight:700}
.log-msg{color:#dce8f0;word-break:break-word}
.log-input .log-level{color:#94a3b8}
.log-qwen .log-level{color:var(--blue)}
.log-tool .log-level{color:var(--green)}
.log-verify .log-level{color:var(--orange)}
.log-confidence .log-level{color:#facc15}
.log-thinking .log-msg{color:#7fb8d4;white-space:pre-wrap}
.log-synthesis .log-msg{color:#86efac;white-space:pre-wrap}
.tool-badge{display:inline-flex;align-items:center;gap:4px;background:rgba(34,197,94,.1);color:#22c55e;border:1px solid rgba(34,197,94,.2);padding:3px 9px;border-radius:5px;font-size:10px;font-weight:600}
.tool-badge.verify{background:rgba(245,158,11,.1);color:var(--orange);border-color:rgba(245,158,11,.2)}

.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--dim);gap:12px;text-align:center;padding:40px}
.empty-state .icon{font-size:48px;opacity:.3}
.empty-state .text{font-size:13px;max-width:280px;line-height:1.6}

/* === Results === */
.result-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;backdrop-filter:blur(12px)}
.result-header{padding:14px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.result-header h2{margin:0;font-size:14px;color:var(--blue)}
.result-body{padding:16px 18px}
.match-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.match-row:last-child{border:none}
.match-name{font-weight:500;font-size:13px}
.match-cas{color:var(--dim);font-size:11px}
.match-score{font-size:12px;font-weight:700;color:var(--blue)}
.verdict{display:inline-block;padding:3px 10px;border-radius:16px;font-size:11px;font-weight:700;text-transform:uppercase}
.verdict-confirmed{background:rgba(34,197,94,.12);color:var(--green)}
.verdict-needs_review{background:rgba(245,158,11,.12);color:var(--orange)}
.verdict-rejected{background:rgba(239,68,68,.12);color:var(--red)}
.verdict-no_results{background:rgba(239,68,68,.12);color:var(--red)}
.reasoning{background:rgba(0,0,0,.3);border-radius:8px;padding:12px 14px;margin:10px 0;font-size:12px;line-height:1.7;color:var(--dim)}
.markdown-content{white-space:normal;word-break:break-word}
.markdown-content h3,.markdown-content h4{color:var(--text);font-size:13px;margin:12px 0 6px}
.markdown-content h3:first-child,.markdown-content h4:first-child{margin-top:0}
.markdown-content p{margin:6px 0}
.markdown-content ul,.markdown-content ol{margin:6px 0;padding-left:18px}
.markdown-content li{margin:3px 0}
.markdown-content strong{color:var(--text);font-weight:700}
.markdown-content hr{border:0;border-top:1px solid rgba(255,255,255,.1);margin:10px 0}
.markdown-content table{width:100%;border-collapse:collapse;margin:8px 0;font-size:11px;background:rgba(0,0,0,.2);border-radius:8px;overflow:hidden}
.markdown-content th,.markdown-content td{border:1px solid rgba(255,255,255,.08);padding:6px 8px;text-align:left}
.markdown-content th{color:var(--text);background:rgba(58,124,165,.08);font-weight:700}
.markdown-content td{color:var(--dim)}

.chat-box{margin-top:14px;border-top:1px solid var(--border);padding-top:14px}
.chat-messages{max-height:200px;overflow-y:auto;margin-bottom:8px}
.chat-msg{padding:8px 12px;margin:5px 0;border-radius:8px;font-size:12px;line-height:1.6}
.chat-msg.user{background:rgba(58,124,165,.08);text-align:right}
.chat-msg.agent{background:rgba(255,255,255,.04)}
.chat-input{display:flex;gap:6px}
.chat-input input{flex:1;padding:8px 12px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.08);border-radius:6px;color:var(--text);font-size:12px}
.actions{display:flex;gap:8px;margin-top:14px}
.actions .btn{flex:1}
.report{background:rgba(0,0,0,.3);border-radius:8px;padding:16px;font-size:12px;line-height:1.8;white-space:pre-wrap;color:var(--dim)}
.hidden{display:none}
.fade-in{animation:fadeIn .3s ease-out}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
footer{padding:12px 20px;border-top:1px solid var(--border);font-size:10px;color:var(--dim);text-align:center}
footer a{color:var(--blue);text-decoration:none}
.verify-banner{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);border-radius:8px;padding:8px 12px;margin:6px 0;font-size:11px;color:var(--orange)}
.verify-done{background:rgba(34,197,94,.06);border:1px solid rgba(34,197,94,.2);border-radius:8px;padding:8px 12px;margin:6px 0;font-size:11px;color:var(--green)}
</style>
</head>
<body>
<div class="layout">

<div class="top-bar">
  <h1>ChemSpectra Agent</h1>
  <div class="meta">
    <span class="badge">Qwen-3.7-Max</span>
    <span>FTIR Analysis Autopilot</span>
    <span>&bull;</span>
    <span>130K reference spectra</span>
  </div>
</div>

<!-- LEFT PANEL -->
<div class="left-panel">

  <!-- Quick Demo -->
  <details class="card" id="demoCard">
    <summary style="cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between">
      <h2 style="margin:0">Quick Demo</h2>
      <span style="font-size:11px;color:var(--dim)">Click to expand</span>
    </summary>
    <p style="margin-top:12px">Select a sample to start analysis instantly.</p>
    <div class="sample-grid" id="sampleGrid"></div>
  </details>

  <!-- Upload -->
  <div class="card" id="uploadCard">
    <h2>Or Upload</h2>
    <form id="uploadForm" enctype="multipart/form-data">
      <label>Spectrum File</label>
      <input type="file" name="file" id="specFile" accept=".csv,.jdx,.dx,.txt,.json">
      <label>Context <span style="font-size:9px;color:var(--dim);text-transform:none;letter-spacing:0">(optional, max 200)</span></label>
      <textarea name="context" id="sampleCtx" rows="2" maxlength="200" placeholder="e.g. Polymer film, suspected PE/PP blend"></textarea>
      <input type="hidden" name="analysis_type" value="identify">
      <button type="submit" class="btn btn-primary" id="submitBtn">Analyze</button>
    </form>
  </div>

  <!-- Pipeline -->
  <div class="card hidden" id="pipelineCard">
    <h2>Pipeline</h2>
    <div class="pipeline" id="pipeline">
      <div class="step pending" id="s1"><span class="step-icon">1</span> Reasoning &amp; tool selection</div>
      <div class="step pending" id="s2"><span class="step-icon">2</span> Executing FTIR.fun tools</div>
      <div class="step pending" id="s3"><span class="step-icon">3</span> Synthesizing results</div>
      <div class="step pending" id="s4"><span class="step-icon">4</span> Human confirmation</div>
      <div class="step pending" id="s5"><span class="step-icon">5</span> Report generation</div>
    </div>
  </div>

  <!-- Results -->
  <div class="hidden" id="resultsSection">
    <div class="result-card fade-in">
      <div class="result-header">
        <h2>Results</h2>
        <div>
          <span class="verdict" id="verdictBadge"></span>
          <span style="color:var(--dim);font-size:11px;margin-left:6px" id="searchMode"></span>
        </div>
      </div>
      <div class="result-body">
        <div id="bestMatch" style="margin-bottom:12px"></div>
        <div class="reasoning" id="reasoningText"></div>
        <div id="decisionSection" class="hidden" style="margin-top:10px">
          <h3 style="font-size:12px;color:var(--blue);margin-bottom:4px">Evidence Arbitration</h3>
          <div class="reasoning" id="decisionText"></div>
        </div>
        <h3 style="font-size:12px;color:var(--dim);margin:12px 0 6px">Candidates</h3>
        <div id="candidateList"></div>
        <div id="toolsSection" class="hidden" style="margin-top:10px">
          <h3 style="font-size:12px;color:var(--blue);margin-bottom:4px">Tools Used</h3>
          <div id="toolsList" style="font-size:11px;color:var(--dim);display:flex;gap:5px;flex-wrap:wrap"></div>
        </div>
        <div id="synthesisSection" class="hidden" style="margin-top:10px">
          <h3 style="font-size:12px;color:var(--dim);margin-bottom:4px">Synthesis</h3>
          <div class="reasoning" id="synthesisText"></div>
        </div>
        <div id="flagsSection" class="hidden" style="margin-top:10px">
          <h3 style="font-size:12px;color:var(--orange);margin-bottom:4px">Flags</h3>
          <ul id="flagsList" style="font-size:11px;color:var(--dim);padding-left:16px"></ul>
        </div>
        <div id="planSection" class="hidden" style="margin-top:10px">
          <p id="planTitle" style="font-size:11px;font-weight:600;color:var(--blue)"></p>
          <ul id="planList" style="font-size:11px;color:var(--dim);padding-left:16px"></ul>
        </div>
        <div class="chat-box" id="chatBox">
          <div class="chat-messages" id="chatMessages"></div>
          <div class="chat-input">
            <input type="text" id="chatInput" placeholder="Ask about the results..." onkeydown="if(event.key==='Enter')sendChat()">
            <button class="btn btn-primary btn-sm" onclick="sendChat()">Send</button>
          </div>
        </div>
        <div class="actions" id="confirmActions">
          <button class="btn btn-success" onclick="confirmResult()">Accept &amp; Report</button>
          <button class="btn btn-warn" onclick="resetForm()">New Analysis</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Report -->
  <div class="hidden" id="reportSection">
    <div class="card fade-in">
      <h2>Report</h2>
      <div class="report" id="reportContent"></div>
      <div style="margin-top:12px;display:flex;gap:8px">
        <button class="btn btn-primary" style="flex:1" onclick="downloadReport()">Download (.md)</button>
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

<!-- RIGHT PANEL: Thinking Stream -->
<div class="right-panel" id="rightPanel">
  <div class="thinking-header">
    <span class="thinking-dot" id="thinkingDot" style="display:none"></span>
    <span class="label" id="thinkingLabel">Agent Reasoning</span>
    <span class="status" id="thinkingStatus"></span>
  </div>
  <div class="thinking-activity" id="thinkingActivity"></div>
  <div class="thinking-body" id="thinkingBody">
    <div class="empty-state" id="emptyState">
      <div class="icon">&#129302;</div>
      <div class="text">Select a sample or upload a spectrum.<br>The agent's reasoning chain will stream here in real time.</div>
    </div>
  </div>
</div>

</div>

<script>
const $ = id => document.getElementById(id);
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
let currentSessionId = null;
let streamBuffers = {thinking: '', synthesis: ''};

/* === Sample grid loader === */
const SAMPLE_DESCRIPTIONS = {
  '20260212214838789416377.dx': 'Unknown Sample A',
  'gelatin.csv': 'Gelatin protein',
  'polypropylene.csv': 'Polypropylene (PP)',
  '20260620134005173506225.csv': 'Unknown mixture sample',
};

(async function loadSamples() {
  try {
    const resp = await fetch('/api/samples');
    const samples = await resp.json();
    const grid = $('sampleGrid');
    samples.forEach(s => {
      const desc = SAMPLE_DESCRIPTIONS[s.filename] || s.filename.replace(/\\.[^.]+$/, '');
      const btn = document.createElement('button');
      btn.className = 'btn-sample';
      btn.innerHTML = '<span class="sample-icon">&#128200;</span><div class="sample-info"><span class="sample-name">' + esc(desc) + '</span><span class="sample-file">' + esc(s.filename) + '</span></div>';
      btn.onclick = () => loadSample(s.url, s.filename, desc);
      grid.appendChild(btn);
    });
  } catch(e) { console.warn('Failed to load samples', e); }
})();

async function loadSample(url, filename, desc) {
  const resp = await fetch(url);
  const blob = await resp.blob();
  const file = new File([blob], filename, {type: blob.type});
  const dt = new DataTransfer();
  dt.items.add(file);
  $('specFile').files = dt.files;
  $('sampleCtx').value = desc;
  $('demoCard').style.opacity = '0.5';
  $('uploadForm').dispatchEvent(new Event('submit', {cancelable: true}));
}

/* === Input validation + rate limiting === */
let lastSubmitTime = 0;
const COOLDOWN_MS = 15000;

function validateForm() {
  const now = Date.now();
  if (now - lastSubmitTime < COOLDOWN_MS) {
    const wait = Math.ceil((COOLDOWN_MS - (now - lastSubmitTime)) / 1000);
    alert('Please wait ' + wait + 's before submitting again.');
    return false;
  }
  const file = $('specFile').files[0];
  const ctx = $('sampleCtx').value;
  if (!file) {
    alert('Please select a spectrum file or use a demo sample above.');
    return false;
  }
  const ext = file.name.split('.').pop().toLowerCase();
  const allowed = ['csv', 'jdx', 'dx', 'txt', 'json', 'spa', 'spc'];
  if (!allowed.includes(ext)) {
    alert('Unsupported file format. Please use CSV, JCAMP-DX (.jdx/.dx), TXT, or JSON.');
    return false;
  }
  if (ctx.length > 200) {
    alert('Sample description is too long (max 200 characters).');
    return false;
  }
  lastSubmitTime = now;
  return true;
}
function setStep(n, status) {
  for (let i = 1; i <= 5; i++) {
    const el = $('s' + i);
    el.className = 'step ' + (i < n ? 'done' : i === n ? status : 'pending');
    el.querySelector('.step-icon').textContent = i < n ? '\\u2713' : i;
  }
}

const TOOL_LABELS = {
  search_library: 'Library Search',
  assess_direction: 'Direction Arbitration',
  cross_validate: 'Cross Validate',
  search_public_cases: 'Public Cases',
};

function appendThinking(text) {
  appendBuffered('thinking', text, 'thinking');
}

function setThinkingLabel(label) {
  $('thinkingLabel').textContent = label;
}

function appendLog(level, message, cls) {
  const el = $('thinkingBody');
  const line = document.createElement('div');
  const safeCls = cls || level || 'qwen';
  line.className = 'log-line log-' + safeCls;
  const now = new Date();
  const ts = now.toLocaleTimeString([], {hour12:false, minute:'2-digit', second:'2-digit'});
  line.innerHTML =
    '<span class="log-time">' + esc(ts) + '</span>' +
    '<span class="log-level">' + esc(level || 'log') + '</span>' +
    '<span class="log-msg">' + esc(message || '') + '</span>';
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function appendBuffered(level, text, cls) {
  if (!text) return;
  streamBuffers[level] = (streamBuffers[level] || '') + text;
  const buf = streamBuffers[level];
  const shouldFlush =
    /[.!?。！？]\\s*$/.test(buf) ||
    /\\n\\s*$/.test(buf) ||
    buf.length >= 180;
  if (!shouldFlush) return;
  const cleaned = buf.replace(/\\s+/g, ' ').trim();
  if (cleaned) appendLog(level, cleaned, cls || level);
  streamBuffers[level] = '';
}

function flushBuffered(level, cls) {
  const buf = (streamBuffers[level] || '').replace(/\\s+/g, ' ').trim();
  if (buf) appendLog(level, buf, cls || level);
  streamBuffers[level] = '';
}

function addActivityBadge(tool, phase) {
  const act = $('thinkingActivity');
  const cls = phase === 'verification' ? 'tool-badge verify' : 'tool-badge';
  const label = TOOL_LABELS[tool] || tool;
  act.innerHTML += `<span class="${cls}">&#8594; ${esc(label)}</span>`;
}

$('uploadForm').addEventListener('submit', async e => {
  e.preventDefault();
  if (!validateForm()) return;
  $('submitBtn').disabled = true;
  $('pipelineCard').classList.remove('hidden');
  $('emptyState').style.display = 'none';
  $('thinkingDot').style.display = '';
  $('thinkingDot').style.animation = '';
  $('thinkingDot').style.background = 'var(--blue)';
  $('thinkingBody').innerHTML = '';
  $('thinkingActivity').innerHTML = '';
  $('thinkingStatus').textContent = '';
  streamBuffers = {thinking: '', synthesis: ''};
  $('resultsSection').classList.add('hidden');
  $('reportSection').classList.add('hidden');
  $('chatMessages').innerHTML = '';
  setStep(1, 'active');
  setThinkingLabel('Qwen-3.7-Max is reasoning...');
  appendLog('input', 'Submitting spectrum to ChemSpectra Agent.', 'input');

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
            flushBuffered('thinking', 'thinking');
            flushBuffered('synthesis', 'synthesis');
            setThinkingLabel(evt.data.label || evt.data.phase);
            appendLog('phase', evt.data.label || evt.data.phase, 'qwen');
            if (evt.data.phase === 'ReAct') setStep(1, 'active');
            else if (evt.data.phase === 'Synthesis') setStep(3, 'active');
            else if (evt.data.phase === 'Verification synthesis') setStep(3, 'active');
            break;
          case 'log':
            appendLog(evt.data.level || 'log', evt.data.message || '', evt.data.level || 'qwen');
            break;
          case 'tool_plan':
            appendLog('qwen', evt.data.message || 'Qwen selected tools.', 'qwen');
            break;
          case 'thinking':
            appendThinking(evt.data.text);
            break;
          case 'synthesis_chunk':
            appendBuffered('synthesis', evt.data.text, 'synthesis');
            break;
          case 'tool_call':
            flushBuffered('thinking', 'thinking');
            flushBuffered('synthesis', 'synthesis');
            setStep(2, 'active');
            addActivityBadge(evt.data.tool, evt.data.phase);
            appendLog('tool', evt.data.message || ('Calling ' + evt.data.tool), evt.data.phase === 'verification' ? 'verify' : 'tool');
            break;
          case 'tool_result':
            flushBuffered('thinking', 'thinking');
            flushBuffered('synthesis', 'synthesis');
            appendLog('tool', evt.data.message || (evt.data.tool + ' returned results'), evt.data.phase === 'verification' ? 'verify' : 'tool');
            break;
          case 'confidence':
            flushBuffered('thinking', 'thinking');
            flushBuffered('synthesis', 'synthesis');
            appendLog('confidence', evt.data.message || 'Confidence calculated.', 'confidence');
            break;
          case 'done':
            flushBuffered('thinking', 'thinking');
            flushBuffered('synthesis', 'synthesis');
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
    $('thinkingStatus').textContent = 'Done';

    if (finalData.step === 'needs_clarification') {
      setStep(1, 'done');
      addChatMsg('agent', finalData.question || 'Could you provide more details?');
      $('resultsSection').classList.remove('hidden');
      $('bestMatch').innerHTML = '<div style="color:var(--orange)">Agent needs more information</div>';
      setMarkdown($('reasoningText'), finalData.question || '');
      $('decisionSection').classList.add('hidden');
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

  setMarkdown($('reasoningText'), c.reasoning || 'Chemical verification completed.');

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
      search_library: 'Library Search',
      assess_direction: 'Direction Arbitration',
      cross_validate: 'Cross Validate',
      search_public_cases: 'Public Cases',
    };
    $('toolsList').innerHTML = toolsCalled.map(t =>
      '<span style="background:rgba(58,124,165,.12);color:var(--blue);padding:3px 10px;border-radius:12px;font-size:11px">' +
      esc(toolLabels[t] || t) + '</span>'
    ).join('');
  } else {
    $('toolsSection').classList.add('hidden');
  }

  const synthesis = c.synthesis || '';
  if (synthesis) {
    $('synthesisSection').classList.remove('hidden');
    setMarkdown($('synthesisText'), synthesis);
  } else {
    $('synthesisSection').classList.add('hidden');
  }

  const direction = c.direction_assessment || {};
  const decisionLines = [];
  if (c.resolved_level) decisionLines.push('Resolved level: ' + c.resolved_level);
  if (direction.dominant_direction) decisionLines.push('Dominant direction: ' + direction.dominant_direction);
  if (typeof direction.entity_share === 'number' && typeof direction.direction_confidence === 'number') {
    const es = (direction.entity_share * 100).toFixed(1);
    const dc = (direction.direction_confidence * 100).toFixed(1);
    decisionLines.push('Confidence shift: locking a single entry = ' + es + '%  ->  locking the ' +
      (direction.dominant_direction || 'material') + ' direction = ' + dc + '%');
  }
  if (direction.reason) decisionLines.push(direction.reason);
  if (c.review_required) decisionLines.push('Human review required before using this result in a report.');
  if (decisionLines.length > 0) {
    $('decisionSection').classList.remove('hidden');
    setMarkdown($('decisionText'), decisionLines.join('\\n'));
    if (c.decision_status) {
      const statusColors = {green: 'var(--green)', yellow: 'var(--orange)', red: 'var(--red)'};
      const color = statusColors[c.decision_status] || 'var(--dim)';
      $('decisionText').innerHTML =
        '<p>Decision status: <span style="color:' + color + ';font-weight:700;text-transform:uppercase">' +
        esc(c.decision_status) + '</span></p>' + $('decisionText').innerHTML;
    }
  } else {
    $('decisionSection').classList.add('hidden');
  }

  const flags = c.flags || [];
  if (flags.length > 0) {
    $('flagsSection').classList.remove('hidden');
    $('flagsList').innerHTML = flags.map(f => '<li>' + esc(f) + '</li>').join('');
  } else {
    $('flagsSection').classList.add('hidden');
  }

  const plan = c.verification_plan || {};
  const steps = plan.steps || [];
  if (steps.length > 0) {
    $('planSection').classList.remove('hidden');
    const goalLabel = plan.goal ? (' <span style="color:var(--dim)">(goal: ' + esc(plan.goal) + ')</span>') : '';
    $('planTitle').innerHTML = 'Recommended next steps to confirm the material' + goalLabel;
    $('planList').innerHTML = steps.map(s => '<li>' + esc(s) + '</li>').join('');
  } else {
    $('planSection').classList.add('hidden');
  }
}

function addChatMsg(role, text) {
  const el = document.createElement('div');
  el.className = 'chat-msg ' + role;
  if (role === 'agent') {
    setMarkdown(el, text);
  } else {
    el.textContent = text;
  }
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
  $('pipelineCard').classList.add('hidden');
  $('resultsSection').classList.add('hidden');
  $('reportSection').classList.add('hidden');
  $('chatMessages').innerHTML = '';
  setStep(1, 'pending');
  $('confirmActions').innerHTML =
    '<button class="btn btn-success" onclick="confirmResult()">Accept &amp; Report</button>' +
    '<button class="btn btn-warn" onclick="resetForm()">New Analysis</button>';
  $('confirmActions').classList.remove('hidden');
  $('demoCard').style.opacity = '1';
  $('thinkingBody').innerHTML = '<div class="empty-state" id="emptyState"><div class="icon">&#129302;</div><div class="text">Select a sample or upload a spectrum.<br>The reasoning chain will stream here in real time.</div></div>';
  $('thinkingDot').style.display = 'none';
  $('thinkingActivity').innerHTML = '';
  $('thinkingStatus').textContent = '';
  setThinkingLabel('Agent Reasoning');
}

function setMarkdown(el, text) {
  el.classList.add('markdown-content');
  el.innerHTML = renderMarkdown(text || '');
}

function renderInlineMarkdown(text) {
  return esc(text)
    .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function isTableSeparator(line) {
  return /^\\|?\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?$/.test(line);
}

function splitTableRow(line) {
  return line.replace(/^\\|/, '').replace(/\\|$/, '').split('|').map(c => c.trim());
}

function renderMarkdown(text) {
  const lines = String(text || '').replace(/\\r\\n/g, '\\n').split('\\n');
  const html = [];
  let listType = null;

  function closeList() {
    if (listType) {
      html.push('</' + listType + '>');
      listType = null;
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const line = raw.trim();

    if (!line) {
      closeList();
      continue;
    }

    if (line.startsWith('|') && lines[i + 1] && isTableSeparator(lines[i + 1].trim())) {
      closeList();
      const headers = splitTableRow(line);
      html.push('<table><thead><tr>' + headers.map(h => '<th>' + renderInlineMarkdown(h) + '</th>').join('') + '</tr></thead><tbody>');
      i += 2;
      for (; i < lines.length; i++) {
        const rowLine = lines[i].trim();
        if (!rowLine.startsWith('|')) {
          i--;
          break;
        }
        const cells = splitTableRow(rowLine);
        html.push('<tr>' + cells.map(c => '<td>' + renderInlineMarkdown(c) + '</td>').join('') + '</tr>');
      }
      html.push('</tbody></table>');
      continue;
    }

    const heading = line.match(/^(#{1,4})\\s+(.+)$/);
    if (heading) {
      closeList();
      const tag = heading[1].length <= 2 ? 'h3' : 'h4';
      html.push('<' + tag + '>' + renderInlineMarkdown(heading[2]) + '</' + tag + '>');
      continue;
    }

    if (/^-{3,}$/.test(line)) {
      closeList();
      html.push('<hr>');
      continue;
    }

    const bullet = line.match(/^[-*]\\s+(.+)$/);
    if (bullet) {
      if (listType !== 'ul') {
        closeList();
        listType = 'ul';
        html.push('<ul>');
      }
      html.push('<li>' + renderInlineMarkdown(bullet[1]) + '</li>');
      continue;
    }

    const numbered = line.match(/^\\d+\\.\\s+(.+)$/);
    if (numbered) {
      if (listType !== 'ol') {
        closeList();
        listType = 'ol';
        html.push('<ol>');
      }
      html.push('<li>' + renderInlineMarkdown(numbered[1]) + '</li>');
      continue;
    }

    closeList();
    html.push('<p>' + renderInlineMarkdown(line) + '</p>');
  }

  closeList();
  return html.join('');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
