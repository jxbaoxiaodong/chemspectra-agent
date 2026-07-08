"""
ChemSpectra Agent — 多轮自主推理 FTIR 光谱分析 Autopilot。
Track 4: Autopilot Agent — Qwen Cloud Hackathon.

核心架构:
  1. ReAct 循环 — Qwen-Max Function Calling 自主选择工具、迭代推理
  2. search_library — 单次 FTIR.fun REST 谱库搜索
  3. assess_direction — 本地确定性 Top-N 方向共识裁决
  4. cross_validate — 本地确定性化学一致性规则
  5. search_public_cases — MCP 公开案例检索
  6. Self-repair — LLM 输出格式错误时带上下文重试
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import dashscope
from dashscope import Generation

from audit import audit_logger
from cross_validation import (
    ENTITY_STRONG_MATCH_MIN_SCORE,
    MATCH_QUALITY_MIN_SCORE,
    assess_direction,
    check_spectrum_quality,
    cross_validate,
    recommend_verification_plan,
)
from tools import DEFAULT_LIBRARY_TOP_K, FtirfunClient

logger = logging.getLogger(__name__)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
FTIRFUN_API_KEY = os.environ.get("FTIRFUN_API_KEY", "")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.7-max")

# Deterministic function-calling behavior for repeatable demo runs.
QWEN_TOOL_TEMPERATURE = 0
MAX_REACT_ITERATIONS = 6
HUMAN_REVIEW_CONFIDENCE_THRESHOLD = MATCH_QUALITY_MIN_SCORE


def _extract_json(text: str) -> dict | list | None:
    """从 LLM 输出中提取 JSON（可能被 markdown 代码块包裹）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    func = tool_call.get("function", tool_call) if isinstance(tool_call, dict) else getattr(tool_call, "function", tool_call)
    if isinstance(func, dict):
        name = func.get("name")
        arguments = func.get("arguments")
    else:
        name = getattr(func, "name", None)
        arguments = getattr(func, "arguments", None)
    call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
    return {
        "id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _serialize_qwen_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
        return {
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
            "tool_calls": [_normalize_tool_call(tc) for tc in tool_calls],
        }
    tool_calls = getattr(message, "tool_calls", None) or []
    return {
        "content": getattr(message, "content", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "tool_calls": [_normalize_tool_call(tc) for tc in tool_calls],
    }


# ── 工具定义（供 Qwen Function Calling 使用）──────────────────────────────────

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "description": (
                "Search the FTIR.fun spectral library once. Returns ranked material matches, "
                "peak explanations, functional-group evidence, confidence, and summary. "
                "This is the primary data retrieval tool. Call it first for any spectrum analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "description": "Number of library matches to return (1-50)",
                        "default": 15,
                    },
                    "sample_type": {
                        "type": "string",
                        "description": (
                            "Sample type ONLY if the user explicitly stated it in the "
                            "request or sample context. Never guess it and never infer "
                            "it from the filename; omit this parameter when not stated."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_direction",
            "description": (
                "Run local deterministic Top-N direction arbitration on search_library results. "
                "MUST be called after search_library when Top-1 score is below 0.85 or when "
                "entity-level identification is not defensible. Returns resolved_level "
                "(entity, library_direction, uncertain_direction) and Green/Yellow/Red status."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_validate",
            "description": (
                "Run local deterministic chemical consistency checks on search_library results. "
                "Checks lead-score gap, material-family functional-group agreement, peak coverage, "
                "background bands, and basic quality. MUST be called after search_library."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_public_cases",
            "description": (
                "Search publicly shared FTIR analysis cases via FTIR.fun MCP. "
                "Only call this when the user explicitly asks for public cases, prior analyses, "
                "or external examples. Do not call it for routine identification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Material name, CAS number, or concise FTIR case query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_peaks",
            "description": (
                "Explain what specific FTIR peak positions indicate in terms of functional groups "
                "and chemical bonds. Use this when the user asks about the meaning of specific "
                "wavenumber peaks — NOT when they want to identify a material. "
                "This is a different intent from search_library."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "peaks": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Peak positions in cm-1 to explain",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional natural language question about the peaks",
                    },
                    "sampling_mode": {
                        "type": "string",
                        "description": "ATR or transmission, if known",
                    },
                },
                "required": ["peaks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_spectrum_quality",
            "description": (
                "Parse the uploaded FTIR spectrum file and check for quality issues: "
                "CO₂/moisture background contamination, high noise, saturation, baseline offset, "
                "and edge truncation. Call this first when the user asks about spectrum quality, "
                "data reliability, or before committing to material identification on a suspect "
                "spectrum. Returns detected peaks and a list of quality warnings."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


@dataclass
class Session:
    """每请求独立的会话状态——确保并发请求互不干扰。"""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    step: str = "idle"
    user_input: str = ""
    sample_context: str = ""
    file_base64: str | None = None
    filename: str = "spectrum.0"
    file_sha256: str = ""
    file_size_bytes: int = 0
    peaks: list[float] | None = None
    tool_calls_log: list[dict] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    search_results: list[dict] = field(default_factory=list)
    search_summary: str = ""
    synthesis: str = ""
    verification: dict = field(default_factory=dict)
    verification_plan: dict = field(default_factory=dict)
    human_confirmed: bool = False
    final_report: str = ""
    conversation: list[dict] = field(default_factory=list)
    # 多轮推理度量
    react_iterations: int = 0
    repair_count: int = 0
    evidence_conflicts: list[dict] = field(default_factory=list)
    confidence_trace: list[float] = field(default_factory=list)
    resolved_level: str = ""
    decision_status: str = ""
    review_required: bool = False
    # 流式事件队列（SSE 端点消费）
    event_queue: queue.Queue = field(default_factory=queue.Queue)


class ChemSpectraAgent:
    """多工具自主路由 FTIR 分析 Agent。

    使用 Qwen Function Calling 让 LLM 自主决定调用哪些工具，
    实现 ReAct 循环而非固定流水线。
    """

    SYSTEM_PROMPT = """You are ChemSpectra, an expert AI agent for FTIR infrared spectral analysis.

You have access to six real analysis tools. Based on the user's request, YOU decide which tools
to call and in what order — this is evidence arbitration, not a fixed pipeline.

AVAILABLE TOOLS:
- search_library: FTIR.fun REST library search. Returns ranked matches, peak explanations, evidence, confidence.
- assess_direction: Local deterministic Top-N direction arbitration. Returns entity/library_direction/uncertain_direction.
- cross_validate: Local deterministic chemical consistency checks on library results.
- search_public_cases: MCP search over public FTIR analysis cases. Different data source.
- explain_peaks: Explain what specific peak positions indicate (functional groups, bonds).
  Use when the user asks about peak meaning — NOT for material identification.
- check_spectrum_quality: Parse the uploaded file and check for background contamination, noise,
  saturation, baseline offset. Use when quality is in question or before identification on a suspect spectrum.

INTENT ROUTING — decide based on what the user is asking:
- "What material is this?" / "Identify this spectrum" → search_library, then cross_validate
- "What does 1715 cm-1 indicate?" / "Explain this peak" → explain_peaks (not search_library)
- "Is my spectrum OK?" / "Check for background noise" → check_spectrum_quality
- Near-tied top candidates (gap ≤ 0.02) or Top-1 score < 0.85 after search_library → also call assess_direction
- User explicitly asks for prior cases → search_public_cases

ARBITRATION LEVELS:
- entity/Green: Top-1 score ≥ 0.85 and direction fully converges — lock the specific compound
- library_direction/Yellow: Strong direction consensus but candidates too close to lock one entity
- uncertain_direction/Red: Candidates diverge — refuse to guess, report honest uncertainty

RULES:
1. Always explain your chemical reasoning step by step.
2. Deterministic confidence is computed by the Python host. Do not invent confidence numbers.
3. Cite functional group evidence with wavenumber ranges (e.g. "1730 cm-1 → C=O ester stretch").
4. Never fabricate CAS numbers or chemical names.
5. Expose conflicts and uncertainty; never hide them.
6. If check_spectrum_quality returns warnings, report them to the user and ask whether to proceed.
"""

    def __init__(self):
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY environment variable required")
        self.ftir = FtirfunClient(api_key=FTIRFUN_API_KEY)
        self._sessions: dict[str, Session] = {}

    def new_session(self) -> Session:
        s = Session()
        self._sessions[s.session_id] = s
        return s

    def _emit(self, session: Session, event_type: str, data: Any) -> None:
        """向 Session 事件队列发射一个 SSE 事件。"""
        if event_type in {"phase", "log", "tool_plan", "tool_call", "tool_result", "confidence", "done", "error"}:
            audit_logger.write_event(
                category="session_event",
                action=event_type,
                session_id=session.session_id,
                payload=data if isinstance(data, dict) else {"value": data},
            )
        session.event_queue.put({"type": event_type, "data": data})

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    # ── Qwen API 调用 ─────────────────────────────────────────────────────────

    def _call_qwen(
        self,
        messages: list[dict],
        thinking: bool = True,
        *,
        session: Session | None = None,
        purpose: str = "text",
        **kwargs,
    ) -> str:
        """调用 Qwen（纯文本模式）。thinking=True 开启深度推理链。"""
        started = time.perf_counter()
        response = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_MODEL,
            messages=messages,
            result_format="message",
            enable_thinking=thinking,
            **kwargs,
        )
        if response.status_code != 200:
            audit_logger.write_event(
                category="llm",
                action=f"{purpose}_error",
                session_id=session.session_id if session else None,
                payload={
                    "provider": "dashscope",
                    "model": QWEN_MODEL,
                    "thinking": thinking,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "messages": messages,
                    "kwargs": kwargs,
                    "status_code": response.status_code,
                    "code": getattr(response, "code", None),
                    "message": getattr(response, "message", None),
                },
            )
            raise RuntimeError(
                f"Qwen API error: {response.code} - {response.message}"
            )
        output_text = response.output.choices[0].message.content
        audit_logger.write_event(
            category="llm",
            action=purpose,
            session_id=session.session_id if session else None,
            payload={
                "provider": "dashscope",
                "model": QWEN_MODEL,
                "thinking": thinking,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "messages": messages,
                "kwargs": kwargs,
                "status_code": response.status_code,
                "response": {
                    "content": output_text,
                },
            },
        )
        return output_text

    def _call_qwen_stream(
        self,
        messages: list[dict],
        thinking: bool = True,
        *,
        session: Session | None = None,
        purpose: str = "stream",
        **kwargs,
    ):
        """流式调用 Qwen，逐块 yield (type, text)。
        type: 'thinking' = reasoning_content, 'content' = 最终回答
        """
        started = time.perf_counter()
        reasoning_chunks: list[str] = []
        content_chunks: list[str] = []
        response = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_MODEL,
            messages=messages,
            result_format="message",
            enable_thinking=thinking,
            stream=True,
            incremental_output=True,
            **kwargs,
        )
        for chunk in response:
            if chunk.status_code != 200:
                audit_logger.write_event(
                    category="llm",
                    action=f"{purpose}_error",
                    session_id=session.session_id if session else None,
                    payload={
                        "provider": "dashscope",
                        "model": QWEN_MODEL,
                        "thinking": thinking,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                        "messages": messages,
                        "kwargs": kwargs,
                        "status_code": chunk.status_code,
                        "code": getattr(chunk, "code", None),
                        "message": getattr(chunk, "message", None),
                    },
                )
                raise RuntimeError(f"Qwen stream error: {chunk.code} - {chunk.message}")
            msg = chunk.output.choices[0].message
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                reasoning_chunks.append(rc)
                yield ("thinking", rc)
            if msg.content:
                content_chunks.append(msg.content)
                yield ("content", msg.content)
        audit_logger.write_event(
            category="llm",
            action=purpose,
            session_id=session.session_id if session else None,
            payload={
                "provider": "dashscope",
                "model": QWEN_MODEL,
                "thinking": thinking,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "messages": messages,
                "kwargs": kwargs,
                "response": {
                    "reasoning_text": "".join(reasoning_chunks),
                    "content_text": "".join(content_chunks),
                    "reasoning_chunk_count": len(reasoning_chunks),
                    "content_chunk_count": len(content_chunks),
                },
            },
        )

    def _call_qwen_json(
        self,
        messages: list[dict],
        session: Session | None = None,
        *,
        purpose: str = "json",
        **kwargs,
    ) -> dict:
        """调用 Qwen 并解析 JSON 响应，解析失败时 self-repair 重试。"""
        raw = self._call_qwen(messages, session=session, purpose=f"{purpose}_initial", **kwargs)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            return parsed

        repair_messages = list(messages) + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"Your previous response could not be parsed as valid JSON.\n"
                    f"Parse error context: expected a JSON object but got: {raw[:200]!r}\n"
                    f"Please return ONLY valid JSON matching the requested schema. "
                    f"No markdown code blocks, no extra text."
                ),
            },
        ]
        logger.info("Self-repair triggered: JSON parse failure, retrying with error context")
        audit_logger.write_event(
            category="llm",
            action=f"{purpose}_repair_triggered",
            session_id=session.session_id if session else None,
            payload={
                "raw_response": raw,
                "messages": messages,
            },
        )
        raw_retry = self._call_qwen(
            repair_messages,
            session=session,
            purpose=f"{purpose}_repair",
            **kwargs,
        )
        parsed_retry = _extract_json(raw_retry)
        if session:
            session.repair_count += 1
        if isinstance(parsed_retry, dict):
            return parsed_retry
        return {"raw_response": raw_retry}

    def _call_qwen_with_tools(
        self,
        messages: list[dict],
        *,
        session: Session | None = None,
        purpose: str = "tool_selection",
    ) -> dict:
        """调用 Qwen（Function Calling 模式），返回完整的 choice 对象。"""
        started = time.perf_counter()
        response = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_MODEL,
            messages=messages,
            tools=AGENT_TOOLS,
            result_format="message",
            temperature=QWEN_TOOL_TEMPERATURE,
        )
        if response.status_code != 200:
            audit_logger.write_event(
                category="llm",
                action=f"{purpose}_error",
                session_id=session.session_id if session else None,
                payload={
                    "provider": "dashscope",
                    "model": QWEN_MODEL,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "messages": messages,
                    "tools": AGENT_TOOLS,
                    "temperature": QWEN_TOOL_TEMPERATURE,
                    "status_code": response.status_code,
                    "code": getattr(response, "code", None),
                    "message": getattr(response, "message", None),
                },
            )
            raise RuntimeError(
                f"Qwen API error: {response.code} - {response.message}"
            )
        message = response.output.choices[0].message
        audit_logger.write_event(
            category="llm",
            action=purpose,
            session_id=session.session_id if session else None,
            payload={
                "provider": "dashscope",
                "model": QWEN_MODEL,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "messages": messages,
                "tools": AGENT_TOOLS,
                "temperature": QWEN_TOOL_TEMPERATURE,
                "status_code": response.status_code,
                "response": _serialize_qwen_message(message),
            },
        )
        return message

    # ── 工具执行分发 ──────────────────────────────────────────────────────────

    def _sanitize_tool_args(
        self,
        session: Session,
        tool_name: str,
        tool_args: dict,
    ) -> tuple[dict, str | None]:
        """Drop LLM-guessed priors the user never stated (provable honesty).

        sample_type is echo-only metadata for the FTIR.fun API, but an invented
        value ("polymer", a type read off the filename) would look like a fed-in
        answer in the reasoning log. The host keeps it only when every word of
        the hint appears in the user's own request/context text.
        """
        if tool_name != "search_library":
            return tool_args, None
        hint = str(tool_args.get("sample_type") or "").strip()
        if not hint:
            return tool_args, None
        stated = f"{session.user_input} {session.sample_context}".lower()
        if all(word in stated for word in hint.lower().split()):
            return tool_args, None
        cleaned = {k: v for k, v in tool_args.items() if k != "sample_type"}
        note = (
            f"Host gate: dropped sample_type='{hint}' — not stated by the user; "
            "the library search runs without an unfounded prior."
        )
        return cleaned, note

    def _execute_tool(self, tool_name: str, tool_args: dict, session: Session) -> dict[str, Any]:
        """Dispatch active tools. No legacy tool aliases are supported."""
        fb64 = session.file_base64
        fname = session.filename
        peaks = session.peaks

        if tool_name == "search_library":
            result = self.ftir.search_library(
                file_base64=fb64, filename=fname, peaks=peaks,
                top_k=tool_args.get("top_k", DEFAULT_LIBRARY_TOP_K),
                sample_type=tool_args.get("sample_type"),
                trace={
                    "session_id": session.session_id,
                    "tool_name": tool_name,
                    "filename": session.filename,
                    "file_sha256": session.file_sha256,
                    "file_size_bytes": session.file_size_bytes,
                },
            )
            if result.get("matches"):
                session.search_results = result["matches"]
                session.search_summary = result.get("summary", "")
            return result

        elif tool_name == "cross_validate":
            search_result = session.tool_results.get("search_library")
            if not search_result:
                return {"success": False, "error": "cross_validate requires search_library result first"}
            return cross_validate(search_result)

        elif tool_name == "assess_direction":
            search_result = session.tool_results.get("search_library")
            if not search_result:
                return {"success": False, "error": "assess_direction requires search_library result first"}
            result = assess_direction(search_result)
            session.resolved_level = result.get("resolved_level", "")
            session.decision_status = result.get("decision_status", "")
            return result

        elif tool_name == "search_public_cases":
            query = tool_args.get("query", session.user_input)
            return self.ftir.search_public_cases(
                query,
                trace={
                    "session_id": session.session_id,
                    "tool_name": tool_name,
                    "query": query,
                },
            )

        elif tool_name == "explain_peaks":
            peaks = tool_args.get("peaks") or []
            query = tool_args.get("query")
            sampling_mode = tool_args.get("sampling_mode")
            if not peaks and session.peaks:
                peaks = session.peaks
            return self.ftir.explain_peaks(
                peaks=peaks,
                query=query,
                sampling_mode=sampling_mode,
                trace={
                    "session_id": session.session_id,
                    "tool_name": tool_name,
                },
            )

        elif tool_name == "check_spectrum_quality":
            if not session.file_base64:
                return {"success": False, "error": "check_spectrum_quality requires an uploaded spectrum file"}
            parse_result = self.ftir.parse_spectrum(
                file_base64=session.file_base64,
                filename=session.filename,
                trace={
                    "session_id": session.session_id,
                    "tool_name": tool_name,
                },
            )
            if not parse_result.get("status") == "ok":
                return {
                    "success": False,
                    "error": f"parse_spectrum failed: {parse_result.get('error') or 'no status returned'}",
                }
            spectrum_points = parse_result.get("spectrum") or []
            detected_peaks = parse_result.get("peaks") or []
            quality = check_spectrum_quality(spectrum_points, detected_peaks)
            quality["parse_status"] = parse_result.get("status")
            quality["detected_peaks"] = detected_peaks[:20]
            return quality

        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def _format_tool_result_for_llm(self, tool_name: str, result: dict) -> str:
        """将工具调用结果格式化为 LLM 可读的文本摘要。"""
        if not result.get("success", True) and result.get("error"):
            return f"[{tool_name}] Error: {result['error']}"

        parts = [f"[{tool_name}] Results:"]

        if tool_name == "check_spectrum_quality":
            parts.append(f"  Quality OK: {result.get('quality_ok', False)}")
            parts.append(f"  Peaks detected: {result.get('n_peaks_detected', 0)}")
            flags = result.get("flags", [])
            if flags:
                parts.append(f"  Quality flags: {', '.join(flags)}")
            warnings = result.get("warnings", [])
            for w in warnings:
                parts.append(f"  WARNING: {w}")
            if not warnings:
                parts.append("  No quality issues detected.")
            return "\n".join(parts)

        if tool_name == "explain_peaks":
            if result.get("peak_explanations"):
                parts.append("  Peak explanations:")
                for pe in result["peak_explanations"][:15]:
                    parts.append(f"    {pe}")
            elif result.get("matches"):
                parts.append("  Functional group assignments:")
                for i, m in enumerate(result["matches"][:5], 1):
                    parts.append(f"    #{i}: {m.get('name', '?')}")
            if result.get("summary"):
                parts.append(f"  Summary: {result['summary']}")
            return "\n".join(parts)

        if tool_name == "assess_direction":
            parts.append(f"  Resolved level: {result.get('resolved_level', 'unknown')}")
            parts.append(f"  Decision status: {result.get('decision_status', 'unknown')}")
            if result.get("dominant_direction"):
                parts.append(f"  Dominant direction: {result['dominant_direction']}")
            parts.append(f"  Direction confidence: {result.get('direction_confidence', 0):.4f}")
            parts.append(f"  Supporting candidates: {result.get('supporting_candidates', 0)}")
            if result.get("supporting_candidate_names"):
                parts.append(
                    "  Supporting candidate names: "
                    + ", ".join(result["supporting_candidate_names"])
                )
            if result.get("competing_directions"):
                parts.append(
                    "  Competing directions: "
                    + json.dumps(result["competing_directions"], ensure_ascii=False)
                )
            parts.append(f"  Reason: {result.get('reason', 'No reason')}")
            return "\n".join(parts)

        if tool_name == "cross_validate":
            parts.append(f"  Summary: {result.get('summary', 'No summary')}")
            if result.get("detected_family"):
                parts.append(f"  Detected family: {result['detected_family']}")
            if result.get("observed_groups"):
                parts.append(f"  Observed groups: {', '.join(result['observed_groups'])}")
            if result.get("confidence_multiplier") is not None:
                parts.append(f"  Confidence multiplier: {result['confidence_multiplier']:.4f}")
            checks = result.get("checks", [])
            if checks:
                parts.append("  Checks:")
                for item in checks:
                    status = "PASS" if item.get("passed") else "FAIL"
                    if not item.get("applicable", True):
                        status = "N/A"
                    parts.append(f"    - {item.get('check')}: {status} | {item.get('detail')}")
            return "\n".join(parts)

        if result.get("search_mode"):
            parts.append(f"  Search mode: {result['search_mode']}")
        if result.get("summary"):
            parts.append(f"  Summary: {result['summary']}")
        if result.get("n_matches") is not None:
            parts.append(f"  Number of matches: {result['n_matches']}")
        if result.get("confidence") is not None:
            parts.append(f"  Confidence: {result['confidence']:.4f}")

        matches = result.get("matches", [])
        if matches:
            parts.append("  Top matches:")
            for i, m in enumerate(matches[:5], 1):
                score = m.get("similarity") or m.get("score", 0)
                parts.append(
                    f"    #{i}: {m.get('name', '?')} (CAS: {m.get('cas', 'N/A')}) "
                    f"score={score:.4f}"
                )

        peak_exps = result.get("peak_explanations", [])
        if peak_exps:
            parts.append("  Peak explanations:")
            for pe in peak_exps[:10]:
                parts.append(f"    {pe}")

        evidence = result.get("evidence", [])
        if evidence:
            parts.append("  Evidence:")
            for ev in evidence[:5]:
                parts.append(f"    - {ev}")

        tc = result.get("task_context", {})
        if tc.get("goal"):
            parts.append(f"  Analysis goal: {tc['goal']}")

        return "\n".join(parts)

    def _has_tool_result(self, session: Session, tool_name: str) -> bool:
        result = session.tool_results.get(tool_name)
        return isinstance(result, dict) and result.get("success", True)

    def _top_score_requires_direction_assessment(self, session: Session) -> bool:
        search_result = session.tool_results.get("search_library") or {}
        matches = search_result.get("matches") or []
        top_score = _float_or_none(search_result.get("confidence"))
        if top_score is None and matches:
            top_score = _float_or_none(matches[0].get("similarity") or matches[0].get("score"))
        if top_score is None:
            return True
        if top_score < ENTITY_STRONG_MATCH_MIN_SCORE:
            return True
        # Near-tied candidates: high score but gap is small — arbitration still
        # needed to determine whether direction fully converges (entity/green) or
        # candidates are genuinely split (library_direction/yellow).
        if len(matches) >= 2:
            s1 = _float_or_none(matches[0].get("similarity") or matches[0].get("score")) or 0.0
            s2 = _float_or_none(matches[1].get("similarity") or matches[1].get("score")) or 0.0
            gap = round(s1 - s2, 4)
            if gap <= 0.02:
                return True
        return False

    def _record_host_required_tool(
        self,
        session: Session,
        messages: list[dict],
        tool_name: str,
        reason: str,
    ) -> None:
        self._emit(session, "tool_call", {
            "tool": tool_name,
            "phase": "required_gate",
            "args": {},
            "message": f"Required evidence gate calling {tool_name}: {reason}",
        })
        result = self._execute_tool(tool_name, {}, session)
        session.tool_calls_log.append({
            "tool": tool_name,
            "args": {},
            "success": result.get("success", True),
            "n_matches": result.get("n_matches"),
            "goal": result.get("task_context", {}).get("goal"),
            "host_required": True,
            "reason": reason,
        })
        session.tool_results[tool_name] = result
        self._emit(session, "tool_result", {
            "tool": tool_name,
            "phase": "required_gate",
            "n_matches": result.get("n_matches"),
            "success": result.get("success", True),
            "message": f"{tool_name} completed required evidence gate.",
        })
        messages.append({
            "role": "user",
            "content": (
                f"Required evidence gate executed {tool_name} because: {reason}\n"
                f"{self._format_tool_result_for_llm(tool_name, result)}"
            ),
        })

    def _enforce_required_evidence_path(
        self,
        session: Session,
        messages: list[dict],
    ) -> bool:
        if not self._has_tool_result(session, "search_library"):
            self._record_host_required_tool(
                session,
                messages,
                "search_library",
                "spectrum identification requires library evidence before synthesis",
            )
            return True
        if (
            self._top_score_requires_direction_assessment(session)
            and not self._has_tool_result(session, "assess_direction")
        ):
            self._record_host_required_tool(
                session,
                messages,
                "assess_direction",
                "near-tied candidates or low Top-1 score — direction arbitration required",
            )
            return True
        if not self._has_tool_result(session, "cross_validate"):
            self._record_host_required_tool(
                session,
                messages,
                "cross_validate",
                "deterministic confidence requires cross-validation",
            )
            return True
        return False

    # ── ReAct 循环核心 ────────────────────────────────────────────────────────

    def run_tool_loop(self, session: Session) -> dict:
        """ReAct 循环——Qwen 自主决定调用工具，循环直到生成最终分析。

        返回包含工具调用日志和最终综合分析的字典。
        """
        session.step = "reasoning"

        spectrum_desc = ""
        if session.file_base64:
            spectrum_desc = f"Spectrum file uploaded: {session.filename}"
        elif session.peaks:
            spectrum_desc = f"Peak positions (cm-1): {', '.join(str(p) for p in session.peaks)}"

        user_prompt = f"""Analyze this FTIR spectrum request.

User request: {session.user_input}
Sample context: {session.sample_context or 'Not provided'}
{spectrum_desc}

Tool routing:
- Material identification → search_library + cross_validate (+ assess_direction if near-tied)
- Peak meaning / functional group question → explain_peaks
- Spectrum quality check / background concern → check_spectrum_quality
- Prior public cases → search_public_cases (only if explicitly requested)

For identification: search_library is the minimum entry point. If Top-1 candidates are
near-tied (gap ≤ 0.02) or score < 0.85, call assess_direction to run direction arbitration.
After receiving tool results, provide your final chemical analysis and synthesis.
"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        max_iterations = MAX_REACT_ITERATIONS
        iteration = 0

        self._emit(session, "phase", {"phase": "ReAct", "label": "Agent reasoning log: selecting analysis tools..."})
        self._emit(session, "log", {
            "level": "input",
            "message": f"Loaded request: {session.user_input}",
        })
        self._emit(session, "log", {
            "level": "input",
            "message": spectrum_desc or "Loaded manual spectrum input.",
        })
        self._emit(session, "log", {
            "level": "qwen",
            "message": f"Sending request to Qwen-3.7-Max with {len(AGENT_TOOLS)} real tool schemas.",
        })

        while iteration < max_iterations:
            iteration += 1
            self._emit(session, "log", {
                "level": "qwen",
                "message": f"ReAct round {iteration}: Qwen is deciding whether tools are needed.",
            })
            response_msg = self._call_qwen_with_tools(messages, session=session, purpose="react_tool_selection")

            try:
                tool_calls = response_msg.tool_calls or []
            except (KeyError, AttributeError):
                tool_calls = []
            if not tool_calls:
                if self._enforce_required_evidence_path(session, messages):
                    continue
                # 最终综合分析——用流式调用让 thinking 可见
                self._emit(session, "phase", {"phase": "Synthesis", "label": "Synthesizing multi-tool results..."})
                self._emit(session, "log", {
                    "level": "qwen",
                    "message": "Qwen stopped calling tools and started synthesizing the evidence.",
                })
                synthesis_messages = list(messages) + [
                    {"role": "user", "content": "Now synthesize all tool results into a final chemical analysis."}
                ]
                synthesis_text = ""
                for chunk_type, chunk_text in self._call_qwen_stream(
                    synthesis_messages,
                    thinking=True,
                    session=session,
                    purpose="react_synthesis_stream",
                ):
                    if chunk_type == "thinking":
                        self._emit(session, "thinking", {"text": chunk_text})
                    else:
                        synthesis_text += chunk_text
                        self._emit(session, "synthesis_chunk", {"text": chunk_text})
                session.synthesis = synthesis_text or response_msg.content or ""
                session.step = "synthesized"
                break

            tool_names = [
                tc.get("function", tc).get("name", "unknown_tool")
                for tc in tool_calls
            ]
            self._emit(session, "tool_plan", {
                "iteration": iteration,
                "tools": tool_names,
                "message": f"Qwen selected {len(tool_names)} of {len(AGENT_TOOLS)} tools: {', '.join(tool_names)}",
            })

            messages.append({
                "role": "assistant",
                "content": response_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.get("id", f"call_{iteration}_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            })

            for i, tc in enumerate(tool_calls):
                func = tc.get("function", tc)
                tool_name = func["name"]
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                else:
                    tool_args = raw_args

                tool_args, sanitize_note = self._sanitize_tool_args(session, tool_name, tool_args)
                if sanitize_note:
                    self._emit(session, "log", {"level": "tool", "message": sanitize_note})

                logger.info("Tool call [%d/%d]: %s(%s)", iteration, i + 1, tool_name, tool_args)
                self._emit(session, "tool_call", {
                    "tool": tool_name,
                    "iteration": iteration,
                    "args": tool_args,
                    "message": f"Calling {tool_name} with {json.dumps(tool_args, ensure_ascii=False)}",
                })

                result = self._execute_tool(tool_name, tool_args, session)

                session.tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "success": result.get("success", True),
                    "n_matches": result.get("n_matches"),
                    "goal": result.get("task_context", {}).get("goal"),
                })
                session.tool_results[tool_name] = result

                top_match = ""
                if tool_name == "search_library":
                    matches = result.get("matches", [])
                    if matches:
                        top_match = f"{matches[0].get('name','?')} (score={matches[0].get('similarity',0):.4f})"
                self._emit(session, "tool_result", {
                    "tool": tool_name,
                    "n_matches": result.get("n_matches"),
                    "top_match": top_match,
                    "success": result.get("success", True),
                    "message": (
                        f"{tool_name} returned {result.get('n_matches', 0) or 0} matches. Top: {top_match}"
                        if top_match else f"{tool_name} returned evidence for synthesis."
                    ),
                })

                result_text = self._format_tool_result_for_llm(tool_name, result)
                call_id = tc.get("id", f"call_{iteration}_{i}")
                messages.append({
                    "role": "tool",
                    "content": result_text,
                    "tool_call_id": call_id,
                })

        session.react_iterations = iteration

        confidence = self._compute_confidence(session)
        self._ensure_decision_state(session)
        conflicts = self._detect_evidence_conflicts(session)
        self._emit(session, "confidence", {
            "confidence": round(confidence, 4),
            "conflicts": len(conflicts),
            "threshold": HUMAN_REVIEW_CONFIDENCE_THRESHOLD,
            "message": (
                f"Deterministic confidence: {confidence:.0%}; "
                f"{len(conflicts)} cross-validation issue(s) flagged for review."
            ),
        })

        return {
            "tools_called": [log["tool"] for log in session.tool_calls_log],
            "tool_details": session.tool_calls_log,
            "synthesis": session.synthesis,
            "metrics": self._build_metrics(session, iteration),
        }

    # ── 证据交叉验证 ────────────────────────────────────────────────────────

    def _build_metrics(self, session: Session, iteration: int | None = None) -> dict[str, Any]:
        total_llm_calls = (
            (iteration + 1 + session.repair_count)
            if iteration is not None else
            (session.react_iterations + 1 + session.repair_count)
        )
        return {
            "react_iterations": session.react_iterations,
            "repair_count": session.repair_count,
            "evidence_conflicts": len(session.evidence_conflicts),
            "confidence_trace": session.confidence_trace,
            "resolved_level": session.resolved_level,
            "decision_status": session.decision_status,
            "review_required": session.review_required,
            "total_llm_calls": total_llm_calls,
        }

    def _detect_evidence_conflicts(self, session: Session) -> list[dict]:
        """Return failed applicable checks from the local cross_validate result."""
        cv_result = session.tool_results.get("cross_validate", {})
        conflicts = [
            item for item in cv_result.get("checks", [])
            if item.get("applicable", True) and item.get("passed") is False
        ]
        session.evidence_conflicts = conflicts
        return conflicts

    def _ensure_decision_state(self, session: Session) -> None:
        if session.resolved_level and session.decision_status:
            return
        top_score = None
        search_result = session.tool_results.get("search_library") or {}
        matches = search_result.get("matches") or []
        if matches:
            top_score = _float_or_none(matches[0].get("similarity") or matches[0].get("score"))
        if top_score is not None and top_score >= ENTITY_STRONG_MATCH_MIN_SCORE:
            session.resolved_level = "entity"
            session.decision_status = "green"
        else:
            session.resolved_level = "uncertain_direction"
            session.decision_status = "red"

    def _compute_confidence(self, session: Session) -> float:
        """Compute deterministic confidence from REST score and local validation."""
        search_result = session.tool_results.get("search_library")
        if not search_result:
            raise RuntimeError("search_library result is required before confidence calculation")

        api_confidence = _float_or_none(search_result.get("confidence"))
        if api_confidence is None:
            matches = search_result.get("matches") or []
            if matches:
                api_confidence = _float_or_none(matches[0].get("similarity"))
            if api_confidence is None:
                raise RuntimeError("search_library result has no confidence or top similarity")

        cv_result = session.tool_results.get("cross_validate")
        if not cv_result:
            raise RuntimeError("cross_validate result is required before confidence calculation")

        multiplier = _float_or_none(cv_result.get("confidence_multiplier"))
        if multiplier is None:
            raise RuntimeError("cross_validate result has no confidence_multiplier")

        confidence = round(min(1.0, max(0.0, api_confidence * multiplier)), 4)
        session.confidence_trace.append(confidence)
        return confidence

    # ── 化学验证（从综合结果中提取结构化判定）────────────────────────────────

    def extract_verification(self, session: Session) -> dict:
        """从 Qwen 的综合分析中提取结构化验证结果。"""
        session.step = "verifying"

        if not session.search_results and not session.synthesis:
            v = {
                "verdict": "no_results",
                "reasoning": "No spectral matches found.",
                "top_candidate": None,
                "confidence_adjusted": 0.0,
                "flags": ["No matches"],
            }
            session.verification = v
            return v

        if not session.confidence_trace:
            raise RuntimeError("Deterministic confidence must be computed before verification extraction")

        top5 = session.search_results[:5]
        matches_str = "\n".join(
            f"  #{m.get('rank', i+1)}: {m.get('name', '?')} "
            f"(CAS: {m.get('cas', 'N/A')}) "
            f"score={m.get('similarity') or m.get('score', 0):.4f}"
            for i, m in enumerate(top5)
        ) if top5 else "No library matches available."

        tools_called = ", ".join(log["tool"] for log in session.tool_calls_log)
        cv_result = session.tool_results.get("cross_validate", {})
        cv_summary = cv_result.get("summary", "No cross-validation summary")
        cv_failed = [
            item for item in cv_result.get("checks", [])
            if item.get("applicable", True) and item.get("passed") is False
        ]
        direction_result = session.tool_results.get("assess_direction", {})

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Based on the multi-tool analysis below, provide a structured verification.

Tools called: {tools_called}

Agent synthesis:
{session.synthesis}

Library matches:
{matches_str}

Cross-validation summary:
{cv_summary}

Failed checks:
{json.dumps(cv_failed, ensure_ascii=False)}

Direction arbitration:
{json.dumps(direction_result, ensure_ascii=False)}

Sample context: {session.sample_context or 'Not provided'}

The deterministic confidence is already computed by Python as {session.confidence_trace[-1]:.4f}.
Do not estimate or change confidence.

Return ONLY valid JSON, no markdown:
{{"verdict": "confirmed|needs_review|rejected",
 "resolved_level": "entity|library_direction|uncertain_direction",
 "reasoning": "Your chemical analysis in 2-3 sentences, citing specific evidence from tool results",
 "top_candidate": "name of best match or null",
 "flags": ["list of concerns if any"]}}
""",
            },
        ]
        v = self._call_qwen_json(messages, session=session, purpose="verification_json")
        if "verdict" not in v:
            v["verdict"] = "needs_review"
        if "reasoning" not in v:
            v["reasoning"] = v.get("raw_response", session.synthesis or "Analysis completed.")
        v["confidence_adjusted"] = session.confidence_trace[-1]
        v["resolved_level"] = session.resolved_level or v.get("resolved_level", "uncertain_direction")
        v["decision_status"] = session.decision_status or "red"
        self._apply_review_gate(session, v)
        session.verification = v
        return v

    def _apply_review_gate(self, session: Session, verification: dict) -> None:
        confidence = session.confidence_trace[-1] if session.confidence_trace else 0.0
        flags = list(verification.get("flags") or [])
        if confidence < HUMAN_REVIEW_CONFIDENCE_THRESHOLD:
            session.review_required = True
            flags.append(
                f"Deterministic confidence {confidence:.4f} is below human-review threshold "
                f"{HUMAN_REVIEW_CONFIDENCE_THRESHOLD:.4f}."
            )
        if session.evidence_conflicts:
            session.review_required = True
            flags.append(f"{len(session.evidence_conflicts)} deterministic evidence conflict(s) require review.")
        if verification.get("resolved_level") == "uncertain_direction":
            session.review_required = True
            flags.append("Resolved level is uncertain_direction; entity identification is not defensible.")
        if session.review_required and verification.get("verdict") == "confirmed":
            verification["verdict"] = "needs_review"
        verification["flags"] = sorted(set(flags))
        verification["review_required"] = session.review_required
        verification["confidence_threshold"] = HUMAN_REVIEW_CONFIDENCE_THRESHOLD

    # ── Human-in-the-loop ─────────────────────────────────────────────────────

    def build_confirmation_payload(self, session: Session) -> dict:
        """构建人类确认界面的结构化数据。"""
        top = session.search_results[0] if session.search_results else {}
        v = session.verification

        candidates = []
        for i, m in enumerate(session.search_results[:5], 1):
            candidates.append({
                "rank": i,
                "name": m.get("name", "Unknown"),
                "cas": m.get("cas", "N/A"),
                "score": m.get("similarity") or m.get("score", 0),
            })

        # 评委/分析师可见的确定性证据明细，全部来自真实工具返回。
        search_result = session.tool_results.get("search_library") or {}
        cv_result = session.tool_results.get("cross_validate") or {}
        quality_result = session.tool_results.get("check_spectrum_quality") or {}

        api_confidence = _float_or_none(search_result.get("confidence"))
        if api_confidence is None and session.search_results:
            api_confidence = _float_or_none(
                session.search_results[0].get("similarity")
                or session.search_results[0].get("score")
            )
        multiplier = _float_or_none(cv_result.get("confidence_multiplier"))
        confidence_breakdown = None
        if api_confidence is not None and multiplier is not None and session.confidence_trace:
            confidence_breakdown = {
                "api_confidence": round(api_confidence, 4),
                "cross_validation_multiplier": round(multiplier, 4),
                "final_confidence": session.confidence_trace[-1],
                "formula": (
                    f"{api_confidence:.4f} (library) x {multiplier:.4f} (rule checks) "
                    f"= {session.confidence_trace[-1]:.4f}"
                ),
            }

        peak_explanations = []
        for pe in (search_result.get("peak_explanations") or [])[:12]:
            if isinstance(pe, dict):
                peak_explanations.append({
                    "peak_cm1": pe.get("peak_cm1"),
                    "assignment": pe.get("assignment"),
                    "evidence_type": pe.get("evidence_type"),
                    "doi": pe.get("doi"),
                    "document_title": pe.get("document_title"),
                })
            else:
                peak_explanations.append({"assignment": str(pe)})

        return {
            "best_match": {
                "name": top.get("name", "Unknown"),
                "cas": top.get("cas", "N/A"),
                "score": top.get("similarity") or top.get("score", 0),
            },
            "verdict": v.get("verdict", "needs_review"),
            "resolved_level": v.get("resolved_level", session.resolved_level or "uncertain_direction"),
            "decision_status": v.get("decision_status", session.decision_status or "red"),
            "review_required": v.get("review_required", session.review_required),
            "confidence_threshold": v.get("confidence_threshold", HUMAN_REVIEW_CONFIDENCE_THRESHOLD),
            "direction_assessment": session.tool_results.get("assess_direction", {}),
            "reasoning": v.get("reasoning", ""),
            "confidence": v.get("confidence_adjusted", 0),
            "confidence_breakdown": confidence_breakdown,
            "cross_validation": {
                "summary": cv_result.get("summary"),
                "confidence_multiplier": cv_result.get("confidence_multiplier"),
                "detected_family": cv_result.get("detected_family"),
                "observed_groups": cv_result.get("observed_groups"),
                "checks": cv_result.get("checks", []),
            } if cv_result else None,
            "peak_explanations": peak_explanations,
            "quality_check": {
                "quality_ok": quality_result.get("quality_ok"),
                "flags": quality_result.get("flags", []),
                "warnings": quality_result.get("warnings", []),
                "summary": quality_result.get("summary"),
            } if quality_result else None,
            "flags": v.get("flags", []),
            "candidates": candidates,
            "search_summary": session.search_summary,
            "tools_called": [log["tool"] for log in session.tool_calls_log],
            "synthesis": session.synthesis,
            "verification_plan": session.verification_plan,
        }

    def handle_followup(self, session: Session, user_message: str) -> dict:
        """处理用户在确认环节的追问。

        追问同样走 Function Calling：Qwen 可以按意图调用工具
        （例如峰位含义问题路由到 explain_peaks），而不是只做纯文本回答。
        """
        session.conversation.append({"role": "user", "content": user_message})
        audit_logger.write_event(
            category="session",
            action="followup_started",
            session_id=session.session_id,
            payload={
                "message": user_message,
                "conversation_length": len(session.conversation),
            },
        )

        top = session.search_results[0] if session.search_results else {}
        v = session.verification

        tools_used = ", ".join(log["tool"] for log in session.tool_calls_log) or "none"

        conv_history = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Agent'}: {m['content']}"
            for m in session.conversation[-6:]
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""You are in the middle of analyzing an FTIR spectrum.

Current analysis state:
- Best match: {top.get('name', 'Unknown')} (CAS: {top.get('cas', 'N/A')})
- Score: {top.get('similarity') or top.get('score', 0):.4f}
- Resolved level: {v.get('resolved_level', session.resolved_level or 'uncertain_direction')}
- Decision status: {v.get('decision_status', session.decision_status or 'red')}
- Tools used: {tools_used}
- Your synthesis: {session.synthesis[:500] if session.synthesis else 'N/A'}
- Your previous reasoning: {v.get('reasoning', 'N/A')}

Conversation so far:
{conv_history}

The user just said: "{user_message}"

Apply the same intent routing as the main analysis: if the question is about the
meaning of specific peak positions, call explain_peaks with those wavenumbers;
if it questions spectrum quality, call check_spectrum_quality. If the existing
tool results already answer the question, respond directly without new tools.
""",
            },
        ]

        followup_tools: list[str] = []
        max_followup_tool_rounds = 2
        for round_index in range(max_followup_tool_rounds):
            response_msg = self._call_qwen_with_tools(
                messages,
                session=session,
                purpose="followup_tool_selection",
            )
            try:
                tool_calls = response_msg.tool_calls or []
            except (KeyError, AttributeError):
                tool_calls = []
            if not tool_calls:
                break

            messages.append({
                "role": "assistant",
                "content": response_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.get("id", f"followup_call_{round_index}_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            })

            for i, tc in enumerate(tool_calls):
                func = tc.get("function", tc)
                tool_name = func["name"]
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                else:
                    tool_args = raw_args

                tool_args, _followup_sanitize_note = self._sanitize_tool_args(session, tool_name, tool_args)
                logger.info("Followup tool call: %s(%s)", tool_name, tool_args)
                result = self._execute_tool(tool_name, tool_args, session)
                followup_tools.append(tool_name)
                session.tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "success": result.get("success", True),
                    "n_matches": result.get("n_matches"),
                    "followup": True,
                })
                session.tool_results[tool_name] = result
                messages.append({
                    "role": "tool",
                    "content": self._format_tool_result_for_llm(tool_name, result),
                    "tool_call_id": tc.get("id", f"followup_call_{round_index}_{i}"),
                })

        messages.append({
            "role": "user",
            "content": (
                "Now answer the user's follow-up question using the evidence above. "
                "Return ONLY valid JSON:\n"
                '{"response": "your helpful answer to the user\'s question or comment",\n'
                ' "action": "none|update_context|re_search",\n'
                ' "updated_context": "if action is update_context, the new combined context"}'
            ),
        })
        result = self._call_qwen_json(messages, session=session, purpose="followup_json")
        response_text = result.get("response", result.get("raw_response", "I can help with that."))
        session.conversation.append({"role": "assistant", "content": response_text})
        result["tools_used"] = followup_tools
        audit_logger.write_event(
            category="session",
            action="followup_completed",
            session_id=session.session_id,
            payload={
                "message": user_message,
                "response": response_text,
                "action": result.get("action", "none"),
                "tools_used": followup_tools,
            },
        )
        return result

    # ── 报告生成 ──────────────────────────────────────────────────────────────

    def generate_report(self, session: Session) -> str:
        """人类确认后生成最终分析报告。"""
        session.step = "reporting"
        session.human_confirmed = True
        audit_logger.write_event(
            category="session",
            action="report_generation_started",
            session_id=session.session_id,
            payload={
                "resolved_level": session.resolved_level,
                "decision_status": session.decision_status,
            },
        )

        top = session.search_results[0] if session.search_results else {}
        v = session.verification
        tools_used = ", ".join(log["tool"] for log in session.tool_calls_log) or "none"

        library_result = session.tool_results.get("search_library", {})
        peak_info = ""
        if library_result.get("peak_explanations"):
            peak_info = (
                "\nPeak explanations from search_library: "
                f"{json.dumps(library_result['peak_explanations'][:10], ensure_ascii=False)}"
            )

        fg_info = ""
        if library_result.get("evidence"):
            fg_info = (
                "\nFunctional group evidence from search_library: "
                f"{json.dumps(library_result['evidence'][:10], ensure_ascii=False)}"
            )

        cv_info = ""
        cv_result = session.tool_results.get("cross_validate", {})
        if cv_result.get("checks"):
            cv_info = (
                "\nDeterministic cross-validation checks: "
                f"{json.dumps(cv_result['checks'], ensure_ascii=False)}"
            )
        direction_info = ""
        direction_result = session.tool_results.get("assess_direction", {})
        if direction_result:
            direction_info = (
                "\nDirection arbitration result: "
                f"{json.dumps(direction_result, ensure_ascii=False)}"
            )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Generate a professional FTIR analysis report in Markdown format.

Sample: {session.sample_context or session.user_input}
Tools used in analysis: {tools_used}

Agent multi-tool synthesis:
{session.synthesis}

Confirmed match: {top.get('name', 'Unknown')} (CAS: {top.get('cas', 'N/A')})
Match score: {top.get('similarity') or top.get('score', 0):.4f}
Chemical reasoning: {v.get('reasoning', 'N/A')}
Resolved level: {v.get('resolved_level', session.resolved_level or 'uncertain_direction')}
Decision status: {v.get('decision_status', session.decision_status or 'red')}
Flags: {', '.join(v.get('flags', [])) or 'None'}
{peak_info}
{fg_info}
{cv_info}
{direction_info}

Include these sections:
1. **Sample Information** — what was analyzed
2. **Analysis Method** — multi-tool FTIR analysis (list which tools were used and why)
3. **Results Summary** — top match, score, and verdict
4. **Chemical Reasoning** — synthesize evidence from all tools used
5. **Peak Analysis** — if peak explanations available, include detailed peak assignments
6. **Quality Notes** — flags, confidence assessment, resolved level, limitations
7. **Analyst Confirmation** — note that human-in-the-loop review was performed

Keep it professional but concise. Use wavenumber evidence where possible.
""",
            },
        ]
        session.final_report = self._call_qwen(
            messages,
            session=session,
            purpose="report_markdown",
        )
        audit_logger.write_event(
            category="session",
            action="report_generated",
            session_id=session.session_id,
            payload={
                "report_length": len(session.final_report),
            },
        )
        return session.final_report

    # ── 完整流水线 ────────────────────────────────────────────────────────────

    def run_pipeline(
        self,
        session: Session,
        user_input: str,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        sample_context: str = "",
    ) -> dict:
        """执行完整分析流程，直到人类确认检查点。

        流程:
        1. 存储光谱数据到 Session
        2. ReAct 循环: Qwen 自主选择工具 → 执行 → 综合分析
        3. 结构化验证
        4. 返回确认界面数据（等待人类确认）
        """
        session.user_input = user_input
        session.sample_context = sample_context
        session.file_base64 = file_base64
        session.filename = filename
        session.peaks = peaks
        session.conversation.append({"role": "user", "content": user_input})
        audit_logger.write_event(
            category="session",
            action="pipeline_started",
            session_id=session.session_id,
            payload={
                "user_input": user_input,
                "sample_context": sample_context,
                "filename": filename,
                "file_sha256": session.file_sha256,
                "file_size_bytes": session.file_size_bytes,
                "peaks": peaks,
            },
        )

        if not file_base64 and not peaks:
            session.step = "clarifying"
            result = {
                "step": "needs_clarification",
                "session_id": session.session_id,
                "question": (
                    "Please provide a spectrum file or peak positions (cm⁻¹) to analyze. "
                    "For example, upload a .spc/.csv/.jdx file or enter peaks like: 2920, 2850, 1460, 720"
                ),
            }
            audit_logger.write_event(
                category="session",
                action="pipeline_needs_clarification",
                session_id=session.session_id,
                payload=result,
            )
            return result

        tool_result = self.run_tool_loop(session)
        self.extract_verification(session)
        session.verification_plan = recommend_verification_plan(
            session.tool_results.get("assess_direction") or {
                "resolved_level": session.resolved_level,
                "dominant_direction": None,
                "top_match": {
                    "name": session.search_results[0].get("name")
                    if session.search_results else None,
                },
            },
            session.tool_results.get("cross_validate"),
        )
        confirmation = self.build_confirmation_payload(session)

        result = {
            "step": "awaiting_confirmation",
            "session_id": session.session_id,
            "tools_called": tool_result["tools_called"],
            "n_tools": len(tool_result["tools_called"]),
            "search_summary": session.search_summary,
            "n_matches": len(session.search_results),
            "confirmation": confirmation,
            "agent_metrics": self._build_metrics(session),
        }
        audit_logger.write_event(
            category="session",
            action="pipeline_completed",
            session_id=session.session_id,
            payload={
                "result": result,
                "verification_plan": session.verification_plan,
            },
        )
        return result
