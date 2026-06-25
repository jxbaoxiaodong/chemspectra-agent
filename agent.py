"""
ChemSpectra Agent — 多轮自主推理 FTIR 光谱分析 Autopilot。
Track 4: Autopilot Agent — Qwen Cloud Hackathon.

核心架构:
  1. ReAct 循环 — Qwen-Max Function Calling 自主选择工具、迭代推理
  2. Evidence Cross-validation — 多工具结果交叉验证、矛盾检测
  3. Self-verification — 低置信度时自动触发深度验证轮
  4. Self-repair — LLM 输出格式错误时带上下文重试
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import dashscope
from dashscope import Generation

from tools import FtirfunClient

logger = logging.getLogger(__name__)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
FTIRFUN_API_KEY = os.environ.get("FTIRFUN_API_KEY", "")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.7-max")


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


# ── 工具定义（供 Qwen Function Calling 使用）──────────────────────────────────

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "identify_material",
            "description": (
                "材料鉴定——将光谱与 130,000+ 参考光谱库匹配，返回排名最高的候选物质。"
                "适用场景: '这是什么材料?'、'鉴定这个样品'、'QC 批次检验'。"
                "这是最常用的工具，当用户想知道样品是什么物质时优先调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "description": "返回匹配数量（1-50）",
                        "default": 10,
                    },
                    "sample_type": {
                        "type": "string",
                        "description": "样品类型提示，如 polymer、pharmaceutical、mineral 等",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_peaks",
            "description": (
                "峰位解释——分析每个 FTIR 峰对应的化学键振动模式。"
                "适用场景: '1730 cm⁻¹ 是什么峰?'、'解释这些峰位'、'光谱解读'。"
                "当用户想理解峰的化学含义时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_type": {
                        "type": "string",
                        "description": "样品类型提示",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_functional_groups",
            "description": (
                "官能团归属——将 FTIR 峰位映射到对应的官能团（C=O、O-H、N-H 等）。"
                "适用场景: '有哪些官能团?'、'峰位对应哪些官能团?'、结构推断。"
                "当用户想知道样品含有哪些官能团时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_type": {
                        "type": "string",
                        "description": "样品类型提示",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_library_topk",
            "description": (
                "谱库 Top-K 快速匹配——快速返回排名最高的 K 个参考光谱，不做深度分析。"
                "适用场景: '找类似光谱'、'快速筛选'、'批量比对'。"
                "当用户只需要快速匹配结果而不需要详细解读时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "description": "返回匹配数量（1-50）",
                        "default": 10,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_public_results",
            "description": (
                "搜索公开分析结果——从已公开的 FTIR 分析案例中检索。"
                "适用场景: '有没有人分析过类似样品?'、'聚乙烯的公开分析'。"
                "当用户想参考他人的分析结果时调用。注意：此工具不需要光谱数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                },
                "required": ["query"],
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
    peaks: list[float] | None = None
    tool_calls_log: list[dict] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    search_results: list[dict] = field(default_factory=list)
    search_summary: str = ""
    synthesis: str = ""
    verification: dict = field(default_factory=dict)
    human_confirmed: bool = False
    final_report: str = ""
    conversation: list[dict] = field(default_factory=list)
    # 多轮推理度量
    react_iterations: int = 0
    verification_rounds: int = 0
    repair_count: int = 0
    evidence_conflicts: list[dict] = field(default_factory=list)
    confidence_trace: list[float] = field(default_factory=list)
    # 流式事件队列（SSE 端点消费）
    event_queue: queue.Queue = field(default_factory=queue.Queue)


class ChemSpectraAgent:
    """多工具自主路由 FTIR 分析 Agent。

    使用 Qwen Function Calling 让 LLM 自主决定调用哪些工具，
    实现 ReAct 循环而非固定流水线。
    """

    SYSTEM_PROMPT = """You are ChemSpectra, an expert AI agent for FTIR infrared spectral analysis.

You have access to multiple analysis tools via the FTIR.fun platform (130,000+ reference spectra).
Based on the user's request, YOU decide which tools to call and in what order.

AVAILABLE TOOLS:
- identify_material: Match spectrum against library, return ranked candidates. Use when user wants to know WHAT the material is.
- explain_peaks: Explain what each peak means chemically (bond vibrations). Use when user asks about specific peaks or wants spectral interpretation.
- assign_functional_groups: Map peaks to functional groups (C=O, O-H, N-H etc.). Use when user wants structural information.
- match_library_topk: Quick top-K library matches without deep analysis. Use for rapid screening.
- search_public_results: Search public analysis results. Use when user wants to reference prior analyses.

DECISION STRATEGY:
- For "identify this material" → call identify_material (primary), optionally explain_peaks for supporting evidence
- For "what are these peaks" → call explain_peaks, optionally assign_functional_groups
- For "what functional groups" → call assign_functional_groups, optionally explain_peaks
- For "deformulate / reverse engineer" → call identify_material + assign_functional_groups + explain_peaks (all three)
- For "QC check" → call identify_material + match_library_topk
- For quick screening → call match_library_topk alone
- You MAY call multiple tools in sequence to build a comprehensive analysis

RULES:
1. Always explain your chemical reasoning step by step
2. When confidence is below 0.8, explicitly flag for human review
3. Cite functional group evidence with wavenumber ranges (e.g. "1730 cm⁻¹ → C=O ester stretch")
4. Never fabricate CAS numbers or chemical names
5. For mixtures, explain which peaks belong to which component
6. After receiving tool results, synthesize them into a coherent analysis
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
        session.event_queue.put({"type": event_type, "data": data})

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    # ── Qwen API 调用 ─────────────────────────────────────────────────────────

    def _call_qwen(self, messages: list[dict], thinking: bool = True, **kwargs) -> str:
        """调用 Qwen（纯文本模式）。thinking=True 开启深度推理链。"""
        response = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_MODEL,
            messages=messages,
            result_format="message",
            enable_thinking=thinking,
            **kwargs,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen API error: {response.code} - {response.message}"
            )
        return response.output.choices[0].message.content

    def _call_qwen_stream(self, messages: list[dict], thinking: bool = True, **kwargs):
        """流式调用 Qwen，逐块 yield (type, text)。
        type: 'thinking' = reasoning_content, 'content' = 最终回答
        """
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
                raise RuntimeError(f"Qwen stream error: {chunk.code} - {chunk.message}")
            msg = chunk.output.choices[0].message
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                yield ("thinking", rc)
            if msg.content:
                yield ("content", msg.content)

    def _call_qwen_json(self, messages: list[dict], session: Session | None = None, **kwargs) -> dict:
        """调用 Qwen 并解析 JSON 响应，解析失败时 self-repair 重试。"""
        raw = self._call_qwen(messages, **kwargs)
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
        raw_retry = self._call_qwen(repair_messages, **kwargs)
        parsed_retry = _extract_json(raw_retry)
        if session:
            session.repair_count += 1
        if isinstance(parsed_retry, dict):
            return parsed_retry
        return {"raw_response": raw_retry}

    def _call_qwen_with_tools(self, messages: list[dict]) -> dict:
        """调用 Qwen（Function Calling 模式），返回完整的 choice 对象。"""
        response = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_MODEL,
            messages=messages,
            tools=AGENT_TOOLS,
            result_format="message",
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen API error: {response.code} - {response.message}"
            )
        return response.output.choices[0].message

    # ── 工具执行分发 ──────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_args: dict, session: Session) -> dict[str, Any]:
        """根据工具名称调用对应的 FTIR.fun API，返回结果。"""
        fb64 = session.file_base64
        fname = session.filename
        peaks = session.peaks

        if tool_name == "identify_material":
            return self.ftir.identify_material(
                file_base64=fb64, filename=fname, peaks=peaks,
                top_k=tool_args.get("top_k", 10),
                sample_type=tool_args.get("sample_type"),
            )

        elif tool_name == "explain_peaks":
            return self.ftir.explain_peaks(
                file_base64=fb64, filename=fname, peaks=peaks,
                sample_type=tool_args.get("sample_type"),
            )

        elif tool_name == "assign_functional_groups":
            return self.ftir.assign_functional_groups(
                file_base64=fb64, filename=fname, peaks=peaks,
                sample_type=tool_args.get("sample_type"),
            )

        elif tool_name == "match_library_topk":
            return self.ftir.match_library_topk(
                file_base64=fb64, filename=fname, peaks=peaks,
                top_k=tool_args.get("top_k", 10),
            )

        elif tool_name == "search_public_results":
            query = tool_args.get("query", session.user_input)
            return self.ftir.search_public_results(query)

        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def _format_tool_result_for_llm(self, tool_name: str, result: dict) -> str:
        """将工具调用结果格式化为 LLM 可读的文本摘要。"""
        if not result.get("success", True) and result.get("error"):
            return f"[{tool_name}] Error: {result['error']}"

        parts = [f"[{tool_name}] Results:"]

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
            spectrum_desc = f"Peak positions (cm⁻¹): {', '.join(str(p) for p in session.peaks)}"

        user_prompt = f"""Analyze this FTIR spectrum request.

User request: {session.user_input}
Sample context: {session.sample_context or 'Not provided'}
{spectrum_desc}

Decide which tools to call to best serve this request. You may call one or multiple tools.
After receiving all tool results, provide your final chemical analysis and synthesis.

IMPORTANT: The spectrum data is already loaded in the system — your tool calls will automatically
use it. You only need to specify additional parameters like top_k or sample_type.
"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        max_iterations = 6
        iteration = 0

        self._emit(session, "phase", {"phase": "ReAct", "label": "Qwen reasoning: selecting tools..."})

        while iteration < max_iterations:
            iteration += 1
            response_msg = self._call_qwen_with_tools(messages)

            tool_calls = getattr(response_msg, "tool_calls", None) or []
            if not tool_calls:
                # 最终综合分析——用流式调用让 thinking 可见
                self._emit(session, "phase", {"phase": "Synthesis", "label": "Synthesizing multi-tool results..."})
                synthesis_messages = list(messages) + [
                    {"role": "user", "content": "Now synthesize all tool results into a final chemical analysis."}
                ]
                synthesis_text = ""
                for chunk_type, chunk_text in self._call_qwen_stream(synthesis_messages, thinking=True):
                    if chunk_type == "thinking":
                        self._emit(session, "thinking", {"text": chunk_text})
                    else:
                        synthesis_text += chunk_text
                        self._emit(session, "synthesis_chunk", {"text": chunk_text})
                session.synthesis = synthesis_text or response_msg.content or ""
                session.step = "synthesized"
                break

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

                logger.info("Tool call [%d/%d]: %s(%s)", iteration, i + 1, tool_name, tool_args)
                self._emit(session, "tool_call", {"tool": tool_name, "iteration": iteration})

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
                if tool_name in ("identify_material", "match_library_topk"):
                    matches = result.get("matches", [])
                    if matches:
                        top_match = f"{matches[0].get('name','?')} (score={matches[0].get('similarity',0):.4f})"
                        if not session.search_results:
                            session.search_results = matches
                            session.search_summary = result.get("summary", "")
                self._emit(session, "tool_result", {
                    "tool": tool_name,
                    "n_matches": result.get("n_matches"),
                    "top_match": top_match,
                    "success": result.get("success", True),
                })

                result_text = self._format_tool_result_for_llm(tool_name, result)
                call_id = tc.get("id", f"call_{iteration}_{i}")
                messages.append({
                    "role": "tool",
                    "content": result_text,
                    "tool_call_id": call_id,
                })

        session.react_iterations = iteration

        confidence = self._estimate_confidence(session)
        conflicts = self._detect_evidence_conflicts(session)

        CONFIDENCE_THRESHOLD = 0.75
        needs_verification = (
            confidence < CONFIDENCE_THRESHOLD
            or len(conflicts) > 0
        )

        verification_result = None
        if needs_verification and session.synthesis:
            logger.info(
                "Auto-verification triggered: confidence=%.3f, conflicts=%d",
                confidence, len(conflicts),
            )
            self._emit(session, "verification_triggered", {
                "confidence": round(confidence, 4),
                "threshold": CONFIDENCE_THRESHOLD,
                "conflicts": len(conflicts),
                "label": f"Low confidence ({confidence:.0%}) — triggering self-verification round...",
            })
            all_issues = list(conflicts)
            if confidence < CONFIDENCE_THRESHOLD:
                all_issues.append({
                    "type": "low_confidence",
                    "confidence": round(confidence, 4),
                    "threshold": CONFIDENCE_THRESHOLD,
                    "severity": "high" if confidence < 0.5 else "medium",
                })

            verification_result = self._run_verification_round(session, all_issues)

            if verification_result.get("verification_synthesis"):
                session.synthesis = verification_result["verification_synthesis"]
                new_confidence = self._estimate_confidence(session)
                logger.info(
                    "Post-verification confidence: %.3f → %.3f",
                    confidence, new_confidence,
                )
                self._emit(session, "verification_done", {
                    "confidence_before": round(confidence, 4),
                    "confidence_after": round(new_confidence, 4),
                    "label": f"Verification complete: {confidence:.0%} → {new_confidence:.0%}",
                })

        return {
            "tools_called": [log["tool"] for log in session.tool_calls_log],
            "tool_details": session.tool_calls_log,
            "synthesis": session.synthesis,
            "metrics": {
                "react_iterations": session.react_iterations,
                "verification_rounds": session.verification_rounds,
                "repair_count": session.repair_count,
                "evidence_conflicts": len(session.evidence_conflicts),
                "confidence_trace": session.confidence_trace,
                "total_llm_calls": iteration + session.verification_rounds + session.repair_count,
            },
        }

    # ── 证据交叉验证 ────────────────────────────────────────────────────────

    def _detect_evidence_conflicts(self, session: Session) -> list[dict]:
        """检查多工具结果之间的矛盾。

        比对 identify_material、explain_peaks、assign_functional_groups 三个工具
        的输出，检测逻辑矛盾（如：鉴定结果说是 PET，但官能团未检出酯基）。
        """
        conflicts = []
        id_result = session.tool_results.get("identify_material", {})
        fg_result = session.tool_results.get("assign_functional_groups", {})
        pe_result = session.tool_results.get("explain_peaks", {})

        if not id_result or (not fg_result and not pe_result):
            return conflicts

        id_matches = id_result.get("matches", [])
        top_match_name = id_matches[0].get("name", "").lower() if id_matches else ""
        top_score = id_matches[0].get("similarity", 0) if id_matches else 0

        fg_evidence = fg_result.get("evidence", [])
        fg_text = " ".join(str(e) for e in fg_evidence).lower()
        pe_text = " ".join(str(e) for e in pe_result.get("peak_explanations", [])).lower()

        expected_groups = {
            "pet": ["ester", "c=o", "aromatic"],
            "polyethylene": ["c-h", "ch2", "methylene"],
            "polypropylene": ["c-h", "ch3", "methyl"],
            "nylon": ["amide", "n-h", "c=o"],
            "polystyrene": ["aromatic", "c-h", "benzene"],
            "cellulose": ["o-h", "c-o", "hydroxyl"],
            "silicone": ["si-o", "si-c", "siloxane"],
        }

        all_evidence_text = fg_text + " " + pe_text
        for material_key, groups in expected_groups.items():
            if material_key in top_match_name:
                missing = [g for g in groups if g not in all_evidence_text]
                if len(missing) >= 2:
                    conflicts.append({
                        "type": "functional_group_mismatch",
                        "material": top_match_name,
                        "expected_groups": groups,
                        "missing_groups": missing,
                        "severity": "high" if top_score > 0.8 else "medium",
                    })
                break

        if id_matches and len(id_matches) >= 2:
            scores = [m.get("similarity", 0) for m in id_matches[:3]]
            if scores[0] > 0 and scores[1] > 0:
                gap = scores[0] - scores[1]
                if gap < 0.05 and scores[0] > 0.7:
                    conflicts.append({
                        "type": "ambiguous_top_candidates",
                        "candidate_1": id_matches[0].get("name", "?"),
                        "candidate_2": id_matches[1].get("name", "?"),
                        "score_gap": round(gap, 4),
                        "severity": "medium",
                    })

        session.evidence_conflicts = conflicts
        return conflicts

    # ── 自主验证轮 ────────────────────────────────────────────────────────

    def _run_verification_round(self, session: Session, conflicts: list[dict]) -> dict:
        """当证据矛盾或低置信度时，自动启动验证轮。

        Agent 被告知其先前分析中的具体问题，然后自主决定调用更多工具来验证。
        """
        session.verification_rounds += 1
        logger.info("Verification round %d triggered, %d conflicts detected",
                     session.verification_rounds, len(conflicts))

        conflict_desc = "\n".join(
            f"- {c['type']}: {json.dumps(c, ensure_ascii=False)}"
            for c in conflicts
        )

        spectrum_desc = ""
        if session.peaks:
            spectrum_desc = f"Peak positions (cm⁻¹): {', '.join(str(p) for p in session.peaks)}"
        elif session.file_base64:
            spectrum_desc = f"Spectrum file: {session.filename}"

        prior_tools = ", ".join(log["tool"] for log in session.tool_calls_log)
        prior_synthesis = session.synthesis[:800] if session.synthesis else "No synthesis yet"

        verification_prompt = f"""You previously analyzed this FTIR spectrum and produced a synthesis, but
automated cross-validation detected potential issues that need investigation.

{spectrum_desc}

Prior analysis tools used: {prior_tools}
Prior synthesis: {prior_synthesis}

ISSUES DETECTED:
{conflict_desc}

Your task: Investigate these issues by calling additional tools if needed. Then produce an
UPDATED synthesis that addresses each issue. If the original analysis was correct, explain why.
If it was wrong, correct it.

Focus on resolving the specific conflicts above. Call tools that can provide clarifying evidence.
"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": verification_prompt},
        ]

        response_msg = self._call_qwen_with_tools(messages)
        tool_calls = getattr(response_msg, "tool_calls", None) or []

        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": response_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.get("id", f"verify_{i}"),
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
                tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

                logger.info("Verification tool call: %s(%s)", tool_name, tool_args)
                self._emit(session, "tool_call", {"tool": tool_name, "iteration": "verify", "phase": "verification"})
                result = self._execute_tool(tool_name, tool_args, session)

                session.tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "success": result.get("success", True),
                    "n_matches": result.get("n_matches"),
                    "goal": result.get("task_context", {}).get("goal"),
                    "phase": "verification",
                })
                session.tool_results[f"{tool_name}_verify"] = result
                self._emit(session, "tool_result", {
                    "tool": tool_name, "phase": "verification",
                    "success": result.get("success", True),
                    "n_matches": result.get("n_matches"),
                })

                result_text = self._format_tool_result_for_llm(tool_name, result)
                messages.append({
                    "role": "tool",
                    "content": result_text,
                    "tool_call_id": tc.get("id", f"verify_{i}"),
                })

            # 验证综合分析用流式，让 thinking 可见
            self._emit(session, "phase", {"phase": "Verification synthesis", "label": "Re-synthesizing with verification evidence..."})
            verification_synthesis = ""
            for chunk_type, chunk_text in self._call_qwen_stream(messages, thinking=True):
                if chunk_type == "thinking":
                    self._emit(session, "thinking", {"text": chunk_text, "phase": "verification"})
                else:
                    verification_synthesis += chunk_text
        else:
            verification_synthesis = response_msg.content or session.synthesis

        return {
            "verification_synthesis": verification_synthesis,
            "tools_called": [tc["function"]["name"] for tc in tool_calls] if tool_calls else [],
            "conflicts_addressed": len(conflicts),
        }

    # ── 置信度评估 ────────────────────────────────────────────────────────

    def _estimate_confidence(self, session: Session) -> float:
        """从工具结果估算当前分析置信度。"""
        scores = []

        id_result = session.tool_results.get("identify_material", {})
        if id_result.get("matches"):
            top_sim = id_result["matches"][0].get("similarity", 0)
            scores.append(top_sim)
            if len(id_result["matches"]) >= 2:
                gap = top_sim - id_result["matches"][1].get("similarity", 0)
                scores.append(min(1.0, gap * 5))

        if id_result.get("confidence"):
            scores.append(id_result["confidence"])

        fg_result = session.tool_results.get("assign_functional_groups", {})
        if fg_result.get("evidence"):
            n_groups = len(fg_result["evidence"])
            scores.append(min(1.0, n_groups / 5))

        if not scores:
            return 0.5

        confidence = sum(scores) / len(scores)
        session.confidence_trace.append(round(confidence, 4))
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
                "confidence_adjusted": 0,
                "flags": ["No matches"],
            }
            session.verification = v
            return v

        top5 = session.search_results[:5]
        matches_str = "\n".join(
            f"  #{m.get('rank', i+1)}: {m.get('name', '?')} "
            f"(CAS: {m.get('cas', 'N/A')}) "
            f"score={m.get('similarity') or m.get('score', 0):.4f}"
            for i, m in enumerate(top5)
        ) if top5 else "No library matches available."

        tools_called = ", ".join(log["tool"] for log in session.tool_calls_log)

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

Sample context: {session.sample_context or 'Not provided'}

Return ONLY valid JSON, no markdown:
{{"verdict": "confirmed|needs_review|rejected",
 "reasoning": "Your chemical analysis in 2-3 sentences, citing specific evidence from tool results",
 "top_candidate": "name of best match or null",
 "confidence_adjusted": 0.0 to 1.0,
 "flags": ["list of concerns if any"]}}
""",
            },
        ]
        v = self._call_qwen_json(messages, session=session)
        if "verdict" not in v:
            v["verdict"] = "needs_review"
        if "reasoning" not in v:
            v["reasoning"] = v.get("raw_response", session.synthesis or "Analysis completed.")
        session.verification = v
        return v

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

        return {
            "best_match": {
                "name": top.get("name", "Unknown"),
                "cas": top.get("cas", "N/A"),
                "score": top.get("similarity") or top.get("score", 0),
            },
            "verdict": v.get("verdict", "needs_review"),
            "reasoning": v.get("reasoning", ""),
            "confidence": v.get("confidence_adjusted", 0),
            "flags": v.get("flags", []),
            "candidates": candidates,
            "search_summary": session.search_summary,
            "tools_called": [log["tool"] for log in session.tool_calls_log],
            "synthesis": session.synthesis,
        }

    def handle_followup(self, session: Session, user_message: str) -> dict:
        """处理用户在确认环节的追问。"""
        session.conversation.append({"role": "user", "content": user_message})

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
- Tools used: {tools_used}
- Your synthesis: {session.synthesis[:500] if session.synthesis else 'N/A'}
- Your previous reasoning: {v.get('reasoning', 'N/A')}

Conversation so far:
{conv_history}

The user just said: "{user_message}"

Respond helpfully. Return ONLY valid JSON:
{{"response": "your helpful answer to the user's question or comment",
 "action": "none|update_context|re_search",
 "updated_context": "if action is update_context, the new combined context"}}
""",
            },
        ]
        result = self._call_qwen_json(messages, session=session)
        response_text = result.get("response", result.get("raw_response", "I can help with that."))
        session.conversation.append({"role": "assistant", "content": response_text})
        return result

    # ── 报告生成 ──────────────────────────────────────────────────────────────

    def generate_report(self, session: Session) -> str:
        """人类确认后生成最终分析报告。"""
        session.step = "reporting"
        session.human_confirmed = True

        top = session.search_results[0] if session.search_results else {}
        v = session.verification
        tools_used = ", ".join(log["tool"] for log in session.tool_calls_log) or "none"

        peak_info = ""
        if "explain_peaks" in session.tool_results:
            pe_result = session.tool_results["explain_peaks"]
            if pe_result.get("peak_explanations"):
                peak_info = f"\nPeak explanations from tool: {json.dumps(pe_result['peak_explanations'][:10], ensure_ascii=False)}"

        fg_info = ""
        if "assign_functional_groups" in session.tool_results:
            fg_result = session.tool_results["assign_functional_groups"]
            if fg_result.get("evidence"):
                fg_info = f"\nFunctional group evidence: {json.dumps(fg_result['evidence'][:10], ensure_ascii=False)}"

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
Flags: {', '.join(v.get('flags', [])) or 'None'}
{peak_info}
{fg_info}

Include these sections:
1. **Sample Information** — what was analyzed
2. **Analysis Method** — multi-tool FTIR analysis (list which tools were used and why)
3. **Results Summary** — top match, score, and verdict
4. **Chemical Reasoning** — synthesize evidence from all tools used
5. **Peak Analysis** — if peak explanations available, include detailed peak assignments
6. **Quality Notes** — flags, confidence assessment, limitations
7. **Analyst Confirmation** — note that human-in-the-loop review was performed

Keep it professional but concise. Use wavenumber evidence where possible.
""",
            },
        ]
        session.final_report = self._call_qwen(messages)
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

        if not file_base64 and not peaks:
            session.step = "clarifying"
            return {
                "step": "needs_clarification",
                "session_id": session.session_id,
                "question": (
                    "Please provide a spectrum file or peak positions (cm⁻¹) to analyze. "
                    "For example, upload a .spc/.csv/.jdx file or enter peaks like: 2920, 2850, 1460, 720"
                ),
            }

        tool_result = self.run_tool_loop(session)
        self.extract_verification(session)
        confirmation = self.build_confirmation_payload(session)

        return {
            "step": "awaiting_confirmation",
            "session_id": session.session_id,
            "tools_called": tool_result["tools_called"],
            "n_tools": len(tool_result["tools_called"]),
            "search_summary": session.search_summary,
            "n_matches": len(session.search_results),
            "confirmation": confirmation,
            "agent_metrics": tool_result.get("metrics", {}),
        }
