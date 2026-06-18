"""
ChemSpectra Agent — AI Autopilot for FTIR Spectral Analysis.
Track 4: Autopilot Agent — Qwen Cloud Hackathon.

Uses Qwen-Max (dashscope) as the reasoning brain + FTIR.fun MCP tools
as the domain engine. Orchestrates the full analysis pipeline with
human-in-the-loop checkpoints at critical decision points.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

# ── Alibaba Cloud: dashscope SDK for Qwen API ──
# PROOF OF ALIBABA CLOUD DEPLOYMENT: This is the primary Alibaba Cloud
# service used. Qwen-Max model via dashscope provides the agent's
# chemical reasoning, intent parsing, and report generation.
import dashscope  # Alibaba Cloud Qwen API SDK
from dashscope import Generation
from openai import OpenAI

from tools import FtirfunMcpClient

logger = logging.getLogger(__name__)

# ── Configuration ──
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
FTIRFUN_API_KEY = os.environ.get("FTIRFUN_API_KEY", "")
QWEN_MODEL = "qwen-max"


@dataclass
class AgentState:
    """Tracks the agent's progress through the analysis pipeline."""
    step: str = "idle"
    user_input: str = ""
    sample_context: str = ""
    spectrum_data: dict | None = None
    search_results: list[dict] = field(default_factory=list)
    chemical_reasoning: str = ""
    human_confirmed: bool = False
    final_report: str = ""


class ChemSpectraAgent:
    """Orchestrates FTIR spectral analysis with Qwen reasoning + MCP tools."""

    SYSTEM_PROMPT = """You are ChemSpectra, an expert AI agent for FTIR infrared
spectral analysis. Your role is to guide the analysis pipeline, interpret
chemical data, and present findings clearly.

CAPABILITIES:
- Parse user intent from natural language and spectrum data
- Call FTIR.fun MCP tools for spectral library search
- Perform chemical reasoning on match results
- Flag ambiguous results for human review
- Generate structured analysis reports

RULES:
1. Always explain your chemical reasoning, not just results
2. When confidence is below 0.8, explicitly flag for human review
3. Cite functional group evidence with wavenumber ranges
4. Never fabricate CAS numbers or chemical names
5. For mixtures, explain which peaks belong to which component
"""

    def __init__(self):
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY environment variable required")
        self.mcp = FtirfunMcpClient(api_key=FTIRFUN_API_KEY)
        self.state = AgentState()

    # ── Qwen LLM call via Alibaba Cloud dashscope ──
    def _call_qwen(self, messages: list[dict], **kwargs) -> str:
        """Call Qwen-Max via Alibaba Cloud dashscope SDK."""
        response = Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_MODEL,
            messages=messages,
            result_format="message",
            **kwargs,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen API error: {response.code} - {response.message}"
            )
        return response.output.choices[0].message.content

    # ── Step 1: Parse user intent ──
    def parse_intent(self, user_input: str, sample_context: str = "") -> dict:
        """Use Qwen to extract structured analysis parameters from user input."""
        self.state.step = "parsing"
        self.state.user_input = user_input
        self.state.sample_context = sample_context

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Parse this FTIR analysis request into structured parameters.

User input: {user_input}
Sample context: {sample_context or 'Not provided'}

Return JSON with:
- analysis_type: "identify" | "compare" | "qc_check" | "deformulate"
- sample_description: brief description of the sample
- expected_material: what the user thinks it might be (or null)
- peak_positions: list of wavenumbers if mentioned
- special_concerns: any specific concerns (contamination, degradation, etc.)
""",
            },
        ]
        result = self._call_qwen(messages)
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"analysis_type": "identify", "sample_description": user_input}

    # ── Step 2: Spectral library search via MCP ──
    def search_library(
        self, file_base64: str | None = None, filename: str = "spectrum.0",
        peaks: list[float] | None = None, query: str | None = None,
    ) -> list[dict]:
        """Call FTIR.fun MCP tool for spectral library search."""
        self.state.step = "searching"

        result = self.mcp.analyze_ftir_spectrum(
            file_base64=file_base64,
            filename=filename,
            peaks=peaks,
            query=query,
            top_k=5,
        )
        self.state.search_results = result.get("matches", [])
        return self.state.search_results

    # ── Step 3: AI chemical verification ──
    def verify_results(self) -> dict:
        """Qwen reviews search results for chemical consistency."""
        self.state.step = "verifying"

        if not self.state.search_results:
            return {"verdict": "no_results", "reasoning": "No spectral matches found."}

        matches_json = json.dumps(self.state.search_results, ensure_ascii=False)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Review these FTIR spectral library matches for chemical consistency.

Sample context: {self.state.sample_context}
Top matches: {matches_json}

Perform chemical reasoning:
1. Are the functional groups consistent with the proposed material?
2. Are there unexplained peaks suggesting a mixture?
3. Is the confidence score reliable based on peak assignments?
4. Should this result be flagged for human review?

Return JSON:
- verdict: "confirmed" | "needs_review" | "rejected"
- reasoning: your chemical analysis (2-3 sentences)
- top_candidate: name of the best match
- confidence_adjusted: your adjusted confidence (0-1)
- flags: list of concerns if any
""",
            },
        ]
        result = self._call_qwen(messages)
        try:
            verification = json.loads(result)
        except json.JSONDecodeError:
            verification = {"verdict": "needs_review", "reasoning": result}
        self.state.chemical_reasoning = verification.get("reasoning", "")
        return verification

    # ── Step 4: Human-in-the-loop checkpoint ──
    def build_confirmation_prompt(self, verification: dict) -> str:
        """Build the human-in-the-loop confirmation message."""
        top = self.state.search_results[0] if self.state.search_results else {}
        return f"""## FTIR Analysis Results

**Best Match:** {top.get('name', 'Unknown')}
**CAS:** {top.get('cas', 'N/A')}
**Similarity:** {top.get('score', 0):.1%}

**AI Chemical Review:** {verification.get('reasoning', 'N/A')}
**Verdict:** {verification.get('verdict', 'needs_review')}

### Top Candidates:
{self._format_candidates()}

---
**Confirm this identification?**
- ✅ Accept — generate final report
- 🔄 Request alternative — search with different parameters
- 📝 Add context — provide additional sample information
"""

    def _format_candidates(self) -> str:
        lines = []
        for i, m in enumerate(self.state.search_results[:5], 1):
            lines.append(
                f"{i}. {m.get('name', '?')} — score: {m.get('score', 0):.3f}"
                + (f" (CAS: {m.get('cas', '')})" if m.get('cas') else "")
            )
        return "\n".join(lines)

    # ── Step 5: Generate final report ──
    def generate_report(self) -> str:
        """Generate the final analysis report after human confirmation."""
        self.state.step = "reporting"
        self.state.human_confirmed = True

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Generate a professional FTIR analysis report.

Sample: {self.state.sample_context or self.state.user_input}
Confirmed match: {json.dumps(self.state.search_results[0] if self.state.search_results else {}, ensure_ascii=False)}
Chemical reasoning: {self.state.chemical_reasoning}

Format as a structured report with:
1. Sample Information
2. Analysis Method
3. Results Summary
4. Spectral Match Details (with DOIs if available)
5. Chemical Reasoning
6. Quality Control Notes
7. Analyst Confirmation (human-in-the-loop)
""",
            },
        ]
        self.state.final_report = self._call_qwen(messages)
        return self.state.final_report

    # ── Full pipeline ──
    def run_pipeline(
        self,
        user_input: str,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        sample_context: str = "",
    ) -> dict:
        """Run the complete analysis pipeline and return structured results."""
        intent = self.parse_intent(user_input, sample_context)
        matches = self.search_library(
            file_base64=file_base64, filename=filename,
            peaks=peaks, query=user_input,
        )
        verification = self.verify_results()
        confirmation = self.build_confirmation_prompt(verification)

        return {
            "intent": intent,
            "matches": matches,
            "verification": verification,
            "confirmation_prompt": confirmation,
            "state": self.state.step,
        }
