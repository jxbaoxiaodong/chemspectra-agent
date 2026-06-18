"""
FTIR.fun MCP Client — wraps the FTIR.fun MCP Server tools
for use by the Qwen Agent.

Provides a clean Python interface to the FTIR.fun spectral
analysis tools exposed via MCP (Model Context Protocol).
"""

from __future__ import annotations

import base64
import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

FTIRFUN_MCP_URL = os.environ.get(
    "FTIRFUN_MCP_URL", "https://ftir.fun/mcp"
)
FTIRFUN_API_KEY = os.environ.get("FTIRFUN_API_KEY", "")


class FtirfunMcpClient:
    """Client for FTIR.fun MCP Server tools."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or FTIRFUN_API_KEY
        self.base_url = FTIRFUN_MCP_URL
        self._session_id: str | None = None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool via Streamable HTTP transport."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
            "id": 1,
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    self.base_url,
                    json=payload,
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()

            if "error" in data:
                return {"success": False, "error": data["error"]}

            result = data.get("result", {})
            content = result.get("content", [])

            # MCP returns content as a list of text/embedded items
            for item in content:
                if item.get("type") == "text":
                    try:
                        return json.loads(item["text"])
                    except json.JSONDecodeError:
                        return {"success": True, "raw_text": item["text"]}

            return {"success": True, "raw_result": result}

        except Exception as e:
            logger.error("MCP tool call failed: %s", e)
            return {"success": False, "error": str(e)}

    def analyze_ftir_spectrum(
        self,
        peaks: list[float] | None = None,
        query: str | None = None,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        top_k: int = 5,
        tolerance_cm1: int = 4,
    ) -> dict:
        """
        Search the FTIR.fun spectral library (130K+ spectra).

        Args:
            peaks: FTIR peak positions in cm^-1
            query: Natural language query with peak positions or context
            file_base64: Base64-encoded spectrum file
            filename: Original filename for format detection
            top_k: Number of candidates to return (1-20)
            tolerance_cm1: Peak matching tolerance in cm^-1 (1-10)

        Returns:
            dict with keys: success, matches, peak_explanations, confidence
        """
        args = {
            "top_k": top_k,
            "tolerance_cm1": tolerance_cm1,
            "filename": filename,
        }
        if peaks:
            args["peaks"] = peaks
        if query:
            args["query"] = query
        if file_base64:
            args["file_base64"] = file_base64

        return self._call_tool("analyze_ftir_spectrum", args)

    def search_public_results(self, query: str) -> dict:
        """Search publicly shared FTIR analysis results."""
        return self._call_tool("search", {"query": query})

    def fetch_result(self, result_id: int) -> dict:
        """Fetch a specific public FTIR analysis result by ID."""
        return self._call_tool("fetch", {"id": result_id})

    @staticmethod
    def encode_file(filepath: str) -> tuple[str, str]:
        """Read and base64-encode a spectrum file.

        Returns:
            (base64_string, filename)
        """
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return data, filename
