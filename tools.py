"""
FTIR.fun API Client — 多工具封装，供 Qwen Agent 自主选择调用。

工具集:
  - analyze_spectrum: 全流程光谱分析（物质鉴定 + 峰解释 + 官能团归属）
  - identify_material: 材料鉴定（专注于匹配结果排序）
  - explain_peaks: 峰位解释（每个峰对应什么化学键振动）
  - assign_functional_groups: 官能团归属（峰位→官能团映射）
  - match_library_topk: 谱库 Top-K 匹配（快速比对）
  - search_public_results: 搜索公开的分析结果

Auth: X-API-Key header.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

FTIRFUN_API_URL = os.environ.get(
    "FTIRFUN_API_URL", "http://127.0.0.1:18080"
)
FTIRFUN_API_KEY = os.environ.get("FTIRFUN_API_KEY", "")


class FtirfunClient:
    """Client for FTIR.fun REST API."""

    def __init__(self, api_url: str = "", api_key: str = ""):
        self.api_url = (api_url or FTIRFUN_API_URL).rstrip("/")
        self.api_key = api_key or FTIRFUN_API_KEY

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    def _build_body(
        self,
        *,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        top_k: int = 5,
        tolerance_cm1: int = 8,
        goal: str | None = None,
        sample_type: str | None = None,
    ) -> dict[str, Any] | None:
        """构建 CanonicalFtirRequest 请求体。返回 None 表示缺少输入。"""
        body: dict[str, Any] = {
            "spectrum": {"type": "ftir"},
            "options": {"top_k": top_k, "tolerance_cm1": tolerance_cm1},
        }
        if file_base64:
            body["file_base64"] = file_base64
            body["filename"] = filename
        elif peaks:
            body["spectrum"]["peaks"] = [float(p) for p in peaks]
        else:
            return None
        if goal or sample_type:
            body["task_context"] = {}
            if goal:
                body["task_context"]["goal"] = goal
            if sample_type:
                body["task_context"]["sample_type"] = sample_type
        return body

    def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        """向 FTIR.fun REST API 发送 POST 请求并返回 JSON。"""
        try:
            # 显式绕过代理访问本地 API（httpx >= 0.23 用 mounts）
            no_proxy = httpx.HTTPTransport()
            with httpx.Client(
                timeout=120.0,
                mounts={"http://127.0.0.1": no_proxy, "http://localhost": no_proxy},
            ) as client:
                resp = client.post(
                    f"{self.api_url}{endpoint}",
                    json=body,
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("API HTTP error: %s %s", e.response.status_code, e.response.text[:200])
            return {"success": False, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error("API call failed: %s", e)
            return {"success": False, "error": str(e)}

    def analyze_spectrum(
        self,
        *,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        query: str | None = None,
        top_k: int = 5,
        tolerance_cm1: int = 8,
    ) -> dict[str, Any]:
        """全流程光谱分析——物质鉴定 + 峰解释 + 官能团归属。"""
        body = self._build_body(
            file_base64=file_base64, filename=filename, peaks=peaks,
            top_k=top_k, tolerance_cm1=tolerance_cm1,
        )
        if body is None:
            return {"success": False, "error": "Either file_base64 or peaks must be provided"}
        return self._post("/ftir/analyze_spectrum", body)

    def identify_material(
        self,
        *,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        top_k: int = 10,
        sample_type: str | None = None,
    ) -> dict[str, Any]:
        """材料鉴定——聚焦于谱库匹配排序和物质识别。"""
        body = self._build_body(
            file_base64=file_base64, filename=filename, peaks=peaks,
            top_k=top_k, goal="identification", sample_type=sample_type,
        )
        if body is None:
            return {"success": False, "error": "Either file_base64 or peaks must be provided"}
        return self._post("/ftir/identify_material", body)

    def explain_peaks(
        self,
        *,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        sample_type: str | None = None,
    ) -> dict[str, Any]:
        """峰位解释——每个峰对应什么化学键振动模式。"""
        body = self._build_body(
            file_base64=file_base64, filename=filename, peaks=peaks,
            top_k=5, goal="explanation", sample_type=sample_type,
        )
        if body is None:
            return {"success": False, "error": "Either file_base64 or peaks must be provided"}
        return self._post("/ftir/explain_peaks", body)

    def assign_functional_groups(
        self,
        *,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        sample_type: str | None = None,
    ) -> dict[str, Any]:
        """官能团归属——将峰位映射到对应的官能团。"""
        body = self._build_body(
            file_base64=file_base64, filename=filename, peaks=peaks,
            top_k=5, goal="functional_groups", sample_type=sample_type,
        )
        if body is None:
            return {"success": False, "error": "Either file_base64 or peaks must be provided"}
        return self._post("/ftir/assign_functional_groups", body)

    def match_library_topk(
        self,
        *,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """谱库 Top-K 快速匹配——返回排名前 K 的参考光谱。"""
        body = self._build_body(
            file_base64=file_base64, filename=filename, peaks=peaks,
            top_k=top_k, goal="matching",
        )
        if body is None:
            return {"success": False, "error": "Either file_base64 or peaks must be provided"}
        return self._post("/ftir/match_library_topk", body)

    def search_public_results(self, query: str) -> dict[str, Any]:
        """Search publicly shared FTIR analysis results (via MCP search tool)."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": query}},
            "id": 1,
        }
        try:
            no_proxy = httpx.HTTPTransport()
            with httpx.Client(
                timeout=30.0,
                mounts={"http://127.0.0.1": no_proxy, "http://localhost": no_proxy},
            ) as client:
                resp = client.post(
                    f"{self.api_url.replace(':18080', ':18081')}/mcp",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                result = data.get("result", {})
                for item in result.get("content", []):
                    if item.get("type") == "text":
                        try:
                            return json.loads(item["text"])
                        except json.JSONDecodeError:
                            return {"success": True, "raw_text": item["text"]}
                return {"success": True, "raw_result": result}
        except Exception as e:
            logger.error("MCP search failed: %s", e)
            return {"success": False, "error": str(e)}

    @staticmethod
    def encode_file(filepath: str) -> tuple[str, str]:
        """Read and base64-encode a spectrum file. Returns (base64_string, filename)."""
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return data, filename
