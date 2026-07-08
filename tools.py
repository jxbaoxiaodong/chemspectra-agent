"""
FTIR.fun client for the active 3-tool ChemSpectra architecture.

Active external tools:
- search_library: one REST call to /ftir/identify_material
- search_public_cases: MCP public-case search

The removed 5-tool/verification implementation is archived under
archive/legacy_5tool_verification_20260702/.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any

import httpx

from audit import audit_logger

logger = logging.getLogger(__name__)

FTIRFUN_API_URL = os.environ.get("FTIRFUN_API_URL", "http://127.0.0.1:18080")
FTIRFUN_API_KEY = os.environ.get("FTIRFUN_API_KEY", "")

# Source: IMPROVEMENT_PLAN.md section 2 requires search_library default top_k=15.
DEFAULT_LIBRARY_TOP_K = 15
DEFAULT_TOLERANCE_CM1 = 8
REST_TIMEOUT_SECONDS = 120.0
MCP_TIMEOUT_SECONDS = 30.0
# Source: fastapi_server/mcp_server.py::find_spectra default limit.
DEFAULT_PUBLIC_CASE_LIMIT = 10


class FtirfunClient:
    """Minimal client for FTIR.fun REST and MCP endpoints."""

    def __init__(self, api_url: str = "", api_key: str = ""):
        self.api_url = (api_url or FTIRFUN_API_URL).rstrip("/")
        self.api_key = api_key or FTIRFUN_API_KEY

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    @staticmethod
    def _sanitize_request_body(body: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(body)
        file_b64 = sanitized.pop("file_base64", None)
        if file_b64 is not None:
            sanitized["file_base64_present"] = True
            sanitized["file_base64_length"] = len(file_b64)
        return sanitized

    def _build_library_body(
        self,
        *,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        top_k: int = DEFAULT_LIBRARY_TOP_K,
        tolerance_cm1: int = DEFAULT_TOLERANCE_CM1,
        sample_type: str | None = None,
    ) -> dict[str, Any] | None:
        body: dict[str, Any] = {
            "spectrum": {"type": "ftir"},
            "options": {"top_k": int(top_k), "tolerance_cm1": int(tolerance_cm1)},
            "task_context": {"goal": "identification"},
        }
        if file_base64:
            body["file_base64"] = file_base64
            body["filename"] = filename
        elif peaks:
            body["spectrum"]["peaks"] = [float(item) for item in peaks]
        else:
            return None
        if sample_type:
            body["task_context"]["sample_type"] = sample_type
        return body

    def _post_json(
        self,
        endpoint: str,
        body: dict[str, Any],
        *,
        trace: dict[str, Any] | None = None,
        action: str = "post_json",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        session_id = (trace or {}).get("session_id")
        request_payload = {
            "api_url": self.api_url,
            "endpoint": endpoint,
            "method": "POST",
            "request_body": self._sanitize_request_body(body),
            "trace": trace or {},
        }
        try:
            no_proxy = httpx.HTTPTransport()
            with httpx.Client(
                timeout=REST_TIMEOUT_SECONDS,
                mounts={"http://127.0.0.1": no_proxy, "http://localhost": no_proxy},
            ) as client:
                response = client.post(
                    f"{self.api_url}{endpoint}",
                    json=body,
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()
                audit_logger.write_event(
                    category="external_api",
                    action=action,
                    session_id=session_id,
                    payload={
                        **request_payload,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                        "status_code": response.status_code,
                        "response_json": data,
                    },
                )
                return data
        except httpx.HTTPStatusError as exc:
            logger.error("FTIR.fun HTTP error: %s %s", exc.response.status_code, exc.response.text[:200])
            audit_logger.write_event(
                category="external_api",
                action=f"{action}_http_error",
                session_id=session_id,
                payload={
                    **request_payload,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "status_code": exc.response.status_code,
                    "response_text": exc.response.text,
                },
            )
            return {"success": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"}
        except Exception as exc:
            logger.error("FTIR.fun call failed: %s", exc)
            audit_logger.write_event(
                category="external_api",
                action=f"{action}_error",
                session_id=session_id,
                payload={
                    **request_payload,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": str(exc),
                },
            )
            return {"success": False, "error": str(exc)}

    def search_library(
        self,
        *,
        file_base64: str | None = None,
        filename: str = "spectrum.0",
        peaks: list[float] | None = None,
        top_k: int = DEFAULT_LIBRARY_TOP_K,
        sample_type: str | None = None,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search the FTIR.fun library once and return the full API result."""
        body = self._build_library_body(
            file_base64=file_base64,
            filename=filename,
            peaks=peaks,
            top_k=top_k,
            sample_type=sample_type,
        )
        if body is None:
            return {"success": False, "error": "Either file_base64 or peaks must be provided"}
        return self._post_json(
            "/ftir/identify_material",
            body,
            trace=trace,
            action="search_library",
        )

    def search_public_cases(self, query: str, trace: dict[str, Any] | None = None) -> dict[str, Any]:
        """Search publicly shared FTIR analysis cases through FTIR.fun MCP."""
        if not self.api_key:
            return {"success": False, "error": "FTIRFUN_API_KEY is required for MCP search_public_cases"}
        started = time.perf_counter()
        session_id = (trace or {}).get("session_id")
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "find_spectra",
                "arguments": {"query": query, "limit": DEFAULT_PUBLIC_CASE_LIMIT},
            },
            "id": 1,
        }
        try:
            no_proxy = httpx.HTTPTransport()
            with httpx.Client(
                timeout=MCP_TIMEOUT_SECONDS,
                mounts={"http://127.0.0.1": no_proxy, "http://localhost": no_proxy},
            ) as client:
                response = client.post(
                    f"{self.api_url.replace(':18080', ':18081')}/mcp",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error("FTIR.fun MCP search failed: %s", exc)
            audit_logger.write_event(
                category="external_api",
                action="search_public_cases_error",
                session_id=session_id,
                payload={
                    "api_url": self.api_url.replace(":18080", ":18081"),
                    "endpoint": "/mcp",
                    "method": "POST",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "request_json": payload,
                    "trace": trace or {},
                    "error": str(exc),
                },
            )
            return {"success": False, "error": str(exc)}

        audit_logger.write_event(
            category="external_api",
            action="search_public_cases",
            session_id=session_id,
            payload={
                "api_url": self.api_url.replace(":18080", ":18081"),
                "endpoint": "/mcp",
                "method": "POST",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "request_json": payload,
                "trace": trace or {},
                "response_json": data,
            },
        )

        result = data.get("result", {})
        for item in result.get("content", []):
            if item.get("type") != "text":
                continue
            text = item.get("text", "")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {"success": True, "raw_text": text}
            return parsed if isinstance(parsed, dict) else {"success": True, "raw_result": parsed}
        return {"success": True, "raw_result": result}

    def explain_peaks(
        self,
        peaks: list[float],
        query: str | None = None,
        sampling_mode: str | None = None,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Explain FTIR peak positions via /ftir/explain_peak_assignments.

        Source: fastapi_server/api.py ExplainPeaksRequest. This endpoint does
        NOT perform a library search — it only returns functional-group
        explanations for the given wavenumber positions.
        """
        body: dict[str, Any] = {"peaks": [float(p) for p in peaks]}
        if query:
            body["query"] = query
        if sampling_mode:
            body["sampling_mode"] = sampling_mode
        return self._post_json(
            "/ftir/explain_peak_assignments",
            body,
            trace=trace,
            action="explain_peaks",
        )

    def parse_spectrum(
        self,
        file_base64: str,
        filename: str,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Parse a spectrum file via /parse-spectrum.

        Source: fastapi_server/api.py. Returns peaks (cm-1) and spectrum
        as list of {"x": wavenumber, "y": intensity} points.
        """
        body: dict[str, Any] = {
            "file_base64": file_base64,
            "filename": filename,
        }
        return self._post_json(
            "/parse-spectrum",
            body,
            trace=trace,
            action="parse_spectrum",
        )

    @staticmethod
    def encode_file(filepath: str) -> tuple[str, str]:
        """Read and base64-encode a spectrum file."""
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as handle:
            data = base64.b64encode(handle.read()).decode("ascii")
        return data, filename
