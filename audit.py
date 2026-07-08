from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


AUDIT_ROOT = Path(__file__).parent / "audit_logs"
AUDIT_ROOT.mkdir(exist_ok=True)


class JsonAuditLogger:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()

    def _safe_component(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
        return cleaned.strip("-") or "unknown"

    def write_event(
        self,
        *,
        category: str,
        action: str,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> Path:
        now = datetime.now()
        day_dir = self.root / now.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        parts = [now.strftime("%Y%m%d_%H%M%S_%f")]
        if session_id:
            parts.append(self._safe_component(session_id))
        parts.append(self._safe_component(category))
        parts.append(self._safe_component(action))
        path = day_dir / ("_".join(parts) + ".json")

        record = {
            "timestamp": now.isoformat(),
            "session_id": session_id,
            "category": category,
            "action": action,
            "payload": payload,
        }
        with self._lock:
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return path


audit_logger = JsonAuditLogger(AUDIT_ROOT)
