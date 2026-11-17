"""
SHA-256 hash-chained, append-only audit log (ported from SecondLife).

Every card change and every decision point is chained to the previous entry,
so the documentation trail is tamper-evident — useful when a TCCC card is
forwarded with the casualty and later reviewed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENESIS = "GENESIS"


def _h(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._prev = GENESIS

    def log(self, kind: str, t: float, payload: dict) -> None:
        entry = {"seq": None, "kind": kind, "t": round(t, 2), "payload": payload, "prev_hash": self._prev}
        entry["seq"] = sum(1 for _ in open(self.path, encoding="utf-8"))
        entry["hash"] = _h(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self._prev = entry["hash"]

    @staticmethod
    def verify(path: str | Path) -> bool:
        prev = GENESIS
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["prev_hash"] != prev or _h({k: v for k, v in e.items() if k != "hash"}) != e["hash"]:
                return False
            prev = e["hash"]
        return True
