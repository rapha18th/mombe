"""Session and registry storage.

A session is one animal, one task, one officer. Captures arrive over time: the
first upload, then one per re-capture instruction. Local storage backs the
simulator and tests; S3 and DynamoDB back the deployed service with the same
interface.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Capture:
    index: int
    muzzle: list[str] = field(default_factory=list)   # image paths
    body: list[str] = field(default_factory=list)     # frame paths (walk-around)
    meta: dict = field(default_factory=dict)          # instruction it answered, device, synthetic tags
    created: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    task: str
    officer: str = "officer-demo"
    claimed_owner: str | None = None
    policy_animal: str | None = None
    loan_threshold_usd: float | None = None
    captures: list[Capture] = field(default_factory=list)
    instructions: list[dict] = field(default_factory=list)
    status: str = "open"
    ledger: dict | None = None

    @property
    def latest(self) -> Capture | None:
        return self.captures[-1] if self.captures else None


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.root / "sessions" / sid / "session.json"

    def create(self, task: str, **kw) -> Session:
        s = Session(id=kw.pop("id", None) or uuid.uuid4().hex[:12], task=task, **kw)
        self.save(s)
        return s

    def save(self, s: Session) -> None:
        p = self._path(s.id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(s), indent=1))

    def load(self, sid: str) -> Session:
        d = json.loads(self._path(sid).read_text())
        d["captures"] = [Capture(**c) for c in d["captures"]]
        return Session(**d)

    def add_capture(self, sid: str, muzzle: list[np.ndarray | str] = (), body: list[np.ndarray | str] = (),
                    meta: dict | None = None) -> Capture:
        s = self.load(sid)
        c = Capture(index=len(s.captures), meta=meta or {})
        cdir = self.root / "sessions" / sid / f"capture_{c.index:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        for kind, items, out in (("muzzle", muzzle, c.muzzle), ("body", body, c.body)):
            for i, it in enumerate(items):
                if isinstance(it, str):
                    out.append(it)                        # reference an existing file
                else:
                    p = cdir / f"{kind}_{i:03d}.jpg"
                    cv2.imwrite(str(p), it, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    out.append(str(p))
        s.captures.append(c)
        self.save(s)
        return c

    def artifact_path(self, sid: str, name: str) -> Path:
        p = self.root / "sessions" / sid / "artifacts" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


class LocalRegistry:
    """Animal records plus approvals and decision traces, as JSON files."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.animals_path = self.root / "animals.json"
        self.approvals_path = self.root / "approvals.json"
        self.animals = json.loads(self.animals_path.read_text()) if self.animals_path.exists() else {}
        self.approvals = json.loads(self.approvals_path.read_text()) if self.approvals_path.exists() else {}

    def _flush(self):
        self.animals_path.write_text(json.dumps(self.animals, indent=1))
        self.approvals_path.write_text(json.dumps(self.approvals, indent=1))

    def get(self, animal_id: str) -> dict | None:
        return self.animals.get(animal_id)

    def put(self, animal_id: str, record: dict) -> None:
        self.animals[animal_id] = {**self.animals.get(animal_id, {}), **record, "updated": time.time()}
        self._flush()

    def request_approval(self, session_id: str, kind: str, summary: dict) -> dict:
        aid = "ap_" + uuid.uuid4().hex[:10]
        self.approvals[aid] = {"id": aid, "session_id": session_id, "kind": kind, "summary": summary,
                               "status": "pending", "created": time.time()}
        self._flush()
        return self.approvals[aid]

    def decide(self, approval_id: str, status: str, by: str, note: str = "") -> dict:
        a = self.approvals[approval_id]
        a.update(status=status, decided_by=by, note=note, decided=time.time())
        self._flush()
        return a

    def approval(self, approval_id: str) -> dict | None:
        return self.approvals.get(approval_id)
