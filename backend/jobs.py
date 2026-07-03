from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

JOB_STATE_FILENAME = ".job_state.json"


@dataclass
class Job:
    id: str
    created_at: float
    status: str  # queued | running | succeeded | failed
    progress: float = 0.0  # 0..1
    step: str = "queued"
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path
    meta: Dict[str, Any] = field(default_factory=dict)

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip("\n"))
        # keep memory bounded
        if len(self.logs) > 5000:
            self.logs = self.logs[-5000:]


def _job_dir(root: Path, job_id: str) -> Path:
    return root / job_id


def _job_state_path(root: Path, job_id: str) -> Path:
    return _job_dir(root, job_id) / JOB_STATE_FILENAME


def _json_safe(obj: Any) -> Any:
    """Ensure values are JSON-serializable (handles numpy scalars, etc.)."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    return str(obj)


def _job_from_state_dict(data: Dict[str, Any]) -> Job:
    return Job(
        id=str(data["id"]),
        created_at=float(data.get("created_at", 0)),
        status=str(data["status"]),
        progress=float(data.get("progress", 0.0)),
        step=str(data.get("step", "")),
        error=data.get("error"),
        logs=list(data.get("logs", [])),
        artifacts={k: str(v) for k, v in (data.get("artifacts") or {}).items()},
        meta=dict(data.get("meta") or {}),
    )


def load_job_from_disk(root: Path, job_id: str) -> Optional[Job]:
    path = _job_state_path(root, job_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    job = _job_from_state_dict(data)
    if job.status in ("queued", "running"):
        job.status = "failed"
        job.error = "后端已重启，任务已中断。请重新点击「开始分析」。"
        job.step = "interrupted"
    return job


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def persist(self, job: Job) -> None:
        d = _job_dir(self.root, job.id)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": job.id,
            "created_at": job.created_at,
            "status": job.status,
            "progress": job.progress,
            "step": job.step,
            "error": job.error,
            "meta": _json_safe(job.meta),
            "artifacts": {k: str(v) for k, v in job.artifacts.items()},
            "logs": job.logs[-500:],
        }
        _job_state_path(self.root, job.id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create(self) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, created_at=time.time(), status="queued")
        with self._lock:
            self._jobs[job_id] = job
        self.persist(job)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
        loaded = load_job_from_disk(self.root, job_id)
        if loaded is None and _job_dir(self.root, job_id).is_dir():
            loaded = Job(
                id=job_id,
                created_at=0.0,
                status="failed",
                progress=0.0,
                step="lost",
                error="找不到任务记录（后端可能已重启）。请重新运行分析。",
            )
        if loaded is None:
            raise KeyError(job_id)
        with self._lock:
            if job_id not in self._jobs:
                self._jobs[job_id] = loaded
            return self._jobs[job_id]

    def list(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

