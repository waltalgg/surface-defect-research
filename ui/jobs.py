from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def jobs_dir() -> Path:
    path = repo_root() / "artifacts" / "ui_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"


def _runner_path() -> Path:
    return repo_root() / "ui" / "runner.py"


def _utc_stamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_job(label: str, command: list[str], config: str = "") -> str:
    job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    payload = {
        "job_id": job_id,
        "label": label,
        "config": config,
        "command": command,
        "status": "queued",
        "created_at": _utc_stamp(),
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "log_path": str((jobs_dir() / f"{job_id}.log").as_posix()),
    }
    with _status_path(job_id).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    runner_cmd = [
        sys.executable,
        str(_runner_path()),
        "--job-id",
        job_id,
        "--label",
        label,
    ]
    if config:
        runner_cmd.extend(["--config", config])
    runner_cmd.append("--")
    runner_cmd.extend(command)

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        runner_cmd,
        cwd=repo_root(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    return job_id


def list_jobs(limit: int = 25) -> list[dict]:
    rows = []
    for path in sorted(jobs_dir().glob("*.json"), reverse=True):
        with path.open("r", encoding="utf-8") as fh:
            rows.append(json.load(fh))
        if len(rows) >= limit:
            break
    return rows


def read_log(log_path: str, tail_lines: int = 400) -> str:
    path = Path(log_path)
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-tail_lines:])
