from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _jobs_dir() -> Path:
    path = _repo_root() / "artifacts" / "ui_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _log_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.log"


def _read_status(job_id: str) -> dict:
    path = _status_path(job_id)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_status(job_id: str, payload: dict) -> None:
    with _status_path(job_id).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Background runner for UI-triggered jobs.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command specified for ui runner.")

    status = _read_status(args.job_id)
    status.update(
        {
            "job_id": args.job_id,
            "label": args.label,
            "config": args.config,
            "command": command,
            "status": "running",
            "started_at": _utc_now(),
            "finished_at": None,
            "returncode": None,
            "log_path": str(_log_path(args.job_id).as_posix()),
        }
    )
    _write_status(args.job_id, status)

    log_path = _log_path(args.job_id)
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_file.write(f"$ {' '.join(command)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=_repo_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()

        returncode = process.wait()

    status["status"] = "completed" if returncode == 0 else "failed"
    status["returncode"] = returncode
    status["finished_at"] = _utc_now()
    _write_status(args.job_id, status)

    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
