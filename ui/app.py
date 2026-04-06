from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import streamlit as st

from ui.jobs import create_job, list_jobs, read_log


CONFIGS_DIR = ROOT / "configs"
REPORTS_DIR = ROOT / "artifacts" / "reports"
PLOTS_COMPARE_DIR = ROOT / "artifacts" / "plots_compare"
PLOTS_SYNTH_DIR = ROOT / "artifacts" / "plots_synthetic"
PLOTS_COMPOSITE_DIR = ROOT / "artifacts" / "plots_synthetic_composite"


def config_names() -> list[str]:
    return sorted(path.name for path in CONFIGS_DIR.glob("*.yaml"))


def run_python_job(label: str, args: list[str], config: str = "") -> str:
    command = [sys.executable] + args
    return create_job(label=label, command=command, config=config)


def run_for_config(action: str, config_name: str) -> str:
    script_map = {
        "prepare": "scripts/prepare_dataset.py",
        "synthetic": "scripts/generate_synthetic.py",
        "train": "scripts/train.py",
        "evaluate": "scripts/evaluate.py",
    }
    script = script_map[action]
    return run_python_job(
        label=f"{action}:{config_name}",
        args=[str(ROOT / script), "--config", f"configs/{config_name}"],
        config=config_name,
    )


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def render_controls() -> None:
    st.subheader("Run Controls")
    configs = config_names()
    selected = st.selectbox("Configuration", configs, index=0 if configs else None)

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Prepare Dataset", use_container_width=True, disabled=not selected):
        st.success(f"Queued job: {run_for_config('prepare', selected)}")
    if col2.button("Generate Synthetic", use_container_width=True, disabled=not selected):
        st.success(f"Queued job: {run_for_config('synthetic', selected)}")
    if col3.button("Train Model", use_container_width=True, disabled=not selected):
        st.success(f"Queued job: {run_for_config('train', selected)}")
    if col4.button("Evaluate", use_container_width=True, disabled=not selected):
        st.success(f"Queued job: {run_for_config('evaluate', selected)}")

    st.divider()
    col5, col6, col7 = st.columns(3)
    if col5.button("Export Reports", use_container_width=True):
        st.success(f"Queued job: {run_python_job('export:reports', [str(ROOT / 'scripts/export_results.py')])}")
    if col6.button("Build Tables + Examples", use_container_width=True):
        st.success(
            f"Queued job: {run_python_job('reports:final', [str(ROOT / 'scripts/build_summary_and_examples.py')])}"
        )
    if col7.button("Build Composite Gallery", use_container_width=True):
        st.success(
            f"Queued job: {run_python_job('plots:composite', [str(ROOT / 'scripts/plot_composite_examples.py')])}"
        )

    if st.button("Build Comparison Plots", use_container_width=True):
        st.success(f"Queued job: {run_python_job('plots:current', [str(ROOT / 'scripts/plot_current_results.py')])}")


def render_jobs() -> None:
    st.subheader("Jobs and Logs")
    jobs = list_jobs(limit=30)
    if not jobs:
        st.info("No jobs yet.")
        return

    labels = [f"{job['status']}: {job['label']} ({job['job_id']})" for job in jobs]
    selected_label = st.selectbox("Recent jobs", labels, index=0)
    selected_job = jobs[labels.index(selected_label)]

    st.write(
        {
            "status": selected_job.get("status"),
            "config": selected_job.get("config"),
            "created_at": selected_job.get("created_at"),
            "started_at": selected_job.get("started_at"),
            "finished_at": selected_job.get("finished_at"),
            "returncode": selected_job.get("returncode"),
        }
    )
    st.text_area("Log output", read_log(selected_job.get("log_path", "")), height=420)


def render_reports() -> None:
    st.subheader("Summary Tables")
    final_rows = read_csv(REPORTS_DIR / "final_results_table.csv")
    current_rows = read_csv(REPORTS_DIR / "current_comparison_summary.csv")

    if final_rows:
        st.caption("Final results table")
        st.dataframe(final_rows, use_container_width=True)
    if current_rows:
        st.caption("Current comparison summary")
        st.dataframe(current_rows, use_container_width=True)
    if not final_rows and not current_rows:
        st.info("No report tables found yet.")


def _render_images_from(directory: Path, title: str) -> None:
    if not directory.exists():
        return
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        return
    st.subheader(title)
    for image_path in images:
        st.image(str(image_path), caption=image_path.name, use_container_width=True)


def render_artifacts() -> None:
    _render_images_from(PLOTS_COMPARE_DIR, "Comparison Plots")
    _render_images_from(PLOTS_SYNTH_DIR, "Synthetic Examples")
    _render_images_from(PLOTS_COMPOSITE_DIR, "Composite Synthetic Examples")


def main() -> None:
    st.set_page_config(page_title="Surface Defect Lab", layout="wide")
    st.title("Surface Defect Lab UI")
    st.caption("Run experiments, inspect logs, and browse current reports and plots.")

    tab1, tab2, tab3 = st.tabs(["Controls", "Jobs", "Results"])
    with tab1:
        render_controls()
    with tab2:
        render_jobs()
    with tab3:
        render_reports()
        render_artifacts()


if __name__ == "__main__":
    main()
