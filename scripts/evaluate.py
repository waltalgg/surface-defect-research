import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_lab.cli import build_parser
from defect_lab.config import load_config
from defect_lab.evaluate import run_evaluation


def main() -> None:
    parser = build_parser("Evaluate a trained checkpoint on the test split.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_evaluation(config)


if __name__ == "__main__":
    main()
