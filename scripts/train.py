import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_lab.cli import build_parser
from defect_lab.config import load_config
from defect_lab.train import run_training


def main() -> None:
    parser = build_parser("Train a baseline model on the prepared dataset.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
