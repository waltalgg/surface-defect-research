import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from defect_lab.cli import build_parser
from defect_lab.config import load_config
from defect_lab.segmentation import run_segmentation_training


def main() -> None:
    parser = build_parser("Train a segmentation model on paired image-mask data.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_segmentation_training(config)


if __name__ == "__main__":
    main()
