import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_lab.cli import build_parser
from defect_lab.config import load_config
from defect_lab.synthetic import generate_synthetic_dataset


def main() -> None:
    parser = build_parser("Generate simple synthetic images from training data.")
    args = parser.parse_args()
    config = load_config(args.config)
    generate_synthetic_dataset(config)


if __name__ == "__main__":
    main()
