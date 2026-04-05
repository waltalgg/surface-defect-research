import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_lab.cli import build_parser
from defect_lab.config import load_config
from defect_lab.dataset_prep import prepare_dataset


def main() -> None:
    parser = build_parser("Prepare dataset splits from raw image folders.")
    args = parser.parse_args()
    config = load_config(args.config)
    prepare_dataset(config)


if __name__ == "__main__":
    main()
