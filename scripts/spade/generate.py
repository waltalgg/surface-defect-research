import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from defect_lab.config import load_config
from defect_lab.spade import generate_spade_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SPADE sample grids.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    generate_spade_samples(config, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
