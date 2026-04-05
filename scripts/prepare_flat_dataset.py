from __future__ import annotations

import shutil
from pathlib import Path


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Convert a flat image directory into class subfolders using filename prefixes.")
    parser.add_argument("--source-dir", required=True, help="Directory with flat image files.")
    parser.add_argument("--target-dir", required=True, help="Output directory with class subfolders.")
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".bmp", ".png"], help="Allowed extensions.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them.")
    return parser.parse_args()


def infer_label(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0].lower()
    return stem.lower()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    extensions = {extension.lower() for extension in args.extensions}

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    image_paths = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_paths:
        raise RuntimeError(f"No matching image files found in {source_dir}")

    copied = 0
    for path in image_paths:
        label = infer_label(path)
        class_dir = target_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        destination = class_dir / path.name
        if args.copy:
            shutil.copy2(path, destination)
        else:
            shutil.move(str(path), destination)
        copied += 1

    print(f"Prepared flat dataset at {target_dir} with {copied} files")


if __name__ == "__main__":
    main()
