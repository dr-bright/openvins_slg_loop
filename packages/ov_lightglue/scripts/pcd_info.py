#!/usr/bin/env python3

import argparse
from pathlib import Path


def read_pcd_fields(path: Path):
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="replace").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            key = parts[0].upper()
            if key == "FIELDS":
                return parts[1:]
            if key == "DATA":
                break
    raise ValueError(f"PCD header has no FIELDS line: {path}")


def main():
    parser = argparse.ArgumentParser(description="Print PCD column names.")
    parser.add_argument("pcd", type=Path, help="Path to a .pcd file")
    args = parser.parse_args()

    fields = read_pcd_fields(args.pcd)
    for field in fields:
        print(field)


if __name__ == "__main__":
    main()
