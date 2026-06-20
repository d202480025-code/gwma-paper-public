#!/usr/bin/env python3
from __future__ import annotations

import argparse

from gwma.config import load_config
from gwma.training.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output = run_training(load_config(args.config))
    print(f"Training artifacts written to {output}")


if __name__ == "__main__":
    main()
