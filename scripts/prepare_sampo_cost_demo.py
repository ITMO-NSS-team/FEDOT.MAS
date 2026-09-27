#!/usr/bin/env python3
"""Create the immutable SAMPO cost-demo split. This is a one-time operation."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampo_cost_demo import construct_split

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--size", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(construct_split(args.seed, args.size), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
