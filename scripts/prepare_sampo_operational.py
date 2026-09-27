#!/usr/bin/env python3
"""Freeze a separate public-only set for operational smoke runs."""
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampo_cost_demo import construct_operational_set
if __name__ == "__main__":
    print(json.dumps(construct_operational_set(), ensure_ascii=False, indent=2))
