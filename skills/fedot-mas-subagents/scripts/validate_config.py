from __future__ import annotations

import argparse
from pathlib import Path

from fedotmas.maw.models import MAWConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a FEDOT.MAS MAW config")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = MAWConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    print(config.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
