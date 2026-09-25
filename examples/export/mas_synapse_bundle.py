"""Export the technology-card MAS from PR #46 without model calls.

Run from the repository root:
    uv run python examples/export/mas_synapse_bundle.py > /tmp/technology_card_bundle.json

The fixture retains its original local model/server identifiers. Before import,
replace them with model and tool ids available in the target Synapse tenant.
"""

import json
from pathlib import Path

from fedotmas.export import to_synapse_bundle
from fedotmas.mas.models import MASConfig


def main() -> None:
    config = MASConfig.model_validate_json(
        Path(__file__).with_name("technology_card_mas.json").read_text()
    )
    export = to_synapse_bundle(config, workflow_id="technology_card_audit")
    print(json.dumps(export.bundle, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
