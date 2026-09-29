"""Export the technology-card MAS from PR #46 without model calls.

Run from the repository root:
    uv run python examples/export/mas_synapse_bundle.py > /tmp/technology_card_bundle.json

The fixture uses the staging tenant tool IDs supplied for the demo. Before
import, replace the local model ID with a model available in Synapse. Other
tenants may require different tool IDs.
"""

import json
from pathlib import Path

from fedotmas.export import to_synapse_bundle
from fedotmas.mas.models import MASConfig


def main() -> None:
    config = MASConfig.model_validate_json(
        Path(__file__).with_name("technology_card_mas.json").read_text(encoding="utf-8")
    )
    export = to_synapse_bundle(config, workflow_id="technology_card_audit")
    print(json.dumps(export.bundle, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
