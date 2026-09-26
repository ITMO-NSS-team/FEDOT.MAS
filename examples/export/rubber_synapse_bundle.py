"""Export the five-agent rubber-recipe MAW from PR #46 without model calls.

Run from the repository root:
    uv run python examples/export/rubber_synapse_bundle.py > /tmp/rubber_recipe_bundle.json

The fixture uses the staging tenant tool IDs supplied for the demo. Before
import, replace the local model ID with a model available in Synapse. Other
tenants may require different tool IDs.
"""

import json
from pathlib import Path

from fedotmas.export import to_synapse_bundle
from fedotmas.maw.models import MAWConfig


def main() -> None:
    config = MAWConfig.model_validate_json(
        Path(__file__).with_name("rubber_recipe_maw.json").read_text(encoding="utf-8")
    )
    export = to_synapse_bundle(config, workflow_id="rubber_recipe_prediction")
    print(json.dumps(export.bundle, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
