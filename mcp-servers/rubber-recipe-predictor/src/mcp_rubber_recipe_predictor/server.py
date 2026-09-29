from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "rubber_recipe_mas"
sys.path.insert(0, str(EXPERIMENT_DIR))

from open_data_predictor import load_rows, predict_properties

mcp = FastMCP(
    "rubber-recipe-predictor",
    instructions=(
        "Use predict_rubber_properties for research-only interpolation inside the "
        "published SBR/NR/N220 domain. The result is not a production or tire-"
        "safety recommendation."
    ),
)


@mcp.tool
def predict_rubber_properties(
    nr_smr20_phr: Annotated[float, Field(ge=0, le=100, description="NR SMR-20, phr")],
    sbr1502_phr: Annotated[float, Field(ge=0, le=100, description="SBR-1502, phr")],
    carbon_black_n220_phr: Annotated[float, Field(ge=20, le=80, description="N220, phr")],
    zinc_oxide_phr: Annotated[float, Field(ge=0, description="Zinc oxide, phr")],
    stearic_acid_phr: Annotated[float, Field(ge=0, description="Stearic acid, phr")],
    tmq_antioxidant_phr: Annotated[float, Field(ge=0, description="TMQ antioxidant, phr")],
    antiozonant_6ppd_phr: Annotated[float, Field(ge=0, description="6PPD antiozonant, phr")],
    process_oil_phr: Annotated[float, Field(ge=0, description="Process oil, phr")],
    sulfur_phr: Annotated[float, Field(ge=0, description="Sulfur, phr")],
    tmtd_accelerator_phr: Annotated[float, Field(ge=0, description="TMTD accelerator, phr")],
    reclaim_phr: Annotated[float, Field(ge=0, description="Reclaim rubber, phr")],
) -> dict[str, Any]:
    """Predict published properties for exactly the supplied rubber recipe."""
    recipe = {
        "nr_smr20_phr": nr_smr20_phr,
        "sbr1502_phr": sbr1502_phr,
        "carbon_black_n220_phr": carbon_black_n220_phr,
        "zinc_oxide_phr": zinc_oxide_phr,
        "stearic_acid_phr": stearic_acid_phr,
        "tmq_antioxidant_phr": tmq_antioxidant_phr,
        "6ppd_antiozonant_phr": antiozonant_6ppd_phr,
        "process_oil_phr": process_oil_phr,
        "sulfur_phr": sulfur_phr,
        "tmtd_accelerator_phr": tmtd_accelerator_phr,
        "reclaim_phr": reclaim_phr,
    }
    return predict_properties(recipe, load_rows())


def main() -> None:
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
