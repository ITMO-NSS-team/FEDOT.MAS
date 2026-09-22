from __future__ import annotations

import json
from pathlib import Path

from open_data_predictor import DEFAULT_DATA, load_rows, predict_properties

HERE = Path(__file__).resolve().parent


def test_open_dataset_is_complete_factorial_grid() -> None:
    rows = load_rows(DEFAULT_DATA)
    assert len(rows) == 20
    assert {row["nr_phr"] for row in rows} == {0.0, 25.0, 50.0, 75.0, 100.0}
    assert {row["carbon_black_n220_phr"] for row in rows} == {
        20.0,
        40.0,
        60.0,
        80.0,
    }
    assert all(row["nr_phr"] + row["sbr_phr"] == 100.0 for row in rows)


def test_supplied_recipe_is_returned_unchanged_with_predictions() -> None:
    request = json.loads((HERE / "prototype_request.json").read_text(encoding="utf-8"))
    result = predict_properties(request, load_rows(DEFAULT_DATA))
    assert result["status"] == "prediction_completed"
    assert result["domain"]["inside_published_grid"] is True
    assert result["recipe_input_unchanged"] == request
    assert set(result["predicted_properties"]) == {
        "thermal_conductivity_w_mk",
        "oil_swelling_pct_1006h",
        "water_swelling_pct_1006h",
        "specific_gravity",
    }
