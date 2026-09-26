from __future__ import annotations

import json
from pathlib import Path
import pytest

from open_data_predictor import DEFAULT_DATA, TARGETS, load_rows, predict_properties, loocv_metrics

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


def test_mape_matches_fixed_loocv_values():
    rows = load_rows()
    expected = [4.49396273, 9.68771460, 3.45543630, 0.59404942]
    for target, value in zip(TARGETS, expected):
        assert loocv_metrics(rows, target)["mape_pct"] == pytest.approx(value)


def test_recipe_error_requires_exact_reference():
    request = json.loads((HERE / "prototype_request.json").read_text(encoding="utf-8"))
    request.update(nr_smr20_phr=55, sbr1502_phr=45, carbon_black_n220_phr=55)
    result = predict_properties(request, load_rows())
    assert result["recipe_validation"]["mape_pct"] is None
    assert result["recipe_validation"]["reason"] == "no_exact_reference"
    request.update(nr_smr20_phr=50, sbr1502_phr=50, carbon_black_n220_phr=60)
    result = predict_properties(request, load_rows())
    validation = result["recipe_validation"]
    assert validation["reference_used_in_training"] is True
    assert validation["mape_pct"] == pytest.approx(sum(validation["ape_pct"].values()) / 4)
    for key in TARGETS:
        assert result["predicted_properties"][key]["loocv_mape_pct"] > 0


def test_mape_with_zero_reference_is_unavailable():
    rows = load_rows()
    rows[0][TARGETS[0]] = 0
    assert loocv_metrics(rows, TARGETS[0])["mape_pct"] is None
