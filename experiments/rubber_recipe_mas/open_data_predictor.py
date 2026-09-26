from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "open_data" / "tire_tread_sbr_nr.csv"
DEFAULT_REQUEST = HERE / "prototype_request.json"

TARGETS = (
    "thermal_conductivity_w_mk",
    "oil_swelling_pct_1006h",
    "water_swelling_pct_1006h",
    "specific_gravity",
)

FIXED_RECIPE = {
    "zinc_oxide_phr": 5.0,
    "stearic_acid_phr": 2.0,
    "tmq_antioxidant_phr": 1.5,
    "6ppd_antiozonant_phr": 1.5,
    "process_oil_phr": 5.0,
    "sulfur_phr": 5.0,
    "tmtd_accelerator_phr": 2.0,
    "reclaim_phr": 5.0,
}


def _features(nr_phr: float, carbon_black_phr: float) -> list[float]:
    nr = (nr_phr - 50.0) / 50.0
    cb = (carbon_black_phr - 50.0) / 30.0
    return [1.0, nr, cb, nr * cb, nr * nr, cb * cb]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    size = len(vector)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Singular regression system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[row][-1] for row in range(size)]


@dataclass(frozen=True)
class SurfaceModel:
    coefficients: tuple[float, ...]

    def predict(self, nr_phr: float, carbon_black_phr: float) -> float:
        return sum(
            coefficient * feature
            for coefficient, feature in zip(
                self.coefficients,
                _features(nr_phr, carbon_black_phr),
                strict=True,
            )
        )


def load_rows(path: Path = DEFAULT_DATA) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if len(rows) != 20:
        raise ValueError(f"Expected 20 open-data rows, received {len(rows)}")
    return rows


def fit_surface(
    rows: list[dict[str, float]], target: str, ridge: float = 1e-6
) -> SurfaceModel:
    feature_rows = [
        _features(row["nr_phr"], row["carbon_black_n220_phr"]) for row in rows
    ]
    width = len(feature_rows[0])
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for features, row in zip(feature_rows, rows, strict=True):
        for left in range(width):
            rhs[left] += features[left] * row[target]
            for right in range(width):
                gram[left][right] += features[left] * features[right]
    for index in range(1, width):
        gram[index][index] += ridge
    return SurfaceModel(tuple(_solve(gram, rhs)))


def loocv_metrics(rows: list[dict[str, float]], target: str) -> dict[str, float | None]:
    actual: list[float] = []
    predicted: list[float] = []
    for held_out, row in enumerate(rows):
        training = rows[:held_out] + rows[held_out + 1 :]
        model = fit_surface(training, target)
        actual.append(row[target])
        predicted.append(
            model.predict(row["nr_phr"], row["carbon_black_n220_phr"])
        )
    errors = [guess - truth for guess, truth in zip(predicted, actual, strict=True)]
    mean = sum(actual) / len(actual)
    residual_sum = sum(error * error for error in errors)
    total_sum = sum((value - mean) ** 2 for value in actual)
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mape_pct": (100 * sum(abs(error / truth) for error, truth in zip(errors, actual, strict=True))
                     / len(actual)) if all(actual) else None,
        "rmse": math.sqrt(residual_sum / len(errors)),
        "r2": 1.0 - residual_sum / total_sum if total_sum else 0.0,
    }


def predict_properties(
    recipe: dict[str, float], rows: list[dict[str, float]]
) -> dict[str, Any]:
    """Predict four published properties for the recipe supplied unchanged."""
    nr_phr = float(recipe["nr_smr20_phr"])
    sbr_phr = float(recipe["sbr1502_phr"])
    carbon_black_phr = float(recipe["carbon_black_n220_phr"])
    if not math.isclose(nr_phr + sbr_phr, 100.0, abs_tol=1e-6):
        raise ValueError("NR SMR-20 and SBR-1502 must sum to 100 phr")

    mismatches = [
        name
        for name, expected in FIXED_RECIPE.items()
        if not math.isclose(float(recipe[name]), expected, abs_tol=1e-6)
    ]
    if mismatches:
        expected = ", ".join(
            f"{name}={FIXED_RECIPE[name]} phr" for name in mismatches
        )
        raise ValueError(
            "The open dataset holds these ingredients fixed; expected " + expected
        )

    nr_values = [row["nr_phr"] for row in rows]
    cb_values = [row["carbon_black_n220_phr"] for row in rows]
    inside = (
        min(nr_values) <= nr_phr <= max(nr_values)
        and min(cb_values) <= carbon_black_phr <= max(cb_values)
    )
    if not inside:
        raise ValueError(
            "Recipe is outside the published interpolation domain: "
            f"NR {min(nr_values):g}-{max(nr_values):g} phr, "
            f"N220 {min(cb_values):g}-{max(cb_values):g} phr"
        )

    models = {target: fit_surface(rows, target) for target in TARGETS}
    metrics = {target: loocv_metrics(rows, target) for target in TARGETS}
    predictions = {
        target: model.predict(nr_phr, carbon_black_phr)
        for target, model in models.items()
    }
    nearest = min(
        rows,
        key=lambda row: (
            ((nr_phr - row["nr_phr"]) / 25.0) ** 2
            + ((carbon_black_phr - row["carbon_black_n220_phr"]) / 20.0) ** 2
        ),
    )
    nearest_distance = math.sqrt(
        ((nr_phr - nearest["nr_phr"]) / 25.0) ** 2
        + ((carbon_black_phr - nearest["carbon_black_n220_phr"]) / 20.0) ** 2
    )

    return {
        "status": "prediction_completed",
        "recipe_validation": recipe_validation(recipe, predictions, rows),
        "recipe_input_unchanged": {key: float(value) for key, value in recipe.items()},
        "predicted_properties": {
            target: {
                "value": round(value, 5),
                "loocv_mae": round(metrics[target]["mae"], 5),
                "loocv_rmse": round(metrics[target]["rmse"], 5),
                "loocv_mape_pct": (round(metrics[target]["mape_pct"], 5)
                                   if metrics[target]["mape_pct"] is not None else None),
                "approx_95pct_range_from_loocv_rmse": [
                    round(value - 1.96 * metrics[target]["rmse"], 5),
                    round(value + 1.96 * metrics[target]["rmse"], 5),
                ],
            }
            for target, value in predictions.items()
        },
        "domain": {
            "inside_published_grid": True,
            "nr_phr_range": [min(nr_values), max(nr_values)],
            "carbon_black_n220_phr_range": [min(cb_values), max(cb_values)],
            "normalized_distance_to_nearest_observation": round(nearest_distance, 4),
            "nearest_observation": {
                "nr_phr": nearest["nr_phr"],
                "sbr_phr": nearest["sbr_phr"],
                "carbon_black_n220_phr": nearest["carbon_black_n220_phr"],
            },
            "training_rows": len(rows),
        },
        "provenance": {
            "doi": "10.5281/zenodo.3838695",
            "license": "CC BY 4.0",
            "source_rows": len(rows),
            "digitized_figures": [4, 10, 11, 14],
        },
        "warnings": [
            "Research screening estimate only; not a production or tire-safety recommendation.",
            "Source values were digitized from plots rather than read from raw instrument files.",
            "The reported range is a heuristic based on LOOCV RMSE, not a calibrated prediction interval.",
            "Laboratory validation is required for this exact recipe and processing regime.",
        ],
    }


def recipe_validation(recipe: dict, predictions: dict, rows: list[dict]) -> dict:
    matched = next((r for r in rows if
        math.isclose(r["nr_phr"], recipe["nr_smr20_phr"], abs_tol=1e-9, rel_tol=0)
        and math.isclose(r["carbon_black_n220_phr"], recipe["carbon_black_n220_phr"], abs_tol=1e-9, rel_tol=0)), None)
    if matched is None:
        return {"mape_pct": None, "reason": "no_exact_reference", "ape_pct": {}}
    errors = {t: 100 * abs(predictions[t] - matched[t]) / abs(matched[t])
              if matched[t] else None for t in TARGETS}
    return {"mape_pct": sum(errors.values()) / len(errors) if all(v is not None for v in errors.values()) else None,
            "ape_pct": errors, "reference": {t: matched[t] for t in TARGETS},
            "reference_used_in_training": True, "recipe": recipe}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict properties of one supplied rubber recipe"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = predict_properties(request, load_rows(args.data))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
