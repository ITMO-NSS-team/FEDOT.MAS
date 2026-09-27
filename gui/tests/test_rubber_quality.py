import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
app = importlib.import_module('server.app')
from server.streaming import rubber_validation_result
from experiments.rubber_recipe_mas.open_data_predictor import (
    load_rows, predict_properties,
)


def test_quality_endpoint():
    data = app.rubber_quality()
    assert data['samples'] == 20
    assert data['method'] == 'LOOCV'
    assert data['mape_pct']['oil_swelling_pct_1006h'] == pytest.approx(9.6877146)
    example = data['baseline_example']
    assert example['training_rows'] == 19
    assert example['reference_used_in_training'] is False
    assert example['recipe']['carbon_black_n220_phr'] == 60
    assert example['mape_pct'] == pytest.approx(3.2122669312)


def test_mcp_wrapped_validation():
    diagnostic = {'mape_pct': None, 'reason': 'no_exact_reference'}
    result = {'status': 'prediction_completed', 'recipe_validation': diagnostic}
    assert rubber_validation_result(result) == diagnostic
    assert rubber_validation_result({'content': [{'type': 'text', 'text': json.dumps(result)}]}) == diagnostic
    assert rubber_validation_result({'result': {'structuredContent': result}}) == diagnostic
    assert rubber_validation_result({'text': 'not JSON'}) is None


def test_heldout_recipe_mape_survives_tool_result_wrapper():
    recipe = {
        'nr_smr20_phr': 50, 'sbr1502_phr': 50, 'carbon_black_n220_phr': 60,
        'zinc_oxide_phr': 5, 'stearic_acid_phr': 2,
        'tmq_antioxidant_phr': 1.5, '6ppd_antiozonant_phr': 1.5,
        'process_oil_phr': 5, 'sulfur_phr': 5, 'tmtd_accelerator_phr': 2,
        'reclaim_phr': 5,
    }
    result = predict_properties(recipe, load_rows())
    wrapped = {'content': [{'type': 'text', 'text': json.dumps(result)}]}
    validation = rubber_validation_result(wrapped)
    assert validation['mape_pct'] == pytest.approx(3.2122669312)
    assert validation['reference_used_in_training'] is False
    assert validation['training_rows'] == 19
