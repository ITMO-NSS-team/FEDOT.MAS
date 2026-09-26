import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
app = importlib.import_module('server.app')
from server.streaming import rubber_validation_result


def test_quality_endpoint():
    data = app.rubber_quality()
    assert data['samples'] == 20
    assert data['method'] == 'LOOCV'
    assert data['mape_pct']['oil_swelling_pct_1006h'] == pytest.approx(9.6877146)


def test_mcp_wrapped_validation():
    diagnostic = {'mape_pct': None, 'reason': 'no_exact_reference'}
    result = {'status': 'prediction_completed', 'recipe_validation': diagnostic}
    assert rubber_validation_result(result) == diagnostic
    assert rubber_validation_result({'content': [{'type': 'text', 'text': json.dumps(result)}]}) == diagnostic
    assert rubber_validation_result({'result': {'structuredContent': result}}) == diagnostic
    assert rubber_validation_result({'text': 'not JSON'}) is None
