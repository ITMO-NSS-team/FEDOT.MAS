"""Tests for fedotmas.routing.models."""

from __future__ import annotations

import warnings

import pytest

from fedotmas.routing.models import LlmPool, LlmPoolEntry


class TestLlmPool:
    def test_empty_pool_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LlmPool(entries=())

    def test_duplicate_models_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            LlmPool(
                entries=(
                    LlmPoolEntry(model="a", input_price_per_1m=1.0),
                    LlmPoolEntry(model="a", input_price_per_1m=2.0),
                )
            )

    def test_models_property_preserves_order(self) -> None:
        pool = LlmPool(
            entries=(
                LlmPoolEntry(model="c", input_price_per_1m=1.0),
                LlmPoolEntry(model="a", input_price_per_1m=1.0),
                LlmPoolEntry(model="b", input_price_per_1m=1.0),
            )
        )
        assert pool.models == ("c", "a", "b")

    def test_by_model_returns_entry(self) -> None:
        e = LlmPoolEntry(model="x", input_price_per_1m=1.0, output_price_per_1m=2.0)
        pool = LlmPool(entries=(e,))
        assert pool.by_model("x") is e

    def test_by_model_raises_on_unknown(self) -> None:
        pool = LlmPool(entries=(LlmPoolEntry(model="x", input_price_per_1m=1.0),))
        with pytest.raises(KeyError):
            pool.by_model("nonexistent")

    def test_warns_when_all_prices_zero(self) -> None:
        with pytest.warns(UserWarning, match="zero prices"):
            LlmPool(entries=(LlmPoolEntry(model="cheap"),))

    def test_warning_lists_offenders(self) -> None:
        with pytest.warns(UserWarning, match=r"\['a', 'c'\]"):
            LlmPool(
                entries=(
                    LlmPoolEntry(model="a"),
                    LlmPoolEntry(model="b", input_price_per_1m=1.0),
                    LlmPoolEntry(model="c"),
                )
            )

    def test_no_warning_when_only_input_price_set(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LlmPool(entries=(LlmPoolEntry(model="x", input_price_per_1m=0.5),))

    def test_no_warning_when_only_output_price_set(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LlmPool(entries=(LlmPoolEntry(model="x", output_price_per_1m=0.5),))
