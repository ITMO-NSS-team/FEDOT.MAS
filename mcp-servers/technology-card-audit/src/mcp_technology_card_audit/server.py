from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field


CARD_ID = "demo://earthworks/бурение_котлованов"
CARD_REFERENCE = (
    "Демо-ТТК ЗР-01, раздел 6 «Технико-экономические показатели», "
    "таблица 6.1"
)

DEMO_CARD: dict[str, Any] = {
    "card_id": CARD_ID,
    "title": "Бурение котлованов под опоры воздушной линии",
    "category": "земляные работы",
    "applicability": "Демонстрационный пример механизированного бурения котлованов.",
    "norms": {
        "crew_productivity_per_shift": {
            "value": 8.0,
            "unit": "котлован/смену",
            "card_ref": f"{CARD_REFERENCE}, строка «Выработка звена за смену»",
        },
        "labor_intensity_per_unit": {
            "value": 0.5,
            "unit": "чел.-дн./котлован",
            "card_ref": f"{CARD_REFERENCE}, строка «Трудоёмкость на единицу объёма»",
        },
        "crew_size": {
            "value": 4,
            "unit": "человек",
            "card_ref": f"{CARD_REFERENCE}, строка «Численность звена»",
        },
    },
    "consistency_check": "4 человека / 8 котлованов за смену = 0,5 чел.-дн./котлован",
    "source_kind": "synthetic_demo_fixture",
    "source_notice": (
        "Числа созданы только для демонстрации MAS и MCP-контракта. "
        "Они не извлечены из 1 803 карт и не являются производственным нормативом."
    ),
}

DEMO_HISTORY: tuple[dict[str, Any], ...] = (
    {
        "work_name": "Бурение скважин диаметром 425 мм",
        "granular_name": "Бурение котлованов",
        "object": "Куст 18",
        "start_date": "2021-05-12",
        "end_date": "2021-05-13",
        "volume": 40.0,
        "working_days": 2.0,
    },
    {
        "work_name": "Бурение котлованов под опоры ВЛ",
        "granular_name": "Бурение котлованов",
        "object": "Куст 7",
        "start_date": "2020-09-03",
        "end_date": "2020-09-08",
        "volume": 9.0,
        "working_days": 4.0,
    },
    {
        "work_name": "Бурение скважин d=425",
        "granular_name": "Бурение котлованов",
        "object": "Куст 24",
        "start_date": "2019-07-15",
        "end_date": "2019-07-15",
        "volume": 25.0,
        "working_days": 1.0,
    },
    {
        "work_name": "Бурение котлованов",
        "granular_name": "Бурение котлованов",
        "object": "Куст 11",
        "start_date": "2021-06-01",
        "end_date": "2021-06-02",
        "volume": 16.0,
        "working_days": 2.0,
    },
    {
        "work_name": "Устройство котлованов бурением",
        "granular_name": "Бурение котлованов",
        "object": "Куст 12",
        "start_date": "2020-08-10",
        "end_date": "2020-08-12",
        "volume": 27.0,
        "working_days": 3.0,
    },
    {
        "work_name": "Бурение скважин под опоры",
        "granular_name": "Бурение котлованов",
        "object": "Куст 2",
        "start_date": "2018-04-04",
        "end_date": "2018-04-05",
        "volume": 18.0,
        "working_days": 2.0,
    },
)


mcp = FastMCP(
    "technology-card-audit",
    instructions=(
        "Use these tools only for the labeled synthetic TTK demonstration. "
        "Never present the fixture as a real norm or as rows from STAIRS/SAMPO."
    ),
)


def _require_card(card_id: str) -> None:
    if card_id != CARD_ID:
        raise ValueError(f"Unknown demo card: {card_id}. Available card: {CARD_ID}")


@mcp.tool
def read_technology_card(
    card_id: Annotated[str, Field(description="Identifier of the technological card")],
) -> dict[str, Any]:
    """Return the demo TTK with quantitative norms and source references."""
    _require_card(card_id)
    return DEMO_CARD


@mcp.tool
def audit_historical_productivity(
    card_id: Annotated[str, Field(description="Identifier of the technological card")],
    upper_multiplier: Annotated[
        float,
        Field(gt=1, description="Flag fact rate strictly above norm times this multiplier"),
    ] = 2.0,
    lower_divisor: Annotated[
        float,
        Field(gt=1, description="Flag fact rate strictly below norm divided by this value"),
    ] = 3.0,
) -> dict[str, Any]:
    """Compare the demo card norm with a synthetic history slice."""
    _require_card(card_id)
    norm_rate = float(DEMO_CARD["norms"]["crew_productivity_per_shift"]["value"])
    high_boundary = norm_rate * upper_multiplier
    low_boundary = norm_rate / lower_divisor
    violations: list[dict[str, Any]] = []

    for row in DEMO_HISTORY:
        start = date.fromisoformat(row["start_date"])
        end = date.fromisoformat(row["end_date"])
        if end < start or row["working_days"] <= 0:
            raise ValueError(f"Invalid history row: {row}")
        fact_rate = row["volume"] / row["working_days"]
        if fact_rate > high_boundary:
            violation_type = "выше нормы более чем в 2 раза"
        elif fact_rate < low_boundary:
            violation_type = "ниже нормы более чем в 3 раза"
        else:
            continue
        violations.append(
            {
                "work_name": row["work_name"],
                "granular_name": row["granular_name"],
                "object": row["object"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "volume": row["volume"],
                "working_days": row["working_days"],
                "fact_rate": round(fact_rate, 3),
                "norm_rate": norm_rate,
                "unit": "котлован/рабочий день",
                "deviation_pct": round((fact_rate / norm_rate - 1) * 100, 1),
                "violation_type": violation_type,
                "card_ref": DEMO_CARD["norms"]["crew_productivity_per_shift"]["card_ref"],
            }
        )

    return {
        "status": "audit_completed",
        "card_id": card_id,
        "matching_field": "granular_name",
        "records_checked": len(DEMO_HISTORY),
        "violations_count": len(violations),
        "thresholds": {
            "norm_rate": norm_rate,
            "high_strictly_above": high_boundary,
            "low_strictly_below": round(low_boundary, 3),
            "unit": "котлован/рабочий день",
        },
        "violations": violations,
        "source_kind": "synthetic_demo_fixture",
        "source_notice": (
            "Исторические строки синтетические и служат для проверки логики отчёта; "
            "это не выгрузка STAIRS/SAMPO."
        ),
    }


def main() -> None:
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
