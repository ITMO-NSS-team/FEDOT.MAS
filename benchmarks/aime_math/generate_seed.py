"""Generate a seed MAWConfig for AIME Math via FEDOT.MAS meta-agent.

Passes an abstract task description (not a concrete problem) to
``MAW.generate_config`` and saves the resulting pipeline to
``seed_config.json``. The abstract description avoids overspecialising the
generated prompts to a single problem instance.

Usage::

    python benchmarks/aime_math/generate_seed.py
    python benchmarks/aime_math/generate_seed.py --single-stage
    python benchmarks/aime_math/generate_seed.py --output custom_seed.json
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fedotmas.common.logging import get_logger
from fedotmas.maw.maw import MAW

from settings import AimeMathSettings

_log = get_logger("fmbench.aime_math.gen_seed")


# Abstract task description. Deliberately avoids any concrete problem text —
# if we pass an example, the generator bakes problem-specific entities into
# the prompts and the seed becomes biased toward that subdomain.
_TASK_DESCRIPTION = """Competition mathematics problems from the AIME
(American Invitational Mathematics Examination).

INPUT: a single problem statement. Topics span algebra, number theory,
combinatorics, geometry, and trigonometry. Problems require multi-step
reasoning and often a non-obvious insight or trick.

OUTPUT: a single non-negative integer between 0 and 999 — the numerical
answer to the problem. Only the integer itself, no units, no explanation,
no LaTeX formatting.

Solutions typically require careful symbolic manipulation, case analysis,
or geometric reasoning. The final answer must be exact (not a decimal
approximation) and is always an integer in the range [0, 999]."""


def _summarize(config) -> str:
    lines: list[str] = []
    lines.append(f"Agents ({len(config.agents)}):")
    for a in config.agents:
        lines.append(f"  - {a.name} (model={a.model}, output={a.output_key})")
        instr = (a.instruction or "").strip().replace("\n", " ")
        if len(instr) > 200:
            instr = instr[:200] + "..."
        lines.append(f"      instruction: {instr}")
    lines.append(f"Pipeline: {config.pipeline.model_dump_json(indent=2)}")
    return "\n".join(lines)


async def main(settings: AimeMathSettings, *, two_stage: bool, output: Path) -> None:
    _log.info("Passing abstract task description to generator (no concrete example)")

    maw = MAW(worker_models=[settings.solver_model], two_stage=two_stage)
    config = await maw.generate_config(_TASK_DESCRIPTION)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(config.model_dump_json(indent=2))
    _log.info("Saved seed config to {}", output)

    print("\n" + "=" * 60)
    print(f"Generated seed ({'two-stage' if two_stage else 'single-stage'}):")
    print("=" * 60)
    print(_summarize(config))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AIME Math seed config")
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Use single-stage generation (default: two-stage pool+pipeline)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "seed_config.json",
        help="Where to save the generated MAWConfig JSON",
    )
    args = parser.parse_args()

    settings = AimeMathSettings()
    asyncio.run(
        main(settings, two_stage=not args.single_stage, output=args.output)
    )
