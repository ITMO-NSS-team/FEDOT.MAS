"""Generate a seed MAWConfig for HotpotQA via FEDOT.MAS meta-agent.

Passes an abstract task description (not a concrete example) to
``MAW.generate_config`` and saves the resulting pipeline to
``seed_config.json``. The abstract description avoids overspecialising the
generated prompts to a single dataset row.

Usage::

    python benchmarks/hotpot_qa/generate_seed.py
    python benchmarks/hotpot_qa/generate_seed.py --single-stage
    python benchmarks/hotpot_qa/generate_seed.py --output custom_seed.json
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fedotmas.common.logging import get_logger
from fedotmas.maw.maw import MAW

from settings import HotpotQASettings

_log = get_logger("fmbench.hotpot_qa.gen_seed")


# Abstract task description. Deliberately avoids naming any specific entities —
# if we pass a concrete example, the generator bakes entities like
# "Iqbal F. Qadir, Dwarka" into the prompts and the seed becomes unusable on
# the rest of the dataset. This description captures the shape of the task
# (input format, multi-hop nature, output style) without overspecialisation.
_TASK_DESCRIPTION = """Multi-hop question answering over Wikipedia paragraphs.

INPUT FORMAT:
- 10 Wikipedia paragraphs, each numbered [N] with a Title and body text.
  Two paragraphs are genuinely relevant to the question; eight are distractors.
- A question (prefixed with "Question:") that typically requires combining
  facts from multiple paragraphs to answer correctly.

OUTPUT: a short factual answer — a single entity, phrase, or number.
Only the answer itself, no explanation.

The task covers bridge questions (entity A is connected to B via shared
attribute), comparison questions (which of X, Y has property Z), and yes/no
questions. Answers are typically 1-10 words."""


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


async def main(settings: HotpotQASettings, *, two_stage: bool, output: Path) -> None:
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
    parser = argparse.ArgumentParser(description="Generate HotpotQA seed config")
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

    settings = HotpotQASettings()
    asyncio.run(
        main(settings, two_stage=not args.single_stage, output=args.output)
    )
