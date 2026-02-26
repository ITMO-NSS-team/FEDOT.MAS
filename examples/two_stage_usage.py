import asyncio
import json

from fedotmas import MAS


async def two_stage_auto():
    """Full-auto with two-stage generation (default behavior)."""
    mas = MAS(mcp_servers=[])
    state = await mas.run("Which language best for low-level Wasm?")
    print(json.dumps(state, indent=2, default=str))


async def two_stage_inspect_pool():
    """Generate pool first, inspect it, then generate pipeline."""
    from fedotmas.meta.pipeline_gen import PipelineGenerator
    from fedotmas.meta.pool_gen import PoolGenerator

    task = ""

    # Stage 1: generate pool
    pool_gen = PoolGenerator()
    pool = await pool_gen.generate(task)

    # Inspect / edit pool
    print(f"Generated {len(pool.agents)} agents:")
    for a in pool.agents:
        print(f"  - {a.name}: {a.instruction[:80]}...")

    # Stage 2: generate pipeline from pool
    pipeline_gen = PipelineGenerator()
    config = await pipeline_gen.generate(task, pool)
    print(config.model_dump_json(indent=2))

    # Execute
    mas = MAS()
    state = await mas.build_and_run(config, task)
    print(json.dumps(state, indent=2, default=str))


async def single_stage_fallback():
    """Explicitly use single-stage generation (legacy behavior)."""
    mas = MAS(two_stage=False)
    state = await mas.run("Simple question answering task")
    print(json.dumps(state, indent=2, default=str))


if __name__ == "__main__":
    # Pick one:
    asyncio.run(two_stage_auto())
    # asyncio.run(two_stage_inspect_pool())
    # asyncio.run(single_stage_fallback())
