import asyncio
import json

from fedotmas import MAW, AgentPoolConfig, to_synapse_bundle
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import AgentPoolEntry

_log = get_logger("fedotmas.examples.export.synapse_bundle")

# Tool ids and descriptions as the target platform lists them for its tenant.
# Generation offers the meta-agent these and nothing else, so the exported
# agents reference tools that exist on the other side.
TENANT_TOOLS = {
    "urbanprojects.getprojectbyid": "Get a Prostor project by id",
    "urbanindicators.getprovision": "Provision indicators for a territory",
}

TASK = "Assess school provision in a selected district and report the gaps"


async def generated_from_scratch():
    maw = MAW(tool_catalog=TENANT_TOOLS)
    config = await maw.generate_config(TASK)

    export = to_synapse_bundle(
        config,
        workflow_id="school_provisioning",
        workflow_name="School provisioning",
        tool_catalog=TENANT_TOOLS,
    )
    _log.info("Bundle: {}", json.dumps(export.bundle, indent=2, ensure_ascii=False))
    _log.info(
        "Linearized branches: {} | degraded loops: {} | unresolved tools: {}",
        export.linearized_branches,
        export.degraded_loops,
        export.unresolved_tools,
    )


async def over_existing_agents():
    # Agents the platform already has. `id` is their identifier there, so the
    # import updates those records instead of adding a second copy.
    pool = AgentPoolConfig(
        agents=[
            AgentPoolEntry(
                id="urban_requirements_gatherer",
                name="requirements_gatherer",
                instruction="Собери уточняющие вопросы по запросу: {user_query}",
            )
        ]
    )

    maw = MAW(tool_catalog=TENANT_TOOLS)
    config = await maw.generate_config(TASK, existing_agents=pool)

    export = to_synapse_bundle(
        config,
        workflow_id="school_provisioning",
        existing_agents=pool,
        tool_catalog=TENANT_TOOLS,
    )
    _log.info("Reused as given: {}", export.reused_agents)
    _log.info("Renamed, so imported as new records: {}", export.renamed_ids)
    _log.info("Their import overwrites: {}", export.overwritten_fields)


if __name__ == "__main__":
    asyncio.run(generated_from_scratch())
    # asyncio.run(over_existing_agents())
