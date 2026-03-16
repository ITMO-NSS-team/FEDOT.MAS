import asyncio

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig
from fedotmas_synapse.plugin import SynapsePlugin
from motor.motor_asyncio import AsyncIOMotorClient

_log = get_logger("fedotmas.examples.synapse.mongo_persistence")

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "fedotmas_demo"


async def main() -> None:
    # --- 1. Connect to MongoDB ---
    client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    _log.info("Connected to MongoDB: {}/{}", MONGO_URI, DB_NAME)

    # --- 2. Create SynapsePlugin (provides session + memory + checkpoint) ---
    plugin = SynapsePlugin(db=db, project_id="demo")

    # --- 3. Handcrafted MAW config: researcher → writer ---
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="researcher",
                instruction="Research the topic: {user_query}. Provide key facts.",
                output_key="research",
            ),
            MAWAgentConfig(
                name="writer",
                instruction=(
                    "Write a concise summary based on the research:\n\n{research}"
                ),
                output_key="summary",
            ),
        ],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[
                MAWStepConfig(type="agent", agent_name="researcher"),
                MAWStepConfig(type="agent", agent_name="writer"),
            ],
        ),
    )

    # --- 4. Run pipeline with MongoDB persistence ---
    task = "What is WebAssembly and why does it matter?"
    maw = MAW(**plugin.mas_kwargs())
    state = await maw.build_and_run(config, task)
    _log.info("Pipeline complete. State keys: {}", list(state.keys()))
    _log.info("Summary: {}", state.get("summary", "(no summary)"))

    # --- 5. Verify sessions in MongoDB ---
    session_count = await db["fedotmas_sessions"].count_documents({})
    _log.info("Sessions in MongoDB: {}", session_count)

    session_doc = await db["fedotmas_sessions"].find_one()
    if session_doc:
        _log.info("Session state keys: {}", list(session_doc.get("state", {}).keys()))
        _log.info("Events count: {}", len(session_doc.get("events", [])))

    # --- 6. Verify checkpoints ---
    cp_count = await db["fedotmas_checkpoints"].count_documents({})
    _log.info("Checkpoints in MongoDB: {}", cp_count)

    async for cp in db["fedotmas_checkpoints"].find().sort("timestamp", 1).limit(3):
        _log.info(
            "  checkpoint: agent={}, phase={}, keys={}",
            cp.get("agent_name"),
            cp.get("phase"),
            list(cp.get("state_snapshot", {}).keys()),
        )

    # --- 7. Memory: ingest session and search ---
    # NOTE: add_session_to_memory() must be called explicitly —
    # run_pipeline() does NOT do this automatically.
    # Retrieve the session we just ran from the session service.
    sessions_resp = await plugin.session_service.list_sessions(app_name="fedotmas")
    if sessions_resp.sessions:
        session = sessions_resp.sessions[-1]
        await plugin.memory_service.add_session_to_memory(session)
        _log.info("Session ingested into memory")

        result = await plugin.memory_service.search_memory(
            app_name=session.app_name,
            user_id=session.user_id,
            query="WebAssembly",
        )
        _log.info("Memory search results: {}", len(result.memories))
        for mem in result.memories[:2]:
            parts = mem.content.parts if mem.content else None
            text = parts[0].text if parts else None
            preview = text[:120] if text else ""
            _log.info("  memory: {}...", preview)

    mem_count = await db["fedotmas_memory"].count_documents({})
    _log.info("Memory docs in MongoDB: {}", mem_count)

    # --- 8. Cleanup ---
    client.close()
    _log.info("Done. Inspect with: mongosh {}", DB_NAME)


if __name__ == "__main__":
    asyncio.run(main())
