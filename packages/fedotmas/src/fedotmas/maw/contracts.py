"""Static checks for the producer/consumer paths in generated handoffs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedotmas.maw.models import MAWConfig


def field_path_is_guaranteed(required: str, declared: str) -> bool:
    """A checked descendant or list expansion guarantees its nonempty container.

    Keep list/object boundaries intact: ``urls[]`` implies ``urls``, but a
    container alone does not guarantee its items or any particular child.
    """
    return (
        declared == required
        or declared == required + "[]"
        or declared.startswith((required + ".", required + "[]."))
    )


def handoff_contract_errors(config: MAWConfig) -> list[str]:
    """Check declared paths without guessing, renaming, or dropping required data.

    A container requirement may be satisfied by a declared child path. The
    converse is not guaranteed: declaring ``locations`` says nothing about
    whether every item has ``locations[].place_name``.
    """
    producers = {agent.output_key: agent for agent in config.agents}
    errors: list[str] = []
    for consumer in config.agents:
        for requirement in consumer.input_requirements:
            source = requirement.source_key
            if source == "user_query":
                errors.append(
                    f"Agent '{consumer.name}': user_query is plain task text, not a "
                    "structured handoff. Reference it in the instruction instead "
                    "of input_requirements."
                )
                continue
            producer = producers.get(source)
            if producer is None:
                errors.append(
                    f"Agent '{consumer.name}' requires unknown source '{source}'."
                )
                continue
            contract = producer.output_contract
            if contract is None:
                errors.append(
                    f"Producer '{producer.name}' needs an output_contract for '{source}', "
                    f"which consumer '{consumer.name}' requires as a structured artifact."
                )
                continue
            declared = set(contract.required_fields) | set(contract.identity_fields)
            needed = set(requirement.required_fields) | set(requirement.identity_fields)
            missing = sorted(
                path
                for path in needed
                if not any(field_path_is_guaranteed(path, field) for field in declared)
            )
            if missing:
                errors.append(
                    f"Handoff '{source}': consumer '{consumer.name}' requires paths "
                    f"{missing}, but producer '{producer.name}' guarantees {sorted(declared)}. "
                    "Make both contracts agree on the actual JSON shape; for list items "
                    "use paths such as locations[].place_name. Do not remove necessary "
                    "fields merely to silence validation."
                )
    return errors
