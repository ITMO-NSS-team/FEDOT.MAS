from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
APP_NAME = "rubber_recipe_mas_terra_replay"
EXPECTED_STATE_KEYS = {
    "formulation_output",
    "chemistry_review",
    "prediction_audit",
    "recipe_prediction",
}


def call_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_session(base_url: str, user_id: str, session_id: str) -> dict[str, Any]:
    path = f"/apps/{APP_NAME}/users/{user_id}/sessions/{session_id}"
    try:
        return call_json(base_url, "GET", path)
    except HTTPError as exc:
        if exc.code != 404:
            raise
    return call_json(base_url, "POST", path, {})


def run_sse(
    base_url: str, user_id: str, session_id: str, task: str
) -> list[dict[str, Any]]:
    payload = {
        "appName": APP_NAME,
        "userId": user_id,
        "sessionId": session_id,
        "newMessage": {"role": "user", "parts": [{"text": task}]},
        "streaming": True,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/run_sse",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    events: list[dict[str, Any]] = []
    with urlopen(request, timeout=30) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events


def extract_tool_prediction(session: dict[str, Any]) -> dict[str, Any] | None:
    for event in session["events"]:
        for part in event.get("content", {}).get("parts", []):
            function_response = part.get("functionResponse")
            if function_response is None:
                continue
            if function_response.get("name") != "predict_rubber_recipe":
                continue
            response = function_response.get("response", {})
            prediction = response.get("structuredContent")
            return prediction if isinstance(prediction, dict) else None
    return None


def verify_session(session: dict[str, Any], task: str) -> dict[str, Any]:
    events = session["events"]
    state = session["state"]
    first_text = events[0]["content"]["parts"][0]["text"] if events else ""
    last_text = events[-1]["content"]["parts"][0].get("text", "") if events else ""
    function_calls = [
        part["functionCall"]
        for event in events
        for part in event.get("content", {}).get("parts", [])
        if "functionCall" in part
    ]
    prediction = extract_tool_prediction(session)
    worker_names = [
        "formulation_specialist",
        "rubber_chemistry_reviewer",
        "prediction_auditor",
    ]
    worker_calls = [
        call for call in function_calls if call.get("name") in worker_names
    ]
    checks = {
        "event_count_is_13": len(events) == 13,
        "state_keys_match": set(state) == EXPECTED_STATE_KEYS,
        "task_utf8_preserved": first_text == task,
        "numerical_tool_was_called": sum(
            call.get("name") == "predict_rubber_recipe" for call in function_calls
        )
        == 1,
        "numerical_tool_returned_recipe": prediction is not None
        and prediction.get("recipe", {}).get("nr_smr20_phr") == 58.0
        and prediction.get("recipe", {}).get("carbon_black_n220_phr") == 60.0,
        "workers_received_tool_result": [
            call.get("name") for call in worker_calls
        ]
        == worker_names
        and all(
            "nr_smr20_phr" in call.get("args", {}).get("request", "")
            and "{tool_result}" not in call.get("args", {}).get("request", "")
            for call in worker_calls
        ),
        "result_is_recipe_prediction": (
            events[-1].get("author") == "rubber_recipe_master"
            and "Лабораторный кандидат SBR/NR/N220" in last_text
            and state.get("recipe_prediction") == last_text
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "session_id": session["id"],
        "event_count": len(events),
        "state_keys": sorted(state),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the Terra recipe GUI session")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--user-id", default="user")
    parser.add_argument("--session-id", default="terra_tool_recipe_result")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "terra_subscription_run" / "gui_verification.json",
    )
    args = parser.parse_args()

    task = (HERE / "task.md").read_text(encoding="utf-8")
    session = ensure_session(args.base_url, args.user_id, args.session_id)
    if not session["events"]:
        run_sse(args.base_url, args.user_id, args.session_id, task)
        session = ensure_session(args.base_url, args.user_id, args.session_id)

    verification = verify_session(session, task)
    prediction = extract_tool_prediction(session)
    if prediction is not None:
        prediction_path = HERE / "terra_subscription_run" / "model_prediction.json"
        prediction_path.write_text(
            f"{json.dumps(prediction, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )
    rendered = json.dumps(verification, ensure_ascii=False, indent=2)
    args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    if verification["status"] != "pass":
        raise SystemExit("GUI session verification failed")


if __name__ == "__main__":
    main()
