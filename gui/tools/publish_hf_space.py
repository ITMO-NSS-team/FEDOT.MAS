"""Публикация автономной replay-копии стенда на Hugging Face Spaces.

    uv run python gui/tools/publish_hf_space.py [--repo user/name] [--public]

Собирает свежую автономную копию (build_offline.py), кладёт её в static-Space
как index.html вместе с README и печатает ссылку. Токен берётся из
HF_PERSONAL_TOKEN в .env репозитория (или из окружения). По умолчанию Space
создаётся приватным — публичным его делает флаг --public или настройки на HF.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GUI = Path(__file__).resolve().parent.parent
REPO_ROOT = GUI.parent
OFFLINE = GUI / "fedotmas-offline.html"

README = """---
title: FEDOT.MAS Demo
emoji: 🕸️
colorFrom: green
colorTo: blue
sdk: static
pinned: false
---

# FEDOT.MAS — демо-стенд (replay)

Автономная копия веб-стенда [FEDOT.MAS](https://github.com/ITMO-NSS-team/FEDOT.MAS):
мультиагентная система собирается по текстовой постановке задачи и решает её.
Все прогоны записаны заранее и воспроизводятся из журнала — ничего не считается
и никуда не отправляется. Живой режим доступен в репозитории (`gui/run.py`).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="jrzkaminski/fedot-mas-demo",
                    help="repo_id Space'а (default: %(default)s)")
    ap.add_argument("--public", action="store_true",
                    help="создать Space публичным (по умолчанию приватный)")
    args = ap.parse_args()

    token = os.getenv("HF_PERSONAL_TOKEN")
    if not token:
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
            token = os.getenv("HF_PERSONAL_TOKEN")
        except ImportError:
            pass
    if not token:
        sys.exit("HF_PERSONAL_TOKEN не найден ни в окружении, ни в .env")

    print("Собираю свежую автономную копию…")
    subprocess.run([sys.executable, str(GUI / "tools" / "build_offline.py")], check=True)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="space", space_sdk="static",
                    exist_ok=True, private=not args.public)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        (staging / "index.html").write_bytes(OFFLINE.read_bytes())
        (staging / "README.md").write_text(README, encoding="utf-8")
        api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=staging,
                          commit_message="Update replay demo")

    owner, name = args.repo.split("/", 1)
    print(f"Готово: https://huggingface.co/spaces/{args.repo}")
    print(f"Страница: https://{owner}-{name.replace('_', '-')}.static.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
