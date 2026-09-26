"""Настройки стенда: пути, ключи окружения, список моделей и инструментов.

Всё, что читается из окружения, собрано здесь — чтобы не искать по всему коду,
какие переменные влияют на запуск.
"""

from __future__ import annotations

import os
import secrets
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fedotmas.common.logging import get_logger

from .infra import ensure_searxng, lightpanda_ready

_log = get_logger("gui.config")

# Корень интерфейса: gui/. Пакет server/ лежит внутри него.
GUI_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = GUI_DIR / "static"

# Идентификатор запуска сервера: интерфейс сравнивает его со своим и очищает список
# сценариев, если сервер перезапускали. Показ должен начинаться с чистого листа.
SERVER_RUN_ID = uuid.uuid4().hex

# .env с ключом провайдера ищем в нескольких местах: в корне репозитория (gui/
# лежит внутри FEDOT.MAS), рядом с интерфейсом и в текущей директории. Так сервер
# запускается откуда угодно и переживает перенос каталога.
for _candidate in (Path.cwd() / ".env", GUI_DIR.parent / ".env",
                   GUI_DIR.parent / "FEDOT.MAS" / ".env", GUI_DIR / ".env"):
    if _candidate.is_file():
        load_dotenv(_candidate, override=False)
        _log.info("Ключи прочитаны из {}", _candidate)
        break

# --- Доступ и ключ провайдера ------------------------------------------------
# Публичный режим (GUI_PUBLIC=1) предназначен для случая, когда сервер отдан наружу
# через туннель или облако. В нём меняются две вещи:
#   * ключ из .env не используется вовсе — свой ключ вводит сам пользователь,
#     поэтому чужие запуски тратят чужие деньги, а не деньги владельца стенда;
#   * каждый запрос к /api/ обязан нести токен доступа, который печатается при
#     старте и вшивается в ссылку. Без него открытый адрес был бы открытым краном
#     к песочнице с исполнением кода.
PUBLIC_MODE = os.getenv("GUI_PUBLIC", "").strip().lower() not in ("", "0", "false", "no", "off")
ACCESS_TOKEN = os.getenv("GUI_ACCESS_TOKEN") or secrets.token_urlsafe(18)

if PUBLIC_MODE:
    # Убираем ключи владельца из окружения процесса до того, как их успеет прочитать
    # FEDOT.MAS: иначе публичная ссылка означала бы публичный доступ к его балансу.
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)

DEFAULT_MODEL = os.getenv("GUI_MODEL", "host/gpt-5.6-terra")
JUDGE_MODEL = os.getenv("GUI_JUDGE_MODEL", "host/gpt-5.6-terra")

MODELS = [
    {"id": "host/gpt-5.6-terra", "label": "GPT-5.6 Terra · подписка Codex", "open": False},
    {"id": "host/gpt-5.6-sol", "label": "GPT-5.6 Sol · подписка Codex", "open": False},
    {"id": "host/gpt-5.6-luna", "label": "GPT-5.6 Luna · подписка Codex", "open": False},
]

# Модели OpenRouter: litellm понимает префикс openrouter/ сам, нужен только
# OPENROUTER_API_KEY (из .env или окна выбора модели).
# Идентификаторы: https://openrouter.ai/api/v1/models.
# Каталог OpenRouter и лицензии карточек разработчиков проверены 2026-09-26.
# Список виден и без ключа: ключ вводится отдельно в окне выбора модели.
_OPEN_SOURCE_OPENROUTER = {
    "qwen/qwen3-235b-a22b-2507": "Qwen3 235B A22B · Apache 2.0",
    "deepseek/deepseek-v3.2": "DeepSeek V3.2 · MIT",
    "z-ai/glm-5": "GLM-5 · MIT",
    "mistralai/mistral-small-2603": "Mistral Small 4 · Apache 2.0",
}
MODELS += [
    {"id": "openrouter/" + slug, "label": label + " · OpenRouter", "open": True}
    for slug, label in _OPEN_SOURCE_OPENROUTER.items()
]

# Рассуждающие модели тратят часть лимита на размышления: с запасом по умолчанию
# агенты не обрываются на середине ответа.
AGENT_MAX_OUTPUT_TOKENS = int(os.getenv("GUI_AGENT_MAX_OUTPUT_TOKENS", "12000"))

# Мета-агент по умолчанию называет агентов по-английски; на защите просили меньше
# английского текста на экране.
RU_HINT = (
    "\n\nВажно: имена агентов, их инструкции и весь текст в конфигурации — на русском языке. "
    "Имена агентов пиши строчными буквами через подчёркивание (например, анализатор_телеметрии)."
)
# Локальные инструменты без внешних ключей и сервисов — проверены запуском.
BASE_TOOLS = (
    ["sandbox-light"]       # быстрые расчёты: только builtins
    # «sandbox» работает через E2B и без E2B_API_KEY возвращает агенту ошибку вместо расчёта.
    # Проверено запуском: sandbox-light считает, sandbox падает с 'E2B_API_KEY'. Без ключа
    # не предлагаем его вовсе — иначе мета-агент выдаёт его расчётному агенту единственным
    # инструментом, и тот остаётся без работающего калькулятора.
    + (["sandbox"] if os.getenv("E2B_API_KEY") else [])
    + [
        "sequential-thinking",  # пошаговый разбор сложных задач
        "document",             # чтение PDF, DOCX, XLSX, CSV и архивов
        "download",             # скачивание файлов по ссылке
        "media",                # разбор изображений, аудио и видео (через тот же ключ провайдера)
    ]
)
# Два прикладных демо доступны сразу. Остальные тяжёлые или нишевые
# инструменты по-прежнему добавляются через GUI_EXTRA_TOOLS.
EXTRA_TOOLS = [
    t.strip()
    for t in os.getenv(
        "GUI_EXTRA_TOOLS",
        "rubber-recipe-predictor,technology-card-audit",
    ).split(",")
    if t.strip()
]
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:18888")

# Lightpanda ставится в ~/.local/bin, которого может не быть в PATH у процесса сервера
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(Path.home() / ".local" / "bin")

WEB_SEARCH = ensure_searxng(SEARXNG_URL)
SCRAPING = lightpanda_ready()
SAFE_TOOLS = (
    BASE_TOOLS
    + (["websearch-searxng"] if WEB_SEARCH else [])
    + (["web-scraping"] if SCRAPING else [])
    + EXTRA_TOOLS
)
# В публичном режиме стенд открыт всем, у кого есть ссылка, а «document» читает
# любой файл хоста по абсолютному пути и «download» пишет файл в любой каталог.
# Агент выполняет то, что попросил гость, — значит ссылка означала бы доступ к
# ~/.ssh и к записи в автозагрузку. Наружу их отдаём только по явному разрешению.
FS_TOOLS = ("document", "download")
ALLOW_FS_PUBLIC = os.getenv("GUI_PUBLIC_ALLOW_FS", "").strip().lower() in ("1", "true", "yes", "on")
if PUBLIC_MODE and not ALLOW_FS_PUBLIC:
    _dropped = [t for t in SAFE_TOOLS if t in FS_TOOLS]
    if _dropped:
        SAFE_TOOLS = [t for t in SAFE_TOOLS if t not in FS_TOOLS]
        _log.warning("Публичный режим: инструменты файловой системы отключены {} "
                     "(включить: GUI_PUBLIC_ALLOW_FS=1)", _dropped)

_log.info("Инструменты живого режима: {}", SAFE_TOOLS)

# Узлы пайплайна ADK называются seq_/par_/loop_ — это не агенты, и в ленте
# выполнения их показывать не нужно.
WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")

# Пульс SSE: туннели и обратные прокси рвут молчащее соединение.
SSE_HEARTBEAT = float(os.getenv("GUI_SSE_HEARTBEAT", "10"))
# Мета-агент временами отдаёт невалидную схему — столько раз пробуем заново.
GENERATE_ATTEMPTS = int(os.getenv("GUI_GENERATE_ATTEMPTS", "3"))

JUDGE_FALLBACK = os.getenv("GUI_JUDGE_FALLBACK", "host/gpt-5.6-terra")
JUDGE_MAX_TOKENS = int(os.getenv("GUI_JUDGE_MAX_TOKENS", "16000"))
JUDGE_RETRY_TIMEOUT = float(os.getenv("GUI_JUDGE_RETRY_TIMEOUT", "90"))

# Реестр готовых MCP-серверов, в котором ищет и сам FEDOT.MAS.
# Искать в реестре можно без ключа, а вот подключиться к найденному серверу — нет:
# все развёрнутые экземпляры отвечают 401 «Missing Authorization header». Проверено
# запросом. Без ключа серверы из реестра показываем, но помечаем как недоступные.
SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY", "").strip()
SMITHERY_SEARCH = "https://registry.smithery.ai/servers"
SMITHERY_DETAIL = "https://registry.smithery.ai/servers/{name}"
