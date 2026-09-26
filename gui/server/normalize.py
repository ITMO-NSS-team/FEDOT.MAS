"""Правка конфигурации, которую отдал мета-агент.

Инструкции агентам работают ненадёжно: модель их регулярно игнорирует. Всё, что
обязано выполняться, закреплено здесь структурно — порядком шагов и составом
инструментов. Каждая функция объясняет, какой сбой она чинит.
"""

from __future__ import annotations

import re
from dataclasses import replace

from fedotmas.common.logging import get_logger
from fedotmas.mcp import HttpMCPServer, StdioMCPServer, resolve_mcp_registry

from .config import SAFE_TOOLS, SMITHERY_API_KEY, WEB_SEARCH
from .prompts import DATA_SOURCE_HINT, TOOL_DESCRIPTIONS

_log = get_logger("gui.normalize")


def _node_agents(node) -> int:
    """Сколько агентов исполняется внутри узла — цена одной итерации цикла."""
    if node.type == "agent":
        return 1
    return sum(_node_agents(c) for c in node.children or [])


def _valid_name(name: str) -> str:
    """ADK требует, чтобы имя агента было корректным идентификатором Python.

    Мета-агент иногда вставляет дефисы и пробелы — тогда сборка падает на валидации.
    """
    cleaned = re.sub(r"\W", "_", (name or "").strip(), flags=re.UNICODE).strip("_")
    if not cleaned:
        cleaned = "agent"
    if cleaned[0].isdigit():
        cleaned = "a_" + cleaned
    return cleaned if cleaned.isidentifier() else "agent"


# Адреса, на которые допустимо отправлять ключ реестра. Реестр отдаёт развёрнутые
# серверы на двух доменах; всё остальное — чужое, ключ туда не уходит.
_SMITHERY_HOSTS = (".smithery.ai", ".run.tools")


def _builtin_names() -> set[str]:
    """Имена всех встроенных MCP-серверов, включая не запрошенные в этот раз."""
    try:
        return set(resolve_mcp_registry("all") or {})
    except Exception:          # реестр не собрался — лучше перестраховаться именами по умолчанию
        return set(SAFE_TOOLS)


def _is_smithery_url(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:         # «https://[::1].smithery.ai/» — не адрес, ключ не подставляем
        return False
    return host.endswith(_SMITHERY_HOSTS)


def _mcp_registry_for(tools: list[str], custom: list | None) -> tuple:
    """Собирает реестр для FEDOT.MAS: встроенные серверы плюс свои по ссылке.

    resolve_mcp_registry принимает готовый словарь, а HttpMCPServer — штатный тип
    самого FEDOT.MAS, поэтому свой сервер подключается без обходных путей.

    Возвращает пару (реестр, имена своих серверов). Имена нужны вызывающему:
    sanitize_config вычищает у агентов всё, чего нет в SAFE_TOOLS, и без этого
    списка подключённый сервер до агента не доходил — он оставался в пуле без дела.
    """
    registry = dict(resolve_mcp_registry(tools) or {})
    rubber = registry.get("rubber-recipe-predictor")
    if isinstance(rubber, StdioMCPServer) and "--directory" in rubber.args:
        directory = rubber.args[rubber.args.index("--directory") + 1]
        registry["rubber-recipe-predictor"] = replace(
            rubber,
            args=(
                "run",
                "--directory",
                directory,
                "python",
                "-m",
                "mcp_rubber_recipe_predictor.server",
            ),
        )
    if not custom:
        return registry, []
    added: list[str] = []
    for item in custom:
        name = _valid_name(item.name)
        # Имя своего сервера не должно перебивать встроенный: иначе запрос подменял бы
        # песочницу или чтение файлов чужим адресом, и агент получал бы оттуда любые
        # «результаты». Сверяем с ПОЛНЫМ списком встроенных, а не с выбранными в этом
        # запросе: иначе достаточно не просить инструмент, чтобы занять его имя.
        if name in registry or name in _builtin_names():
            name = _valid_name(f"custom_{name}")
            while name in registry:
                name += "_"
            _log.warning("Имя своего MCP-сервера занято встроенным, переименован в {}", name)
        headers = dict(item.headers or {})
        # Ключ реестра подставляем только на адреса самого реестра. Раньше признак
        # «source: smithery» приходил от клиента вместе с адресом — и любой запрос
        # заставлял сервер отправить ключ владельца куда угодно.
        if getattr(item, "source", None) == "smithery" and SMITHERY_API_KEY:
            if _is_smithery_url(item.url):
                headers.setdefault("Authorization", f"Bearer {SMITHERY_API_KEY}")
            else:
                _log.warning("Адрес {} не принадлежит реестру — ключ не подставлен", item.url)
        registry[name] = HttpMCPServer(
            url=item.url, headers=headers,
            description=f"Свой MCP-сервер: {item.url}", tags=("custom",))
        added.append(name)
        _log.info("Подключён свой MCP-сервер | имя={} адрес={} ключ={}",
                  name, item.url, "есть" if headers.get("Authorization") else "нет")
    return registry, added


def sanitize_config(config, kind: str, extra_tools: list[str] | None = None,
                    *, available_tools=None):
    """Приводит имена агентов и ключи состояния к виду, который принимает ADK.

    *extra_tools* — имена своих MCP-серверов: они разрешены наравне со встроенными,
    иначе чистка неизвестных инструментов выбросила бы их из агентов.
    """
    allowed_tools = set(SAFE_TOOLS) | set(extra_tools or [])
    if available_tools is not None:
        allowed_tools &= set(available_tools)
    agents = list(getattr(config, "agents", None) or ([config.coordinator] + list(config.workers)))
    renames: dict[str, str] = {}
    keys: dict[str, str] = {}

    for agent in agents:
        new = _valid_name(agent.name)
        if new != agent.name:
            renames[agent.name] = new
            agent.name = new
        key = getattr(agent, "output_key", None)
        if key:
            new_key = _valid_name(key)
            if new_key != key:
                keys[key] = new_key
                agent.output_key = new_key

    if keys:                                   # ссылки {ключ} в инструкциях должны совпадать
        for agent in agents:
            for old, new in keys.items():
                agent.instruction = agent.instruction.replace("{" + old, "{" + new)

    if renames and kind == "mas":
        # Координатор зовёт воркеров по имени прямо в тексте инструкции: без
        # переписывания текстов он продолжит звать несуществующий инструмент.
        for agent in agents:
            for old, new in renames.items():
                if agent.instruction:
                    agent.instruction = agent.instruction.replace(old, new)
                desc = getattr(agent, "description", None)
                if desc:
                    agent.description = desc.replace(old, new)

    if renames and kind != "mas":
        def walk(node):
            if node.type == "agent" and node.agent_name in renames:
                node.agent_name = renames[node.agent_name]
            for child in node.children or []:
                walk(child)
        walk(config.pipeline)

    if kind != "mas" and getattr(config, "pipeline", None) is not None:
        root = config.pipeline
        # Мета-агента просят не заворачивать весь пайплайн в цикл, но он это правило нарушает.
        # Последствия проверены на прогонах: агенты повторяют работу с накопленным контекстом,
        # а сборщик на втором круге не дополняет ответ, а сокращает его втрое.
        if root.type == "loop" and _node_agents(root) > 2:
            _log.warning("Цикл вокруг всего пайплайна выпрямлен | агентов={}", _node_agents(root))
            root.type = "sequential"
            root.max_iterations = None

        # Сборщику нужен не «поразмышлять», а перенести числа предметных агентов в итог.
        for agent in agents:
            if "sequential-thinking" in (agent.tools or []) and _ROLE_COLLECTOR.search(agent.name):
                agent.tools = [t for t in agent.tools if t != "sequential-thinking"]
                _log.info("У сборщика убран sequential-thinking | агент={}", agent.name)

        pool = {a.name for a in agents}       # узел на несуществующего агента роняет сборку
        missing: list[str] = []

        def prune(node):
            kept = []
            for child in node.children or []:
                if child.type == "agent" and child.agent_name not in pool:
                    missing.append(child.agent_name)
                    continue
                prune(child)
                kept.append(child)
            if node.children is not None:
                node.children = kept

        prune(config.pipeline)
        if missing:
            _log.warning("Из пайплайна убраны неизвестные агенты | {}", missing)

    dropped: dict[str, list[str]] = {}
    for agent in agents:                      # ссылка на неподнятый сервер роняет сборку целиком
        unknown = [t for t in (agent.tools or []) if t not in allowed_tools]
        if unknown:
            dropped[agent.name] = unknown
            agent.tools = [t for t in agent.tools if t in allowed_tools]
            agent.instruction += (
                "\n\nСледующие инструменты не подключены в этом запуске: "
                + ", ".join(unknown)
                + ". Не вызывай их и не утверждай, что получил от них результат. "
                "Если без них задачу решить нельзя, явно укажи ограничение; "
                "не подменяй отсутствующие результаты предположениями."
            )

    if renames or keys:
        _log.info("Имена приведены к идентификаторам | агенты={} ключи={}", renames, keys)
    if dropped:
        _log.warning("Недоступные инструменты убраны из конфигурации | {}", dropped)
    return config


_URL_RE = re.compile(r"https?://\S+")
_DATA_HINT_RE = re.compile(
    # Намеренно широко: лишний инструмент безвреден, а пропуск приводит к тому,
    # что агент уходит в веб-поиск и сочиняет данные вместо расчёта по файлу.
    r"скача|фактическ|датасет|dataset|\bcsv\b|выгрузк|суточн|наблюден|измерен|"
    r"временн\w+ ряд|ряд\w* (температур|значен|данн)|архив\w* (данн|наблюден|погод)",
    re.IGNORECASE)


def _ensure_data_tools(config, kind: str, text: str) -> None:
    """Если в задаче есть ссылка на файл с данными — выдаём инструменты скачивания и чтения.

    Проверено прогоном: без этого мета-агент оставляет агенту один веб-поиск, тот ищет
    «фактические температуры» в вебе вместо скачивания файла, и расчёт идёт по справочным
    значениям — ровно то, что задача запрещает.
    """
    # Ссылки может не быть вовсе: «найди ряд фактических температур» — тоже работа с данными,
    # и без download/document агент уходит в веб-поиск и подставляет справочные значения.
    if not (_URL_RE.search(text or "") or _DATA_HINT_RE.search(text or "")):
        return
    needed = [t for t in ("download", "document", "sandbox-light", "websearch-searxng")
              if t in SAFE_TOOLS]
    agents = list(getattr(config, "agents", None) or list(config.workers))
    if not agents:
        return

    # Цель выбираем ПО ТЕМЕ, а не по тому, у кого инструменты уже есть: проверено прогоном —
    # мета-агент выдал download расчётчику, а температуры собирал другой агент с одним поиском,
    # и тот сделал 36 запросов к погодным агрегаторам вместо скачивания рядов.
    topic = re.compile(r"температур|погод|климат|метео|данн|ряд|файл|измер|статист|наблюден",
                       re.IGNORECASE)
    skip = re.compile(r"критик|провер|качеств|сборщик|итог|заключ", re.IGNORECASE)
    target = (next((a for a in agents if topic.search(a.name) and not skip.search(a.name)), None)
              or next((a for a in agents if "download" in (a.tools or []) and not skip.search(a.name)), None)
              or next((a for a in agents if not skip.search(a.name)), None) or agents[0])
    target.tools = list(dict.fromkeys(list(target.tools or []) + needed))
    # Раньше здесь стоял ранний выход, если мета-агент уже раздал нужные инструменты. Из-за него
    # порядок работы с файлом и справочник источников не дописывались — агент имел download,
    # но уходил в веб-поиск и сочинял данные. Инструкция важнее самих инструментов.
    if "archive-api.open-meteo" in (target.instruction or ""):
        return
    target.instruction = (target.instruction or "") + (
        "\n\nРабота с файлом данных обязательна и выполняется строго так:\n"
        "1) вызови download со ссылкой из запроса; в ответе он вернёт ПУТЬ к сохранённому файлу;\n"
        "2) вызови document ровно с этим путём (не придумывай имя файла) и обязательно "
        "с max_lines=200, иначе таблица придёт обрезанной до 20 строк;\n"
        "3) перенеси значения из прочитанной таблицы в код списком литералов и посчитай в "
        "песочнице sandbox-light. Песочница не видит файлы и не умеет скачивать — она считает "
        "только по значениям, которые ты впишешь в код. Это не повод отказываться от расчёта.\n"
        "Если ссылки на файл в задаче НЕТ, возьми её из справочника проверенных открытых "
        "источников ниже, подставив свои координаты и даты. Ряды суточной погоды (температура, "
        "осадки, ветер) за любой прошедший период — архив Open-Meteo, отдаёт чистый CSV без ключа:\n"
        "https://archive-api.open-meteo.com/v1/archive?latitude=59.94&longitude=30.31"
        "&start_date=2024-01-01&end_date=2024-01-31&daily=temperature_2m_mean"
        "&timezone=Europe%2FMoscow&format=csv\n"
        "(latitude/longitude — координаты нужного города, start_date/end_date — период, "
        "daily=temperature_2m_mean — суточная средняя температура.)\n"
        "Веб-поиск файл не заменяет: справочные и нормативные значения вместо "
        "фактических данных из файла подставлять запрещено. Если файл получить не удалось, найди "
        "те же ФАКТИЧЕСКИЕ значения за нужный период в открытом архиве наблюдений через веб-поиск "
        "и укажи источник; подставлять климатическую норму или справочное среднее вместо факта "
        "нельзя. Если не вышло и это — прямо напиши, что фактические данные получить не удалось, "
        "и не выдумывай числа.\n"
        "Полученные величины выведи явным списком «показатель = значение с единицей», чтобы "
        "следующие агенты брали их дословно, а не пересчитывали по памяти.\n"
        "ОБЯЗАТЕЛЬНО положи в свой ответ САМ РЯД значений (список чисел с датами) и точный "
        "адрес выгрузки: следующие агенты не имеют доступа к скачанному файлу и без ряда "
        "в тексте начинают сочинять свой собственный.\n"
        "Если ряд длинный (больше 40 значений), целиком его не печатай, но ОБЯЗАТЕЛЬНО приведи: "
        "точный адрес выгрузки, число полученных строк, первые и последние три строки с датами, "
        "минимум и максимум с датами и таблицу помесячных (или понедельных) агрегатов. "
        "Без этих доказательств происхождения проверяющий засчитает твой ряд за выдуманный.\n"
        "ВСЕ агрегаты по ряду считаешь ТЫ САМ и только вызовом песочницы sandbox-light: "
        "среднее, минимум и его дату, счётчики по порогам, суммы, градусо-сутки. "
        "Тут же проверь тождество «сумма (20 − t) = N × (20 − среднее)» и приведи обе стороны "
        "равенства числами. Соседние агенты ряд не пересчитывают — если ты посчитаешь на глаз, "
        "ошибка уйдёт в итог целиком. Опубликуй агрегаты списком «показатель = значение».")
    _log.info("Агенту выданы инструменты работы с файлом | агент={} инструменты={}", target.name, needed)

    _ensure_data_first(config, kind, target.name)

    # Жёсткая передача данных: ADK сам подставит содержимое состояния вместо {ключ?}.
    # Просить агента «взять ряд у соседа» бесполезно — состояние он видит не всегда,
    # а увидев пустоту, начинает сочинять собственный ряд.
    key = getattr(target, "output_key", None)
    if key:
        for agent in agents:
            if agent is target:
                continue
            if "{" + key in (agent.instruction or ""):
                continue
            if _ROLE_CRITIC.search(agent.name) or _ROLE_COLLECTOR.search(agent.name):
                # Сборщик и критик числа не считают, но именно на них ряд «портится»:
                # проверено прогоном — у добытчика ГС 935, в итоговой таблице 992.
                agent.instruction = (agent.instruction or "") + (
                    "\n\nАГРЕГАТЫ И РЯД ОТ АГЕНТА «" + target.name + "» (подставляются "
                    "автоматически):\n{" + key + "?}\n"
                    "Каждое число в итоге сверь с этим блоком. Любое расхождение — ошибка "
                    "переноса: бери значение отсюда, а не из промежуточных пересказов. "
                    "Пересчитывать ряд заново запрещено. Не переокругляй: переноси значения "
                    "с той точностью, с какой они посчитаны, иначе итог разойдётся с расчётом.")
                continue
            agent.instruction = (agent.instruction or "") + (
                "\n\nДАННЫЕ, ДОБЫТЫЕ АГЕНТОМ «" + target.name + "» (подставляются сюда "
                "автоматически):\n{" + key + "?}\n"
                "Агрегаты по этому ряду (среднее, минимум, счётчики, суммы) уже посчитаны "
                "в песочнице агентом-добытчиком — бери их ГОТОВЫМИ и не пересчитывай ряд "
                "заново ни в уме, ни в коде. Свой ряд наблюдений писать запрещено. "
                "Если блок выше пуст — данных нет: так и напиши, не выдумывай их.")
        _log.info("Ряд добытчика прокинут в инструкции считающих | ключ={}", key)


# Роли агентов определяем по имени: мета-агент называет их по-русски и осмысленно.
_ROLE_CRITIC = re.compile(r"критик|провер|качеств|аудит|реценз|валид", re.IGNORECASE)
_ROLE_COLLECTOR = re.compile(r"сборщик|сборка|агрегатор|агрегац|итог|заключ|финал|обобщ", re.IGNORECASE)


_ID_RE = re.compile(
    r"кадастров|\bИНН\b|\bОГРН\b|\bОКПО\b|\bВИН\b|\bVIN\b|артикул|госномер|"
    r"\bISBN\b|ГОСТ\s*\d|СП\s*\d|реестр|выписк", re.IGNORECASE)


def _ensure_lookup_tools(config, kind: str, text: str) -> None:
    """Протокол работы с конкретным объектом: идентификатор -> страницы -> параметры.

    Проверено прогоном (урбанистика): поиск находил страницу с адресом и площадью участка,
    но агент работал по сниппетам и до параметров не добирался — в итог шли типовые значения
    и честная ничья. Сниппеты не содержат цифр; страницы нужно открывать."""
    if not _ID_RE.search(text or ""):
        return
    agents = list(getattr(config, "agents", None) or list(config.workers))
    if not agents:
        return
    skip = re.compile(r"критик|провер|качеств|сборщик|итог|заключ", re.IGNORECASE)
    topic = re.compile(r"поиск|данн|объект|реестр|правов|огранич|градостро|контрагент|провер",
                       re.IGNORECASE)
    target = (next((a for a in agents if topic.search(a.name) and not skip.search(a.name)), None)
              or next((a for a in agents if "websearch-searxng" in (a.tools or [])
                       and not skip.search(a.name)), None)
              or next((a for a in agents if not skip.search(a.name)), None) or agents[0])
    needed = [t for t in ("websearch-searxng", "web-scraping") if t in SAFE_TOOLS]
    target.tools = list(dict.fromkeys(list(target.tools or []) + needed))
    target.instruction = (target.instruction or "") + (
        "\n\nВ задаче назван конкретный объект с идентификатором. Работай по протоколу:\n"
        "0) ПЕРВЫМ ДЕЙСТВИЕМ установи, ЧТО это за объект и ГДЕ он: найди поиском по полному "
        "идентификатору и выпиши адрес, населённый пункт, район и регион, приведя цитату из "
        "найденного текста и ссылку. Пока объект не опознан, искать ограничения и нормативы "
        "бессмысленно — привяжешь чужие правила. Опубликуй эту привязку отдельным блоком "
        "в начале своего ответа;\n"
        "1) затем ищи остальное по полному идентификатору, а если пусто — по его частям "
        "(кадастровый квартал, начало номера, название);\n"
        "2) ВНИМАТЕЛЬНО прочитай сниппеты верхних результатов: адрес, площадь и статус объекта "
        "часто уже там — тогда бери их оттуда со ссылкой. Если в сниппетах нужного нет, ОТКРОЙ "
        "2–3 страницы С ВЕРХА выдачи (в первую очередь те, чей сниппет уже упоминает твой "
        "объект) и вытащи параметры из текста. Интерактивные карты (адреса со словами map, "
        "kadastrovaya-karta, публичная карта) не открывай: они рисуются скриптами и текста "
        "не отдают, время потратишь зря;\n"
        "2а) регион, адрес и принадлежность объекта бери ТОЛЬКО из найденного источника. "
        "Не выводи их из общих соображений и не подставляй соседний регион: ошибка в регионе "
        "тянет за собой неверные нормативы и обесценивает весь ответ;\n"
        "3) выпиши найденное списком «параметр = значение — источник (ссылка)».\n"
        "Только если страницы не открылись или параметров там нет — скажи это прямо и переходи "
        "к типовым значениям с пометкой. Типовые значения вместо непрочитанных страниц — ошибка.\n"
        "ЕСЛИ ПАРАМЕТР ЗАКРЫТ (платная версия, только по выписке ЕГРН, нет в открытых источниках) — это ТОЖЕ результат поиска, и его надо зафиксировать: напиши, какой именно параметр недоступен, где ты это увидел и со ссылкой. Затем не выдумывай одно число, а посчитай сценарно: возьми 2–3 правдоподобных значения параметра, доведи расчёт до чисел для каждого и покажи, как меняется вывод. Доказанная недоступность плюс сценарный расчёт сильнее одного выдуманного значения.")
    _log.info("Включён протокол работы с объектом | агент={}", target.name)


def _node_agent_names(node) -> list[str]:
    """Имена агентов внутри узла пайплайна, в порядке обхода."""
    if node.type == "agent":
        return [node.agent_name]
    out: list[str] = []
    for child in node.children or []:
        out.extend(_node_agent_names(child))
    return out


def _drop_agent_node(node, name: str) -> bool:
    """Убирает узел агента из дерева. True, если убрали."""
    kept, found = [], False
    for child in node.children or []:
        if child.type == "agent" and child.agent_name == name:
            found = True
            continue
        if _drop_agent_node(child, name):
            found = True
        if child.type == "agent" or (child.children or []):
            kept.append(child)          # пустые группы не тащим дальше
    if node.children is not None:
        node.children = kept
    return found


def _ensure_data_first(config, kind: str, data_agent: str) -> None:
    """Ставит агента-добытчика первым шагом пайплайна.

    Проверено логами: мета-агент ставил расчётчика в одну параллельную ветку с добытчиком,
    и расчёт стартовал раньше, чем появились данные. Агент, не увидев ряда, писал свой
    выдуманный прямо в коде песочницы — «примерные данные», «заглушка».
    """
    if kind == "mas" or getattr(config, "pipeline", None) is None:
        return
    root = config.pipeline
    names = _node_agent_names(root)
    if not names or names[0] == data_agent or data_agent not in names:
        return

    node_cls = type(root)
    if not _drop_agent_node(root, data_agent):
        return
    step = node_cls(type="agent", agent_name=data_agent)
    if root.type == "sequential":
        root.children = [step] + list(root.children or [])
    else:
        config.pipeline = node_cls(type="sequential", children=[step, root])
    _log.info("Агент-добытчик поставлен первым шагом | агент={}", data_agent)


# Роли, которые работают ПО чужим результатам: планировать, приоритизировать и
# рекомендовать можно только после того, как кто-то посчитал.
_ROLE_DEPENDENT = re.compile(r"планир|приоритиз|рекоменд|стратег|синтез|ранжир|решени",
                             re.IGNORECASE)


def _ensure_dependent_after_parallel(config, kind: str) -> None:
    """Уводит зависимого агента из параллельной ветки за неё.

    Наблюдено в журнале: агент планирования обслуживания стоял в одной параллельной
    ветке с агентом анализа параметров и стартовал одновременно с ним. На вход ему
    пришло 795 токенов вместо данных — план он написал вслепую, а весь прогон выдал
    2 945 токенов вместо привычных полутора сотен тысяч. Это тот же класс ошибки, что
    уже лечит _ensure_data_first, только про другую роль.
    """
    if kind == "mas" or getattr(config, "pipeline", None) is None:
        return
    root = config.pipeline
    node_cls = type(root)
    if root.type == "parallel":
        root = node_cls(type="sequential", children=[root])
        config.pipeline = root
    if root.type != "sequential":
        return

    children, moved = [], []
    for child in list(root.children or []):
        children.append(child)
        if child.type != "parallel":
            continue
        names = _node_agent_names(child)
        dependent = [n for n in names if _ROLE_DEPENDENT.search(n)
                     and not _ROLE_COLLECTOR.search(n) and not _ROLE_CRITIC.search(n)]
        # Уводить всех нельзя: в ветке должен остаться хоть кто-то, кто добывает данные.
        if not dependent or len(dependent) >= len(names):
            continue
        for name in dependent:
            if _drop_agent_node(child, name):
                children.append(node_cls(type="agent", agent_name=name))
                moved.append(name)
    if moved:
        root.children = children
        _log.info("Зависимые агенты выведены из параллельной ветки | агенты={}", moved)


def _ensure_calculator(config, kind: str) -> None:
    """Даёт песочницу всем, кто работает с числами, и снимает то, что счёт подменяет.

    Класс ошибок, найденный прогонами: возможность оказывается не у того агента, которому
    она нужна. Данные достались одному агенту, песочница — соседнему, и суммы считались
    в уме: ошибки в градусо-сутках, в числе суток и в переводе МВт·ч в Гкал. Инструмент
    sequential-thinking у расчётного агента уводит в рассуждения вместо вычислений.
    Сборщик — единственный, кому песочница не нужна: он переносит чужие числа, а не считает.
    """
    if "sandbox-light" not in SAFE_TOOLS:
        return
    agents = list(getattr(config, "agents", None) or list(config.workers))

    for agent in agents:
        is_critic = bool(_ROLE_CRITIC.search(agent.name))
        is_collector = bool(_ROLE_COLLECTOR.search(agent.name)) and not is_critic
        tools = list(agent.tools or [])

        if not is_collector and "sandbox-light" not in tools:
            tools.append("sandbox-light")
            agent.instruction = (agent.instruction or "") + (
                "\n\nОтдельно проверь СОГЛАСОВАННОСТЬ итога: не противоречат ли выводы и ранжирования собственным таблицам ответа, совпадают ли числа в тексте и в таблицах. Расхождение таблицы с выводом — такой же тяжёлый дефект, как арифметическая ошибка.\n"
                "Пересчитай ключевые числа предшественников в песочнице sandbox-light и укажи "
                "каждое расхождение: сходятся ли суммы, следует ли вывод из приведённых чисел. "
                "Отдельно проверь РАЗМЕРНОСТИ: подставь единицы входящих величин в формулу и "
                "убедись, что результат действительно имеет ту единицу, которой подписан. "
                "Ватты, умноженные на часы, дают Вт·ч, а не Гкал: перевод обязателен "
                "(1 Гкал = 1163 кВт·ч = 1,163 МВт·ч). Отдельно проверь порядок величины: если из градусо-суток (°C·сут) получают энергию, обязан присутствовать множитель 24 ч/сут; его отсутствие занижает итог ровно в 24 раза. Оценивать числа на глаз запрещено."
                if is_critic else
                "\n\nВсе арифметические действия выполняй в песочнице sandbox-light, а не в уме: "
                "суммы, средние, доли, переводы единиц и итоговые показатели. Приводи и результат, "
                "и числа, из которых он получен. Код пиши только на Python: JavaScript "
                "(const, =>, toFixed) песочница не исполняет.\n"
                "Данные для расчёта бери ТОЛЬКО из состояния — из того, что уже добыли предыдущие "
                "агенты. Собственный ряд наблюдений в коде писать запрещено: комментарии вида "
                "«примерные данные», «для простоты», «заглушка», «эмуляция» означают выдуманный "
                "ряд, и весь расчёт поверх него недействителен. Если нужного ряда в состоянии нет, "
                "так и напиши, что данных нет, вместо того чтобы их придумать.\n"
                "Ни одно число не должно появляться в расчёте без объявления: СНАЧАЛА выпиши все "
                "исходные величины списком «величина = значение (единица) — статус», где статус "
                "это «установлено по источнику» со ссылкой, «принято типовым для расчёта» или "
                "«требует подтверждения», и только потом считай. Скрытое допущение, от которого "
                "ведётся расчёт, обесценивает результат: проверяющий не поймёт, откуда взялось "
                "итоговое число.\n"
                "Проверки тождеств выполняй с неокруглёнными значениями и только в таблице "
                "округляй: проверка с округлённой средней даёт ложное расхождение, и её засчитают "
                "как ошибку.\n"
                "Выписывай единицы рядом с каждым множителем и сокращай их — так ловится потерянный множитель.\n"
                "Частая ошибка: градусо-сутки имеют размерность °C·СУТКИ, а не °C·часы. Энергия = мощность × время, поэтому нужен переход к часам: Вт × °C·сут × 24 ч/сут ÷ 10⁶ = МВт·ч. Без множителя 24 результат занижается ровно в 24 раза.\n"
                "Прикинь порядок итога независимым грубым способом и сравни с расчётом: расхождение больше чем в 2 раза означает потерянный или лишний множитель — ищи его, а не публикуй результат.\n"
                "Перед выводом проверяй размерность: подставь единицы в формулу и убедись, что "
                "результат имеет заявленную единицу. Если формула из задачи даёт другую единицу, "
                "чем подписано в задаче, приведи обе величины и явный перевод, а не подписывай "
                "результат чужой единицей. Ориентиры: 1 Гкал = 1163 кВт·ч = 1,163 МВт·ч; "
                "Вт × ч = Вт·ч.")
            _log.info("Агенту выдана песочница | агент={} роль={}", agent.name,
                      "критик" if is_critic else "расчёт")

        # Считающему агенту sequential-thinking вредит: он рассуждает вместо вычислений
        if "sandbox-light" in tools and "sequential-thinking" in tools:
            tools = [t for t in tools if t != "sequential-thinking"]
            _log.info("У считающего агента снят sequential-thinking | агент={}", agent.name)

        agent.tools = list(dict.fromkeys(tools))


def _ensure_web_tool(config, kind: str) -> None:
    """Мета-агент иногда забывает назначить поиск — выдаём его подходящему агенту сам."""
    agents = list(getattr(config, "agents", None) or list(config.workers))
    if any("websearch-searxng" in (a.tools or []) for a in agents):
        return

    prefer = re.compile(r"поиск|источник|ограничен|контекст|сбор|данн|исслед|research|search", re.IGNORECASE)
    skip = re.compile(r"критик|валид|провер|сборщик|агрегатор|заключ|итог", re.IGNORECASE)
    target = next((a for a in agents if prefer.search(a.name) and not skip.search(a.name)), None)
    target = target or next((a for a in agents if not skip.search(a.name)), None) or (agents[0] if agents else None)
    if target is None:
        return

    target.tools = list(target.tools or []) + ["websearch-searxng"]
    target.instruction += ("\n\nОбязательно используй инструмент веб-поиска для внешних сведений "
                           "и приводи ссылку на источник каждого утверждения.")
    _log.info("Веб-поиск добавлен агенту {}", target.name)


def _tools_hint(web: bool = True, tools: list[str] | None = None) -> str:
    allowed = tools if tools is not None else SAFE_TOOLS
    names = [t for t in allowed if t != "websearch-searxng"]
    parts = [TOOL_DESCRIPTIONS.get(t, t) for t in names]
    if web and WEB_SEARCH and "websearch-searxng" in allowed:
        parts.insert(0, "websearch-searxng — веб-поиск. ОБЯЗАТЕЛЬНО назначь его хотя бы одному агенту, "
                        "который собирает внешние сведения, и потребуй в его инструкции приводить ссылки на источники")
    return "; ".join(parts) + ". Других инструментов нет, назначай только эти и только там, где они нужны"


def _with_data_source(query: str, task: str) -> str:
    """Дописывает в запрос адрес открытого архива, если задача требует фактических данных.

    Измерено на прогонах: когда адрес источника назван в задаче, агент скачивает файл
    в 6 случаях из 6; когда не назван — в 1 из 8, остальные подменяют факты справочными
    значениями. Просить об этом мета-агента бесполезно — он игнорирует; поэтому кодом.
    """
    text = f"{query} {task}"
    if _URL_RE.search(text) or not _DATA_HINT_RE.search(text):
        return query
    _log.info("В запрос подставлен адрес открытого архива данных")
    return query + DATA_SOURCE_HINT
