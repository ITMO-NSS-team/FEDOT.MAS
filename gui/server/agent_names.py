"""Латинские имена агентов MAS на время запуска — русские остаются на экране.

В MAS каждый исполнитель становится у координатора инструментом с именем агента
(режим single_turn в ADK), а OpenAI принимает в имени инструмента только
[a-zA-Z0-9_-]. Кириллическое имя роняло первый же ход координатора:
400 «Invalid 'tools[3].name': string does not match pattern». MAW это не задевает —
там агенты инструментами не становятся.

Просить мета-агента о латинских именах нельзя: имена видны на экране, и по ним
normalize узнаёт роли (критик, сборщик). Поэтому сценарий хранит русские имена, а
запуск получает копию конфигурации с латинскими; события потока и итог переводятся
обратно через AgentNames.
"""

from __future__ import annotations

import re
import unicodedata

_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}
# Годится и в имя инструмента OpenAI (до 64 символов), и в идентификатор агента ADK
_TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_MAX_LEN = 48                      # запас под суффикс, если имена совпадут после транслитерации
# «user» ADK не принимает как имя агента, transfer_to_agent — его собственный инструмент
_RESERVED = {"user", "transfer_to_agent"}


def latin_name(name: str, taken: set[str]) -> str:
    """Транслитерирует имя в [a-z0-9_] и делает его уникальным среди *taken*."""
    text = "".join(_CYRILLIC.get(ch, ch) for ch in (name or "").lower())
    # Диакритику других алфавитов снимаем, всё остальное становится разделителем
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:_MAX_LEN].rstrip("_") or "agent"
    if slug[0].isdigit():
        slug = "a_" + slug
    candidate, n = slug, 2
    while candidate in taken or candidate in _RESERVED:
        candidate, n = f"{slug}_{n}", n + 1
    taken.add(candidate)
    return candidate


def _names_regex(names) -> re.Pattern:
    """Имя целиком — не кусок другого слова и не подстановка {ключ} состояния."""
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?<![\w{{])(?:{alternatives})(?![\w?}}])")


class AgentNames:
    """Имена запуска → имена сценария: по ним поток и итог показывают русские имена."""

    def __init__(self, shown: dict[str, str] | None = None) -> None:
        self.shown = dict(shown or {})
        self._renamed = {run: orig for run, orig in self.shown.items() if run != orig}
        self._regex = _names_regex(self._renamed) if self._renamed else None

    def is_agent(self, name: str) -> bool:
        return name in self.shown

    def name(self, name: str) -> str:
        return self.shown.get(name, name)

    def text(self, text: str) -> str:
        if not self._regex or not text:
            return text
        return self._regex.sub(lambda m: self._renamed[m.group(0)], text)


def latinize_mas(config) -> AgentNames:
    """Переводит имена агентов MASConfig в латиницу прямо в *config* — в копии для запуска.

    Ключи состояния не меняются. Исполнителю без output_key строитель FEDOT.MAS дал бы
    ключ «<имя>_output»; фиксируем его по исходному имени, иначе артефакт лёг бы под
    латинский ключ и интерфейс его не узнал бы.
    """
    agents = [config.coordinator, *config.workers]
    for worker in config.workers:
        worker.output_key = worker.output_key or f"{worker.name}_output"

    keep = {a.name for a in agents if _TOOL_NAME.fullmatch(a.name) and a.name not in _RESERVED}
    taken = set(keep)
    shown: dict[str, str] = {}
    for agent in agents:
        run_name = agent.name if agent.name in keep else latin_name(agent.name, taken)
        shown[run_name] = agent.name
        agent.name = run_name

    # Координатор зовёт исполнителей по именам из своей инструкции: они должны совпадать
    # с именами инструментов, иначе первый вызов уходит в несуществующий инструмент.
    renamed = {orig: run for run, orig in shown.items() if run != orig}
    if renamed:
        regex = _names_regex(renamed)
        for agent in agents:
            agent.instruction = regex.sub(lambda m: renamed[m.group(0)], agent.instruction or "")
            agent.description = regex.sub(lambda m: renamed[m.group(0)], agent.description or "")
    return AgentNames(shown)
