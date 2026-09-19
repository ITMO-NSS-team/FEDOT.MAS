"""Схемы запросов к API.

Одно место, где видно, что именно принимает каждый эндпоинт.
"""

from __future__ import annotations

from pydantic import BaseModel


class KeyIn(BaseModel):
    """Ключ провайдера, введённый пользователем в интерфейсе.

    Адрес провайдера сюда намеренно не принимается: он задаётся владельцем стенда
    в .env. Иначе любой посетитель мог бы перенаправить все вызовы моделей —
    и ключи следующих посетителей — на свой адрес.
    """
    key: str = ""


class CustomMCP(BaseModel):
    """Свой MCP-сервер по ссылке: FEDOT.MAS поддерживает HTTP-транспорт напрямую."""
    name: str
    url: str
    headers: dict[str, str] | None = None
    # "smithery" — сервер выбран из реестра: ключ к нему подставит сервер, браузеру
    # он не отдаётся. Пусто — сервер добавлен пользователем по своей ссылке.
    source: str | None = None


class GenerateIn(BaseModel):
    task: str
    custom_mcp: list[CustomMCP] | None = None
    query: str | None = None    # нужен, чтобы увидеть ссылки на файлы с данными
    kind: str = "maw"
    model: str | None = None
    tools: list[str] | None = None
    russian: bool = True
    web: bool = True


class RunIn(BaseModel):
    config: dict
    query: str
    kind: str = "maw"
    tools: list[str] | None = None
    model: str | None = None
    custom_mcp: list[CustomMCP] | None = None


class PrepareIn(BaseModel):
    text: str
    kind: str = "maw"
    model: str | None = None
    web: bool = True
    tools: list[str] | None = None


class BaselineIn(BaseModel):
    query: str
    model: str | None = None


class EffortIn(BaseModel):
    task: str
    config: dict | None = None
    kind: str = "maw"
    model: str | None = None


class JudgeIn(BaseModel):
    query: str
    system_answer: str
    single_answer: str
    model: str | None = None


class ExportIn(BaseModel):
    presets: list[dict]
