"""Схемы запросов к API.

Одно место, где видно, что именно принимает каждый эндпоинт.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    kind: str = "mas"
    model: str | None = None
    tools: list[str] | None = None
    russian: bool = True
    web: bool = True


class RunIn(BaseModel):
    config: dict
    query: str
    kind: str = "mas"
    tools: list[str] | None = None
    model: str | None = None
    custom_mcp: list[CustomMCP] | None = None


class PrepareIn(BaseModel):
    text: str
    kind: str = "mas"
    model: str | None = None
    web: bool = True
    tools: list[str] | None = None


class BaselineIn(BaseModel):
    query: str
    model: str | None = None


class EffortIn(BaseModel):
    task: str
    config: dict | None = None
    kind: str = "mas"
    model: str | None = None


class JudgeIn(BaseModel):
    query: str
    system_answer: str
    single_answer: str
    model: str | None = None


class SyntheticExamplesIn(BaseModel):
    """Запрос на тестовые переформулировки для оценщика качества."""

    query: str = Field(min_length=1, max_length=12_000)
    count: int = Field(default=3, ge=1, le=10)
    model: str | None = None


class ReviewIn(BaseModel):
    query: str
    system_answer: str
    trace: list[dict] = Field(default_factory=list)
    model: str | None = None


class ExportIn(BaseModel):
    presets: list[dict]
