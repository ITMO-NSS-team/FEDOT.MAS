"""Сборка автономной страницы: один HTML-файл, работающий без сервера.

    python3 build_offline.py [имя_файла]

Стили, скрипт и сценарии встраиваются внутрь. Файл можно открыть двойным кликом
или переслать — переключение сценариев, вкладки, воспроизведение и граф работают
офлайн. Живой режим в нём недоступен: для реальных запусков нужен run.py.
"""

from __future__ import annotations

import datetime as dt
import html
import pathlib
import sys

GUI = pathlib.Path(__file__).resolve().parent.parent
STATIC = GUI / "static"
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else GUI / "fedotmas-offline.html"


def inline(js: str) -> str:
    """Готовит JS к вклейке внутрь <script>.

    Любая строка сценария может содержать «</script>» — например, в тексте задачи.
    Без экранирования она закрывает тег досрочно: копия ломается, а подготовленное
    содержимое может дописать в страницу свою разметку.
    """
    return js.replace("</script", "<\\/script")


def read(name: str) -> str:
    # index.html, app.js и стили лежат в static/, заглушка бэкенда — рядом со сборщиком
    for base in (STATIC, pathlib.Path(__file__).resolve().parent):
        if (base / name).is_file():
            return (base / name).read_text(encoding="utf-8")
    raise FileNotFoundError(name)


def main() -> None:
    page = read("index.html")
    styles = read("styles.css")

    # внешние файлы заменяем встроенными блоками
    page = page.replace('<link rel="stylesheet" href="styles.css">',
                        "<style>\n" + styles + "\n</style>")

    # Инлайним каждый <script src="..."> из index.html по порядку — включая
    # кейс-пресеты (*_preset.js), которых сборщик раньше не знал. Суффикс ?v=N
    # нужен только против кеша браузера, при чтении файла его отбрасываем.
    import re

    def _inline_tag(m: re.Match) -> str:
        name = m.group(1).split("?", 1)[0]
        blocks = ""
        if name == "app.js":
            # Заглушка бэкенда идёт перед app.js: она подменяет fetch до первого
            # запроса, поэтому кнопки, форма сценария и вкладки работают без сервера.
            blocks += "<script>\n" + inline(read("mock_backend.js")) + "\n</script>\n"
        blocks += "<script>\n" + inline(read(name)) + "\n</script>"
        if name == "presets.js" and (STATIC / "presets_custom.js").exists():
            # Сценарии, созданные через форму, живут в localStorage:
            # без выгрузки они в копию не попадут
            blocks += "\n<script>\n" + inline(read("presets_custom.js")) + "\n</script>"
        return blocks

    page = re.sub(r'<script src="([^"]+)"></script>', _inline_tag, page)

    stamp = dt.datetime.now().strftime("%d.%m.%Y %H:%M")
    page = page.replace("</head>",
                        f"<!-- Автономная копия интерфейса FEDOT.MAS, собрана {html.escape(stamp)}. "
                        "Запросы имитируются заглушкой: ничего не считается и никуда не отправляется. -->\n</head>")

    assert "href=\"styles.css\"" not in page and "src=\"app.js\"" not in page, "остались внешние ссылки"
    OUT.write_text(page, encoding="utf-8")
    print(f"собрано: {OUT}  ({OUT.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
