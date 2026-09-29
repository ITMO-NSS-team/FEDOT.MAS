# Пример интеграции с ProtoLLM

Зависимость от [aimclub/ProtoLLM](https://github.com/aimclub/protollm)
зафиксирована на коммите `c61708b0031c7fa668862ec805541daaef436397`.
Пример использует настоящий `create_llm_connector` из ProtoLLM для независимой
текстовой проверки отчёта FEDOT.MAS через OpenRouter. Это отдельная интеграция,
не замена штатного судьи GUI и не свидетельство точности прогнозов.

Окружение отдельное: ProtoLLM включает зависимости RAG и LangChain, которые
не требуются основному GUI. Из корня репозитория (нужны uv и Python 3.12):

```sh
uv sync --project examples/protollm --locked
```

Установите `OPENROUTER_API_KEY` в окружении, например в PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "ваш ключ"
uv run --project examples/protollm examples/protollm/review_report.py examples/protollm/sample_report.txt
```

Вместо `sample_report.txt` можно передать сохранённый текст отчёта МАС в UTF-8.
`--model qwen/qwen3-32b` меняет модель. Вызов отправляет весь текст отчёта
в OpenRouter и оплачивается по тарифу выбранной модели; не передавайте секреты.
Ключ не записывается в файлы. Пример делает один запрос без автоматических повторов.
Образец отчёта намеренно содержит неуспешный расчёт: проверяющий не должен
объявлять его успешным. Внешний LLM-вызов не входит в автоматические тесты.

Проверка настоящего коннектора с имитацией HTTP, без ключа и оплаты:

```sh
uv run --project examples/protollm --locked examples/protollm/smoke_test.py
```
