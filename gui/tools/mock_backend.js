/* Имитация бэкенда для автономной копии страницы.
 *
 * Подменяет fetch до загрузки app.js и отвечает заготовленными данными на те же
 * адреса, что и run.py. Нужна только для показа: кнопки, форма своего
 * сценария, прогресс генерации, воспроизведение прогона, сравнение и вердикт
 * судьи работают, но ничего не считается и в модели не отправляется.
 */
(function () {
  "use strict";

  // Признак автономной копии: по нему интерфейс подхватывает записанные прогоны из
  // presets.js. На живом стенде этого флага нет, и список остаётся пустым — там
  // показывают только то, что создали сами.
  window.OFFLINE_DEMO = true;

  const MODELS = [
    { id: "openrouter/qwen/qwen3-235b-a22b-2507", label: "Qwen3 235B A22B · OpenRouter", open: true },
    { id: "openrouter/deepseek/deepseek-v3.2", label: "DeepSeek V3.2 · OpenRouter", open: true },
    { id: "openrouter/z-ai/glm-5", label: "GLM 5 · OpenRouter", open: true },
    { id: "openrouter/mistralai/mistral-small-2603", label: "Mistral Small 4 · OpenRouter", open: true },
    { id: "openrouter/qwen/qwen3-32b", label: "Qwen3 32B · OpenRouter", open: true },
    { id: "openrouter/qwen/qwen3-coder", label: "Qwen3 Coder · OpenRouter", open: true },
    { id: "openrouter/deepseek/deepseek-chat-v3.1", label: "DeepSeek V3.1 · OpenRouter", open: true },
    { id: "openrouter/meta-llama/llama-3.3-70b-instruct", label: "Llama 3.3 70B Instruct · OpenRouter", open: true },
    { id: "openrouter/moonshotai/kimi-k2.5", label: "Kimi K2.5 · OpenRouter", open: true },
    { id: "openrouter/openai/gpt-oss-120b", label: "GPT-OSS 120B · OpenRouter", open: true },
    { id: "host/gpt-5.6-terra", label: "GPT-5.6 Terra · подписка Codex", open: false },
    { id: "host/gpt-5.6-sol", label: "GPT-5.6 Sol · подписка Codex", open: false },
    { id: "host/gpt-5.6-luna", label: "GPT-5.6 Luna · подписка Codex", open: false },
  ];
  const TOOLS = [
    ["sandbox-light", "расчёты на чистом Python"],
    ["sequential-thinking", "пошаговый разбор сложных задач"],
    ["document", "чтение PDF, DOCX, XLSX, CSV"],
    ["download", "скачивание файлов по ссылке"],
    ["media", "разбор изображений, аудио и видео"],
    ["websearch-searxng", "веб-поиск через локальный SearXNG"],
    ["web-scraping", "открыть страницу и вытащить содержимое"],
  ];

  // Данные берём у выбранного сценария: тогда прогон, сравнение и вердикт относятся к нему,
  // а не к первому попавшемуся. Если ничего не выбрано — первый сценарий с записью.
  function donor() {
    const list = window.PRESETS || [];
    const active = document.querySelector(".preset.active");
    const id = active && active.dataset ? active.dataset.id : null;
    const chosen = id ? list.find((p) => p.id === id) : null;
    if (chosen && (chosen.trace || []).length) return chosen;
    return list.find((p) => (p.trace || []).length) || chosen || null;
  }

  const json = (data) => new Response(JSON.stringify(data),
    { status: 200, headers: { "Content-Type": "application/json" } });

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // Поток SSE: события уходят с задержками, чтобы прогресс и журнал оживали как в живом режиме
  function stream(events) {
    const encoder = new TextEncoder();
    return new Response(new ReadableStream({
      async start(controller) {
        for (const [delay, payload] of events) {
          await sleep(delay);
          controller.enqueue(encoder.encode("data: " + JSON.stringify(payload) + "\n\n"));
        }
        controller.close();
      },
    }), { status: 200, headers: { "Content-Type": "text/event-stream" } });
  }

  function mockConfig(kind) {
    const d = donor();
    if (d && d.config) return JSON.parse(JSON.stringify(d.config));
    return { agents: [], pipeline: { type: "sequential", children: [] } };
  }

  function generateEvents(kind) {
    return [
      [400, { type: "agent_start", agent: "pool_generator" }],
      [1400, { type: "agent_done", agent: "pool_generator", ms: 7200 }],
      [300, { type: "agent_start", agent: "pipeline_generator" }],
      [1600, { type: "agent_done", agent: "pipeline_generator", ms: 19400 }],
      [200, { type: "done", ok: true, kind, config: mockConfig(kind),
              gen: { tokens: 8168, seconds: 26.6 }, model: "openai/gpt-4.1-mini" }],
    ];
  }

  function runEvents() {
    const d = donor();
    const trace = (d && d.trace) || [];
    const events = [];
    let tokens = 0;
    trace.forEach((step) => {
      tokens += step.tokens || 0;
      events.push([700, { type: "agent_start", agent: step.agent }]);
      events.push([500, { type: "text", agent: step.agent, text: step.text, tokens: step.tokens || 0 }]);
      if (step.tool) events.push([300, { type: "tool", agent: step.agent, tool: step.tool }]);
      events.push([200, { type: "agent_done", agent: step.agent, ms: step.ms || 1000 }]);
    });
    const state = {};
    trace.forEach((s, i) => { state["шаг_" + (i + 1) + "_" + s.agent] = s.text; });
    if (d && d.answer) state["итоговый_ответ"] = d.answer;
    events.push([400, { type: "done", elapsed: 88.4, tokens: tokens || 124970, state }]);
    return events;
  }

  const original = window.fetch.bind(window);

  window.fetch = function (input, init) {
    const url = String(typeof input === "string" ? input : (input && input.url) || "");
    const body = (() => { try { return JSON.parse((init && init.body) || "{}"); } catch { return {}; } })();

    if (url.includes("api/status")) {
      return Promise.resolve(json({
        live: true, model: MODELS[0].id, models: MODELS,
        judge_model: "google/gemini-2.5-pro",
        safe_tools: TOOLS.map((t) => t[0]),
        tools: TOOLS.map(([id, note]) => ({ id, note })),
        web_search: true, scraping: true, has_key: true, mock: true,
      }));
    }

    if (url.includes("api/prepare")) {
      const text = (body.text || "").trim();
      return sleep(1200).then(() => json({
        ok: true,
        task: "Постановка для мета-агента выделена автоматически из текста задачи. "
            + "Предметные агенты работают параллельно, затем сборщик составляет единый ответ, "
            + "а проверяющий контролирует полноту и согласованность чисел.",
        query: text,
        model: "openai/gpt-4.1-mini",
      }));
    }

    if (url.includes("api/generate_stream")) return Promise.resolve(stream(generateEvents(body.kind || "maw")));
    if (url.includes("api/generate")) {
      return sleep(1500).then(() => json({ ok: true, kind: body.kind || "maw", config: mockConfig(body.kind),
        gen: { tokens: 8168, seconds: 26.6 }, model: "openai/gpt-4.1-mini" }));
    }
    if (url.includes("api/run")) return Promise.resolve(stream(runEvents()));

    if (url.includes("api/synthetic_examples")) {
      const source = String(body.query || "").trim();
      const variants = [
        `Переформулируй задачу и выполни её: ${source}`,
        `Нужно решить следующую задачу, сохранив все её условия: ${source}`,
        `Выполни запрос ниже без изменения исходных ограничений: ${source}`,
      ].slice(0, Math.max(1, Math.min(10, Number(body.count) || 1)));
      return sleep(800).then(() => json({ ok: true, examples: variants,
        model: body.model || MODELS[0].id, tokens: 180 }));
    }

    if (url.includes("api/effort")) {
      // Итог трудоёмкости — сумма часов по подзадачам, как и в живом режиме
      const subtasks = [
        { name: "Собрать исходные данные по объекту", hours: 3,
          note: "выгрузки, справочники, приведение к одному виду" },
        { name: "Найти и проверить нормативные значения в официальных источниках", hours: 4,
          note: "поиск документа, сверка редакции, ссылки" },
        { name: "Свести таблицу и посчитать показатели", hours: 2.5,
          note: "расчёт по каждой строке, производные метрики" },
        { name: "Сверить расчёты с нормативами", hours: 1.5,
          note: "сопоставление значений, отметка расхождений" },
        { name: "Собрать единый отчёт", hours: 3, note: "перенос чисел, ссылки, оформление" },
        { name: "Перепроверить арифметику и выводы", hours: 2, note: "контрольный пересчёт" },
      ];
      const total = subtasks.reduce((a, t) => a + t.hours, 0);
      return sleep(1400).then(() => json({ ok: true, subtasks,
        total_hours: Math.round(total * 100) / 100,
        total_days: Math.round((total / 8) * 10) / 10,   // восьмичасовой день, как на сервере
        model: "openai/gpt-4.1-mini" }));
    }

    if (url.includes("api/baseline")) {
      const d = donor();
      return sleep(1800).then(() => json(Object.assign(
        { ok: true, answer: "(демонстрационная копия: ответ одной модели показан из записи прогона)",
          model: "openai/gpt-4.1-mini", tokens: 4975, seconds: "47,7" },
        (d && d.baseline) || {}, { ok: true })));
    }

    if (url.includes("api/judge")) {
      const d = donor();
      return sleep(2000).then(() => json(Object.assign(
        { ok: true, winner: "system", verdict: "(демонстрационная копия: вердикт показан из записи прогона)",
          model: "google/gemini-2.5-pro" },
        (d && d.judge) || {}, { ok: true })));
    }

    return original(input, init);
  };

  // Заметная пометка, чтобы копию не приняли за рабочий стенд
  document.addEventListener("DOMContentLoaded", () => {
    const mark = document.createElement("div");
    mark.className = "mock-mark";
    mark.textContent = "демонстрационная копия · без реальных запусков";
    mark.title = "Кнопки и формы работают, но ничего не считается: данные показаны из записи прогонов";
    const bar = document.querySelector(".topbar-right");
    if (bar) bar.insertBefore(mark, bar.firstChild);
  });
})();
