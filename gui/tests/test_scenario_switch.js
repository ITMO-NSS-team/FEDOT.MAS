const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const app = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");

function scenario(id, answer, trace = []) {
  return {
    id, title: id, kind: "maw", query: `запрос ${id}`, trace, auto: "—", manual: "—",
    config: { agents: [{ name: `агент_${id}` }], pipeline: {} },
    answer,
  };
}

function gui() {
  const elements = new Map();
  const views = {};
  const S = { preset: null, abort: null, backend: {}, custom: [], hidden: [],
    models: { run: "openrouter/qwen/qwen3" }, factor: 1, evaluationEpoch: 0 };
  const $ = id => {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, {
      innerHTML: "", textContent: "", value: "", style: {},
      classList: {
        toggle(name, force) { if (force ?? !classes.has(name)) classes.add(name); else classes.delete(name); },
        add: name => classes.add(name), remove: name => classes.delete(name),
        contains: name => classes.has(name),
      },
      closest: () => ({ title: "" }), querySelectorAll: () => [],
      });
    }
    return elements.get(id);
  };
  let stopped = 0;
  const context = {
    S, $, document: { querySelectorAll: () => [], querySelector: () => null },
    performance: { now: () => 100 }, TextDecoder, AbortController,
    setInterval: () => 1, clearInterval: () => {},
    pause: () => {},
    stopLive: () => { stopped++; S.abort?.abort(); S.abort = null; },
    indexAgents: p => new Map(p.config.agents.map(a => [a.name, a])),
    parseTarget: () => 0, pipelineDepth: () => 1, allTools: () => new Map(),
    renderSources: () => {}, renderGraph: () => {}, renderMode: () => {},
    renderAnswer: () => { views.answer = S.answer?.text || null; },
    renderInspector: () => { views.agents = [...S.agents.keys()]; },
    renderEffort: p => { views.effort = p?.id || null; },
    renderSyntheticExamples: () => {},
    resetRun: () => { $("feed").innerHTML = "Журнал появится после запуска системы"; },
    showRecordedRun: p => { if (p.trace.length) $("feed").innerHTML = p.trace.map(e => e.text).join("\n"); },
    loadRubberQuality: () => {}, normalizedSynthetic: x => x,
    hoursText: String, esc: String, displayMeta: String,
    scenarioList: () => S.custom,
    showTab: () => {}, setPlayIcon: () => {}, liveMessage: () => {},
  };
  vm.createContext(context);
  function load(start, end) {
    vm.runInContext(app.slice(app.indexOf(start), app.indexOf(end, app.indexOf(start))), context);
  }
  load("function loadPreset(p) {", "// Показывает сохранённый журнал");
  load("async function liveRun() {", "/* ─────────────────────────── Тема");
  load("function judgeProgress() {", "async function runComparisonJudge()");
  load("function showEmptyState() {", "function renderPresetList()");
  return { context, S, $, views, stopped: () => stopped };
}

test("switching from a recorded run to a new scenario updates every tab", () => {
  const { context, S, $, views, stopped } = gui();
  const old = scenario("старый", "старый ответ", [{ text: "старый журнал" }]);
  const fresh = scenario("новый", null);
  S.custom = [old, fresh];

  context.loadPreset(old);
  assert.equal(views.answer, "старый ответ");
  assert.equal($("feed").innerHTML, "старый журнал");

  const active = new AbortController();
  S.abort = active;
  context.loadPreset(fresh);
  assert.equal(active.signal.aborted, true);
  assert.equal(S.abort, null);
  assert.equal(stopped(), 2);
  assert.equal(S.preset, fresh);
  assert.equal($("query").value, "запрос новый");
  assert.equal(views.answer, null);
  assert.match($("feed").innerHTML, /Журнал появится/);
  assert.deepEqual(views.agents, ["агент_новый"]);
  assert.equal(views.effort, "новый");
});

test("late events from the previous run cannot overwrite the selected scenario", async () => {
  const { context, S, $, views } = gui();
  const old = scenario("старый", null);
  const fresh = scenario("новый", null);
  S.custom = [old, fresh];
  context.loadPreset(old);

  let releaseRead;
  context.fetch = async () => ({ ok: true, body: { getReader: () => ({
    read: () => new Promise(resolve => { releaseRead = resolve; }),
  }) } });
  const running = context.liveRun();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(typeof releaseRead, "function");

  context.loadPreset(fresh);
  releaseRead({ done: false, value: Buffer.from(
    'data: {"type":"done","tokens":100,"elapsed":1,"state":{"answer":"чужой ответ"}}\n\n',
  ) });
  await running;

  assert.equal(S.preset, fresh);
  assert.equal(S.answer, null);
  assert.equal(fresh.answer, null);
  assert.equal(views.answer, null);
  assert.match($("feed").innerHTML, /Журнал появится/);
});

test("removing the last scenario clears inspector tabs and the task field", () => {
  const { context, S, $, views } = gui();
  context.loadPreset(scenario("старый", "старый ответ", [{ text: "старый журнал" }]));
  $("agent-cards").innerHTML = "старые агенты";
  $("tool-cards").innerHTML = "старые инструменты";
  $("json").innerHTML = "старый JSON";

  context.showEmptyState();

  assert.equal(S.preset, null);
  assert.equal(S.agents.size, 0);
  assert.equal($("query").value, "");
  assert.equal($("agent-cards").innerHTML, "");
  assert.equal($("tool-cards").innerHTML, "");
  assert.equal($("json").innerHTML, "");
  assert.equal(views.effort, null);
  assert.match($("answer").innerHTML, /Ответ системы появится/);
  assert.match($("feed").innerHTML, /Журнал появится/);
});

test("an old judge result cannot hide progress for the newly selected scenario", () => {
  const { context, S, $ } = gui();
  const previous = context.judgeProgress();
  S.evaluationEpoch++;
  const current = context.judgeProgress();

  previous.stop();
  assert.equal($("judge-progress").classList.contains("hidden"), false);
  current.stop();
  assert.equal($("judge-progress").classList.contains("hidden"), true);
});
