const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const app = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
const elements = new Map();
const storage = new Map();
const context = {
  S: {factor: 1, custom: [], hidden: []},
  $: id => {
    if (!elements.has(id)) elements.set(id, {
      style: {}, value: "task", querySelectorAll: () => [],
    });
    return elements.get(id);
  },
  localStorage: {getItem: k => storage.get(k), setItem: (k, v) => storage.set(k, v)},
  LS_CUSTOM: "custom", LS_HIDDEN: "hidden", LS_RUN: "server",
  nfmt: String, fmtTime: String, esc: String,
  document: {querySelector: () => null},
  performance: {now: () => 100}, TextDecoder, AbortController,
  setInterval: () => 1,
};
for (const name of ["pushMessage", "stopLive", "renderAnswer", "showTab",
  "setPlayIcon", "liveMessage", "renderPresetList"]) context[name] = () => {};
vm.createContext(context);
function load(start, end) {
  vm.runInContext(app.slice(app.indexOf(start), app.indexOf(end, app.indexOf(start))), context);
}
load("function recordedRunStats(", "/* ─");
load("function storeScenarios()", "function scenarioList()");
load("function persistPreset()", "function stopLive()");
load("async function liveRun()", "/* ─");
load("function resetOnServerRestart(", "function presetFromConfig(");

async function run(events) {
  context.S.preset = {id: "test", custom: true, kind: "maw", config: {agents: []},
    runStats: {tokens: 99999, elapsed: 10}};
  context.S.custom = [context.S.preset];
  context.S.models = {run: "test"};
  let read = false;
  context.fetch = async () => ({ok: true, body: {getReader: () => ({read: async () => {
    if (read) return {done: true};
    read = true;
    return {value: Buffer.from(events.map(e => `data: ${JSON.stringify(e)}\n\n`).join("")), done: false};
  }})}});
  await context.liveRun();
  // Simulate closing the page and loading only JSON from storage.
  const saved = JSON.parse(storage.get("custom"))[0];
  context.S.tokens = 0;
  context.showRecordedRun(saved);
  return saved;
}
(async () => {
  for (const total of [12345, 0]) {
    const saved = await run([
      {type: "text", agent: "worker", text: "answer", tokens: 5},
      {type: "tokens", tokens: 300},
      {type: "done", tokens: total, elapsed: 12.34, state: {answer: "result"}},
    ]);
    assert.equal(saved.runStats.tokens, total);
    assert.equal(context.S.tokens, total);
    assert.equal(context.S.seconds, 12.34);
    assert.equal(elements.get("m-tokens").textContent, String(total));
  }
  const empty = await run([{type: "done", tokens: 42, elapsed: 2, state: {a: "result"}}]);
  assert.equal(empty.trace.length, 0);
  assert.equal(context.S.tokens, 42);
  const fallback = await run([{type: "tokens", tokens: 77}, {type: "done", elapsed: 1}]);
  assert.equal(fallback.runStats.tokens, 77);
  assert.equal(context.recordedRunStats({answerMeta: "MASConfig · 12\u00a0345 токенов · 3 с",
    trace: [{tokens: 1}]}).tokens, 12345);
  assert.equal(context.recordedRunStats({trace: [{tokens: 10}, {tokens: 20}]}).tokens, 30);
  const before = storage.get("custom");
  const scenarios = context.S.custom;
  context.resetOnServerRestart("new-server");
  assert.equal(storage.get("custom"), before);
  assert.equal(context.S.custom, scenarios);
  console.log("Run statistics: live totals, reload, legacy reports and server restart OK");
})().catch(e => { console.error(e); process.exitCode = 1; });
