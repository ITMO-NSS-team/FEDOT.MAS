const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
const css = fs.readFileSync(path.join(__dirname, "../static/styles.css"), "utf8");

function feedContext() {
  const messages = [];
  const feed = {appendChild: node => messages.push(node), scrollHeight: 100, innerHTML: ""};
  const elements = {feed, graph: {querySelectorAll: () => []},
    "m-tokens": {}, "m-time": {}, "p-fill": {style: {}}};
  const context = {
    S: {agents: new Map(), factor: 1},
    $: id => elements[id],
    recordedRunStats: () => ({tokens: 0, elapsed: 0}),
    nfmt: String, fmtTime: String,
    document: {createElement: () => ({innerHTML: "", className: ""})},
    roleOf: () => ({kind: "worker", icon: "user"}),
    ROLE_CLASS: {worker: "role-worker"},
    ICONS: {user: ["M1 1h2"]},
    esc: value => String(value).replace(/[&<>\"]/g,
      char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[char]),
  };
  vm.createContext(context);
  function load(start, end) {
    const from = source.indexOf(start);
    assert.ok(from >= 0);
    vm.runInContext(source.slice(from, source.indexOf(end, from)), context);
  }
  load("function mdToHtml(src) {", "/* ─────────────────────────── Ответ системы");
  load("function pushMessage(e) {", "/* ─────────────────────────── Плеер");
  load("function showRecordedRun(preset) {", "/* ─────────────────────────── Живой режим");
  load("function liveMessage(kind, agent, text, tool) {", "function liveActivate(name) {");
  return {context, messages};
}

test("live and recorded feed messages render model Markdown", () => {
  const {context, messages} = feedContext();
  const markdown = "## Расчёт\n**MAPE**: 5 %\n- опыт 1\n- опыт 2";
  context.liveMessage("шаг агента", "аналитик", markdown);
  context.showRecordedRun({trace: [{agent: "аналитик", phase: "шаг агента", text: markdown}]});
  assert.equal(messages.length, 2);
  assert.equal(messages[0].innerHTML, messages[1].innerHTML);
  const html = messages[0].innerHTML;
  assert.match(html, /class="msg-text md"/);
  assert.match(html, /<h5>Расчёт<\/h5>/);
  assert.match(html, /<b>MAPE<\/b>: 5 %/);
  assert.match(html, /<ul><li>опыт 1<\/li><li>опыт 2<\/li><\/ul>/);
  assert.match(css, /\.msg-text\.md ul/);
  assert.doesNotMatch(html, /\*\*MAPE\*\*/);
});

test("Markdown in the feed stays escaped; code keeps literal asterisks", () => {
  const {context, messages} = feedContext();
  context.pushMessage({agent: "<img src=x>", phase: "результат",
    text: '<img src=x onerror=alert(1)> **готово**\n`**literal**`\n```python\nprint("**raw**")\n```'});
  const html = messages[0].innerHTML;
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.match(html, /<b>готово<\/b>/);
  assert.match(html, /<code>\*\*literal\*\*<\/code>/);
  assert.match(html, /<pre><code>print\(&quot;\*\*raw\*\*&quot;\)<\/code><\/pre>/);
});
