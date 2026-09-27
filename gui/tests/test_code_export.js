const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
const start = source.indexOf("async function exportCode() {");
const end = source.indexOf("function renderModelKeyAccess()", start);

test("source button downloads the selected MAS without sending the API key or log", async () => {
  assert.ok(start >= 0 && end > start);
  const requests = [];
  const button = { disabled: false };
  const link = { click() { this.clicked = true; }, remove() {} };
  const context = {
    S: { preset: {title: "Пример", kind: "maw", config: {agents: []}, tools: ["document"],
      customMcp: [{name: "local", url: "https://example.test/mcp"}], trace: [{private: "log"}]},
      models: {run: "openrouter/qwen/qwen3-32b"} },
    $: () => button, Blob,
    document: {createElement: () => link, body: {appendChild() {}}},
    URL: {createObjectURL: () => "blob:archive", revokeObjectURL() {}},
    setTimeout() {}, alert(message) { throw new Error(message); },
    fetch: async (url, init) => {
      requests.push({url, init});
      return {ok: true, blob: async () => new Blob(["zip"])};
    },
  };
  vm.createContext(context);
  vm.runInContext(source.slice(start, end), context);
  await context.exportCode();
  assert.equal(requests[0].url, "api/export-code");
  const body = JSON.parse(requests[0].init.body);
  assert.equal(body.kind, "maw");
  assert.equal(body.model, "openrouter/qwen/qwen3-32b");
  assert.deepEqual(body.tools, ["document"]);
  assert.equal(body.trace, undefined);
  assert.equal(body.key, undefined);
  assert.equal(link.clicked, true);
  assert.match(link.download, /код_МАС\.zip$/);
  assert.equal(button.disabled, false);
});
