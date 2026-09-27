const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
const renderer = source.match(/function renderModelKeyAccess\(\) \{[\s\S]*?\n\}/)?.[0];

function modelKeyUI(backend) {
  const button = { hidden: false, classList: { toggle(_, hidden) { button.hidden = hidden; } } };
  const info = { textContent: "" };
  const elements = { "models-key": button, "models-key-info": info };
  vm.runInNewContext(`${renderer}\nrenderModelKeyAccess();`, {
    S: { backend },
    $: id => elements[id],
  });
  return { hidden: button.hidden, message: info.textContent };
}

test("server-side OpenRouter key does not prompt for another one", () => {
  const ui = modelKeyUI({ openrouter_ready: true, public: false, user_key: false });
  assert.equal(ui.hidden, true);
  assert.match(ui.message, /вводить ключ.*не нужно/);
});

test("public mode still offers users their own key", () => {
  const ui = modelKeyUI({ openrouter_ready: false, public: true, user_key: false });
  assert.equal(ui.hidden, false);
  assert.match(ui.message, /нужен ключ/);
});

test("offline GUI does not offer a key form that cannot reach the API", () => {
  const ui = modelKeyUI(null);
  assert.equal(ui.hidden, true);
  assert.match(ui.message, /Нет связи с сервером/);
});

test("new server default is not overridden by old saved model choices", () => {
  assert.equal(source.includes('loadStored("fedotmas-models",'), false);
  assert.match(source, /loadStored\("fedotmas-models-v2"/);
});
