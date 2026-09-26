const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const staticDir = path.join(__dirname, "../static");
const context = {window: {}};
vm.createContext(context);
for (const file of ["rubber_preset.js", "technology_card_preset.js"]) {
  vm.runInContext(fs.readFileSync(path.join(staticDir, file), "utf8"), context);
}
for (const p of context.window.STARTUP_PRESETS) {
  const b = p.breakdown;
  assert.equal(b.subtasks.length, 6);
  assert.equal(b.total_hours, b.subtasks.reduce((sum, t) => sum + t.hours, 0));
  assert.equal(b.total_days, b.total_hours / 8);
  assert.equal(b.total_hours, p.kind === "mas" ? 24 : 28);
  assert.match(p.manualNote, /Время не измерялось/);
  assert.equal(p.effortRevision, 1);
}
const app = fs.readFileSync(path.join(staticDir, "app.js"), "utf8");
const box = {innerHTML: ""};
context.$ = () => box;
context.esc = String;
context.num = String;
context.hoursText = h => `${h} чел.-ч`;
context.daysText = d => `${d} дня`;
vm.runInContext(app.slice(app.indexOf("function renderEffort(p) {"),
                         app.indexOf("/* ─────────────────────────── Источники данных")), context);
for (const p of context.window.STARTUP_PRESETS) {
  context.renderEffort(p);
  assert.equal((box.innerHTML.match(/class="effort-row"/g) || []).length, 6);
  assert.match(box.innerHTML, /Время не измерялось/);
  assert.ok(box.innerHTML.includes(`${p.breakdown.total_hours} чел.-ч`));
}
const migration = app.slice(app.indexOf("  let queryUpdated = false;"),
                           app.indexOf("  // В автономной копии список"));
context.S = {custom: context.window.STARTUP_PRESETS.map(p => ({
  id: p.id, query: "user input", queryRevision: p.queryRevision,
  manual: "old", config: {untouched: true},
}))};
let stores = 0;
context.storeScenarios = () => stores++;
vm.runInContext(migration, context);
assert.equal(stores, 1);
for (const saved of context.S.custom) {
  assert.equal(saved.query, "user input");
  assert.equal(saved.config.untouched, true);
  assert.equal(saved.effortRevision, 1);
  assert.equal(saved.breakdown.subtasks.length, 6);
}
// Re-running initialization must not overwrite later user changes.
context.S.custom[0].manual = "user estimate";
vm.runInContext(`{ ${migration} }`, context);
assert.equal(stores, 1);
assert.equal(context.S.custom[0].manual, "user estimate");
console.log("Effort totals, assumptions and saved-scenario migration: OK");
