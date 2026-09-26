const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const app = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const context = {esc: String};
vm.createContext(context);
vm.runInContext(app.slice(app.indexOf('function rubberQualityHtml('), app.indexOf('async function loadRubberQuality(')), context);
assert.equal(context.rubberQualityHtml(null), '');
assert.equal(context.rubberQualityHtml({tools: ['technology-card-audit']}), '');
const preset = {tools: ['rubber-recipe-predictor'], rubberQuality: {samples: 20,
  mape_pct: {thermal_conductivity_w_mk: 4.49396, oil_swelling_pct_1006h: 9.68771,
    water_swelling_pct_1006h: 3.45544, specific_gravity: 0.59405}}};
let html = context.rubberQualityHtml(preset);
for (const value of ['4,49 %', '9,69 %', '3,46 %', '0,59 %']) assert.ok(html.includes(value));
preset.rubberValidation = {mape_pct: null, reason: 'no_exact_reference'};
assert.match(context.rubberQualityHtml(preset), /Нет контрольных измерений/);
preset.rubberValidation = {mape_pct: 0, ape_pct: {specific_gravity: 0}};
html = context.rubberQualityHtml(preset);
assert.match(html, /0,00 % — среднее/);
assert.match(html, /не независимая проверка/);
assert.match(context.rubberQualityHtml({tools: preset.tools}), /нет данных/);
assert.equal(context.rubberQualityHtml(JSON.parse(JSON.stringify(preset))), html);
console.log('Rubber MAPE rendering, missing references, zero and persistence: OK');
