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
assert.doesNotMatch(context.rubberQualityHtml(preset), /Ниже приведён/);
preset.rubberQuality.baseline_example = {
  recipe: {nr_phr: 50, sbr_phr: 50, carbon_black_n220_phr: 60},
  training_rows: 19, reference_used_in_training: false,
  predicted: {thermal_conductivity_w_mk: 0.486775, oil_swelling_pct_1006h: 21.43316,
    water_swelling_pct_1006h: 29.026877, specific_gravity: 1.214267},
  measured: {thermal_conductivity_w_mk: 0.460, oil_swelling_pct_1006h: 21.5,
    water_swelling_pct_1006h: 31, specific_gravity: 1.210},
  ape_pct: {thermal_conductivity_w_mk: 5.82, oil_swelling_pct_1006h: 0.31,
    water_swelling_pct_1006h: 6.36, specific_gravity: 0.35},
  mape_pct: 3.2122733565,
};
html = context.rubberQualityHtml(preset);
for (const value of ['NR/SBR 50/50', 'N220 60 phr', '19 рецептурам', '0,487',
  '0,460', '21,433', '21,500', '3,21 %', 'отложенной точке']) assert.ok(html.includes(value));
assert.match(html, /Нет контрольных измерений[\s\S]*Ниже приведён/);
preset.rubberValidation = {mape_pct: 0, ape_pct: {specific_gravity: 0}};
html = context.rubberQualityHtml(preset);
assert.match(html, /0,00 % — среднее/);
assert.match(html, /не независимая проверка/);
assert.match(context.rubberQualityHtml({tools: preset.tools}), /нет данных/);
assert.equal(context.rubberQualityHtml(JSON.parse(JSON.stringify(preset))), html);
console.log('Rubber MAPE rendering, missing references, zero and persistence: OK');
