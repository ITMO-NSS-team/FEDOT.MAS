const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const zlib = require('node:zlib');

const staticDir = path.join(__dirname, '../static');
const source = file => fs.readFileSync(path.join(staticDir, file), 'utf8');
const context = {window: {}};
vm.createContext(context);
vm.runInContext(source('rubber_reference_run.js'), context);
vm.runInContext(source('rubber_reference_review.js'), context);
vm.runInContext(source('rubber_preset.js'), context);

const [reference, original] = context.window.STARTUP_PRESETS;
assert.equal(reference.id, 'rubber_heldout_reference_run_20260927');
assert.equal(reference.real, true);
assert.equal(reference.installOnExisting, true);
assert.equal(reference.title, 'Прогноз свойств рецепта резины · MAW');
const oldHeading = 'Прогнозные значения и сравнение с опубликованными данными';
const newHeading = 'Прогнозные значения и сравнение с фактом';
assert.ok(reference.answer.includes(newHeading));
assert.ok(!reference.answer.includes(oldHeading));
assert.equal(reference.trace.filter(event => JSON.stringify(event).includes(newHeading)).length, 2);
assert.match(reference.query, /NR SMR-20 — 50 phr; SBR-1502 — 50 phr; технический углерод N220 — 60 phr/);
assert.match(original.query, /NR SMR-20 — 55 phr; SBR-1502 — 45 phr; технический углерод N220 — 55 phr/);
assert.ok(Math.abs(reference.rubberValidation.mape_pct - 3.2122669311542347) < 1e-8);
assert.equal(reference.rubberValidation.reference_used_in_training, false);
assert.equal(reference.rubberValidation.training_rows, 19);
assert.match(reference.answer, /3\.21%/);
assert.match(reference.answer, /0\.4868/);
assert.equal(reference.runStats.tokens, 47442);
assert.equal(reference.review.ok, true);
assert.equal(reference.review.model, 'openrouter/deepseek/deepseek-v3.2');
assert.match(reference.review.verdict, /Общий вердикт[\s\S]*ВЫПОЛНЕНО/i);
assert.ok(reference.review.tokens > 0);
assert.ok(reference.trace.some(event => event.tool === 'predict_rubber_properties'));
assert.ok(reference.trace.some(event => event.io?.outputKey === 'property_prediction'));
assert.ok(reference.config.agents.every(agent => agent.model === reference.model && agent.max_output_tokens === 12000));
const archived = JSON.parse(zlib.gunzipSync(Buffer.from(fs.readFileSync(
  path.join(__dirname, '../recordings/rubber_reference_20260927.json.gz.b64'), 'utf8').trim(),
  'base64')).toString('utf8'));
assert.equal(reference.answer, archived.done.state.property_prediction.replaceAll(oldHeading, newHeading));
assert.equal(reference.review.verdict, archived.review.verdict);
assert.equal(reference.rubberValidation.mape_pct, archived.rubber_validation.mape_pct);
assert.ok(archived.events.some(event => event.type === 'tool_result' && event.rubber_validation));

const app = source('app.js');
const qualityContext = {esc: String};
vm.createContext(qualityContext);
vm.runInContext(app.slice(app.indexOf('function rubberQualityHtml('),
                          app.indexOf('async function loadRubberQuality(')), qualityContext);
const qualityHtml = qualityContext.rubberQualityHtml(reference);
assert.match(qualityHtml, /MAPE текущего расчёта[\s\S]*3,21 %/);
assert.doesNotMatch(qualityHtml, /Нет контрольных измерений/);
const migrate = app.slice(app.indexOf('  const retiredRubberId ='),
                          app.indexOf('  // В автономной копии список'));
context.S = {custom: [
  {...original, query: 'пользовательская правка'},
  {...reference, title: 'Шины · контрольная рецептура с MAPE',
    answer: archived.done.state.property_prediction,
    trace: context.window.RUBBER_REFERENCE_RUN.trace.map(event => ({...event}))},
], hidden: []};
let saves = 0;
context.storeScenarios = () => saves++;
vm.runInContext(migrate, context);
assert.equal(saves, 1);
assert.equal(context.S.custom[0].id, original.id);
assert.equal(context.S.custom[0].query, 'пользовательская правка');
assert.equal(context.S.custom[1].title, reference.title);
assert.equal(context.S.custom[1].answer, reference.answer);
assert.equal(context.S.custom[1].trace.filter(event => JSON.stringify(event).includes(newHeading)).length, 2);
assert.equal(context.S.hidden.includes(original.id), true);
vm.runInContext(app.slice(app.indexOf('function scenarioList()'),
                          app.indexOf('/** Стартовый вид')), context);
assert.equal(vm.runInContext('scenarioList().length', context), 1);
assert.equal(vm.runInContext('scenarioList()[0].id', context), reference.id);
vm.runInContext(`{ ${migrate} }`, context);
assert.equal(saves, 1);
assert.equal(context.S.custom.length, 2);
console.log('Recorded tire reference scenario, held-out MAPE and migration: OK');
