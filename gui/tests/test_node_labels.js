const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const app = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
const agent = name => ({ type: "agent", agent_name: name });
const longName = "координатор_проверки_результатов_технологической_карты";

function svg(tag, attrs = {}, ...children) {
  return {
    tag, attrs, children,
    appendChild(child) { this.children.push(child); },
  };
}

const context = {
  S: { agents: new Map() }, el: svg,
  ICONS: { user: ["M0 0"] }, ROLE_CLASS: { worker: "worker" },
  roleOf: () => ({ icon: "user", kind: "worker" }),
  trunc: (value, length) => value.length > length ? value.slice(0, length - 1) + "…" : value,
};
vm.createContext(context);
vm.runInContext(app.slice(app.indexOf("const NW ="), app.indexOf("/* ─────────────────────────── Отрисовка графа")), context);
vm.runInContext(app.slice(app.indexOf("function drawNode("), app.indexOf("function renderGraph()")), context);

test("long agent names wrap without losing characters or shrinking to illegibility", () => {
  const lines = context.wrapNodeName(longName);
  assert.ok(lines.length >= 3);
  assert.ok(lines.every(line => Array.from(line).length <= 20));
  assert.equal(lines.join(""), longName);

  const unbroken = "оченьдлинноеназваниеагентабезразделителей";
  assert.equal(context.wrapNodeName(unbroken).join(""), unbroken);

  const height = context.nodeHeight(longName);
  const rendered = context.drawNode({ name: longName, x: 0, y: 0, h: height }, 0);
  const group = rendered.children[0];
  const name = group.children.find(node => node.attrs.class === "node-name");
  const subtitle = group.children.find(node => node.attrs.class === "node-sub");
  const box = group.children.find(node => node.attrs.class === "node-box");
  assert.deepEqual(name.children.filter(node => node.tag === "tspan").map(node => node.children[0]), Array.from(lines));
  assert.equal(box.attrs.height, height);
  assert.ok(subtitle.attrs.y <= height - 12);
});

test("MAW and MAS layouts leave room for wrapped nodes and their edges", () => {
  const parallel = context.layoutMAW({ type: "parallel", children: [
    agent(longName), agent("короткий"),
  ] }, 900);
  const [tall, short] = parallel.nodes;
  assert.ok(tall.h > short.h);
  assert.ok(short.y >= tall.y + tall.h + 24);
  assert.ok(parallel.edges.some(edge => edge.toName === longName && edge.to.y === tall.y + tall.h / 2));

  const mas = context.layoutMAS({
    coordinator: { name: longName },
    workers: ["первый", "второй", "третий", "четвёртый", "пятый"].map((name, index) => ({
      name: index === 0 ? longName + "_исполнитель" : name,
    })),
  });
  const coordinator = mas.nodes[0];
  const firstRow = mas.nodes.slice(1, 4);
  const secondRow = mas.nodes.slice(4);
  assert.equal(coordinator.h, context.nodeHeight(longName));
  assert.ok(Math.min(...secondRow.map(node => node.y)) > Math.max(...firstRow.map(node => node.y + node.h)));
  assert.ok(mas.edges.every(edge => edge.from.y === coordinator.h));
});
