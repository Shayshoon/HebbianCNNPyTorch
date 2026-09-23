import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(here, "..");
const workspaceDir = here;
const SKILL_DIR = "C:\\Users\\bayan\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.909.12148\\skills\\presentations";
const RUNTIME_PYTHON = "C:\\Users\\bayan\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const TMP_DIR = path.join(here, ".build");
const FINAL_PPTX = process.env.PRESENTATION_OUTPUT
  ? path.resolve(process.env.PRESENTATION_OUTPUT)
  : path.join(here, "Hebbian_CNN_Compute_Comparison.pptx");

const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const FONT = resolvePresentationFont({ fontFamily: "Arial" });
const W = 1280;
const H = 720;

const C = {
  bg: "#F7F8F4",
  paper: "#FFFFFF",
  ink: "#17222F",
  muted: "#526272",
  faint: "#D9E0E5",
  grid: "#D9E0E5",
  a1: "#0072B2",
  a1Pale: "#DCEFF8",
  a2: "#E69F00",
  a2Pale: "#FBECCB",
  b: "#009E73",
  bPale: "#DDF3EC",
  bp: "#4D4D4D",
  bpPale: "#E6E6E6",
  warn: "#A33A3A",
  warnPale: "#F7E5E5",
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else quoted = !quoted;
    } else if (ch === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      row.push(field);
      field = "";
      if (row.some((v) => v.length)) rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  const headers = rows.shift();
  return rows.map((values) => Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""])));
}

async function csv(name) {
  return parseCsv(await fs.readFile(path.join(projectDir, "results", "data", name), "utf8"));
}

function extractScriptNotes(markdown) {
  const notes = new Map();
  const re = /## Slide (\d+) — ([^\n]+)\s+\*\*Speaker:\*\*\s*([^\n]+)[\s\S]*?\*\*Exact narration:\*\*\s*([\s\S]*?)(?=\n## Slide |\n## Timing summary)/g;
  for (const match of markdown.matchAll(re)) {
    notes.set(Number(match[1]), {
      title: match[2].trim(),
      speaker: match[3].trim(),
      narration: match[4].trim(),
    });
  }
  if (notes.size !== 8) throw new Error(`Expected 8 script sections, found ${notes.size}`);
  return notes;
}

const [sameWork, methodLevel, persistent, accuracy, scriptMarkdown] = await Promise.all([
  csv("compute_same_work.csv"),
  csv("compute_method_level.csv"),
  csv("memory_persistent_state.csv"),
  csv("accuracy_context.csv"),
  fs.readFile(path.join(here, "presentation_script.md"), "utf8"),
]);
const scriptNotes = extractScriptNotes(scriptMarkdown);

function row(rows, predicate) {
  const found = rows.find(predicate);
  if (!found) throw new Error("Required frozen row not found");
  return found;
}

const batch = ["A1", "A2", "B", "BP"].map((method) =>
  Number(row(sameWork, (r) => r.scope === "one_batch" && r.method === method).total_scalar_operations),
);
const method20 = ["A1", "A2", "B"].map((method) =>
  Number(row(methodLevel, (r) => r.method === method && r.epochs === "20").total_scalar_operations),
);
const bp20 = Number(row(methodLevel, (r) => r.method === "BP" && r.epochs === "20").total_scalar_operations);
const bp100 = Number(row(methodLevel, (r) => r.method === "BP" && r.epochs === "100").total_scalar_operations);
const persistentMb = ["A1", "A2", "B", "BP"].map((method) =>
  Number(row(persistent, (r) => r.method === method).persistent_megabytes),
);

const expectedBatch = [8667340084, 1380339683, 17267771387, 24430482934];
if (batch.some((v, i) => v !== expectedBatch[i])) throw new Error("Frozen one-batch totals changed");
if (Math.abs(batch[3] / batch[1] - 17.698891) > 0.000001) throw new Error("Frozen BP/A2 ratio changed");
if (Math.round(bp100) !== 1222377021050100) throw new Error("Frozen BP 100-epoch total changed");
if (persistentMb.join(",") !== "7.089,0.106,10.677,14.488") throw new Error("Frozen persistent-state totals changed");

const accuracyLookup = (source, method, epochs) => Number(row(
  accuracy,
  (r) => r.source === source && r.method === method && r.epochs === String(epochs),
).accuracy_percent);

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: opts.size ?? 22,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
    wrap: "square",
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

function addBox(slide, x, y, w, h, fill, line = fill, radius = 14) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function addLine(slide, x, y, w, h, color = C.faint, width = 2) {
  return slide.shapes.add({
    geometry: "line",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function addFrame(slide, title, number) {
  slide.background.fill = C.bg;
  addText(slide, title, 64, 35, 1020, 54, { size: 34, bold: true });
  addLine(slide, 64, 99, 1152, 1, C.faint, 1);
  addText(slide, String(number).padStart(2, "0"), 1170, 40, 46, 30, {
    size: 15, bold: true, color: C.muted, align: "right",
  });
}

function addNotes(slide, index, sources) {
  const n = scriptNotes.get(index);
  slide.speakerNotes.textFrame.setText(
    `Speaker: ${n.speaker}\n\n${n.narration}\n\nSources: ${sources}`,
  );
  slide.speakerNotes.setVisible(true);
}

function styleChart(chart) {
  applyPresentationChartFont(chart, { fontFamily: FONT });
  return chart;
}

function addMethodLabel(slide, x, y, label, color, detail, width = 260) {
  addBox(slide, x, y + 7, 12, 46, color, color, 6);
  addText(slide, label, x + 24, y, width - 24, 28, { size: 22, bold: true, color });
  addText(slide, detail, x + 24, y + 29, width - 24, 38, { size: 15, color: C.muted });
}

// Slide 1
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  addBox(slide, 0, 0, 20, H, C.a1, C.a1, 0);
  addText(slide, "Hebbian CNNs versus\nbackpropagation", 82, 96, 760, 180, {
    size: 48, bold: true, color: C.paper,
  });
  addText(slide, "A hardware-independent compute and memory comparison", 86, 301, 705, 48, {
    size: 24, color: "#C6D0D8",
  });
  addText(slide, "local  Δw", 892, 151, 260, 62, { size: 36, bold: true, color: C.a1, align: "center" });
  addLine(slide, 940, 238, 165, 1, "#40505F", 2);
  addText(slide, "global  ∇L", 892, 265, 260, 62, { size: 36, bold: true, color: "#C6D0D8", align: "center" });
  addText(slide, "Bayan & Shay", 86, 510, 560, 52, { size: 32, bold: true, color: C.paper });
  addText(slide, "CIFAR-10 · HebbianCNNPyTorch", 88, 575, 560, 36, { size: 19, color: "#C6D0D8" });
  addLine(slide, 86, 642, 1090, 1, "#40505F", 1);
  addNotes(slide, 1, "Frozen project report and repository artifacts.");
}

// Slide 2
{
  const slide = presentation.slides.add();
  addFrame(slide, "Research question and evidence", 2);

  addText(slide, "Backpropagation", 112, 132, 380, 36, { size: 25, bold: true, color: C.bp, align: "center" });
  const bpNodes = [
    addBox(slide, 96, 201, 112, 72, C.bpPale, C.bp),
    addBox(slide, 248, 201, 112, 72, C.bpPale, C.bp),
    addBox(slide, 400, 201, 112, 72, C.warnPale, C.warn),
  ];
  addText(slide, "layers", 107, 219, 90, 34, { size: 20, bold: true, align: "center" });
  addText(slide, "loss", 259, 219, 90, 34, { size: 20, bold: true, align: "center" });
  addText(slide, "backward", 411, 219, 90, 34, { size: 20, bold: true, color: C.warn, align: "center" });
  slide.shapes.connect(bpNodes[0], bpNodes[1], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.bp, width: 2 }, head: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(bpNodes[1], bpNodes[2], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.bp, width: 2 }, head: { type: "triangle", width: "sm", length: "sm" } });
  addText(slide, "Global derivatives, gradients, optimizer state", 103, 302, 410, 54, { size: 18, color: C.muted, align: "center" });

  addText(slide, "Hebbian learning", 745, 132, 380, 36, { size: 25, bold: true, color: C.a1, align: "center" });
  for (let i = 0; i < 3; i += 1) {
    addBox(slide, 726 + i * 150, 201, 120, 72, C.a1Pale, C.a1);
    addText(slide, `layer ${i + 1}`, 740 + i * 150, 213, 92, 24, { size: 18, bold: true, color: C.a1, align: "center" });
    addText(slide, "local update", 738 + i * 150, 241, 96, 18, { size: 14, color: C.muted, align: "center" });
  }
  addText(slide, "Input patch + winner + current weights", 721, 302, 430, 54, { size: 18, color: C.muted, align: "center" });

  addLine(slide, 640, 130, 1, 250, C.faint, 1);
  addText(slide, "What resources do these learning paths require in the concrete repository?", 120, 402, 1040, 48, {
    size: 27, bold: true, align: "center",
  });
  addText(slide, "MODELED PROXIES", 137, 500, 800, 28, { size: 16, bold: true, color: C.a1, align: "center" });
  addText(slide, "scalar operations", 137, 537, 250, 32, { size: 22, bold: true });
  addText(slide, "tensor storage", 427, 537, 230, 32, { size: 22, bold: true });
  addText(slide, "logical movement", 687, 537, 250, 32, { size: 22, bold: true });
  addText(slide, "NOT MEASURED", 979, 500, 170, 28, { size: 16, bold: true, color: C.warn });
  addText(slide, "electricity · runtime · GPU peak", 979, 537, 215, 58, { size: 18, bold: true, color: C.warn });
  addNotes(slide, 2, "Frozen analytical methodology in final_report.md.");
}

// Slide 3
{
  const slide = presentation.slides.add();
  addFrame(slide, "Architecture and four execution models", 3);

  const stages = [
    { x: 72, w: 130, title: "Input", sub: "3 × 32 × 32", fill: C.paper, line: C.muted },
    { x: 242, w: 170, title: "Conv 1", sub: "100 · 5×5 · pool 2", fill: C.a1Pale, line: C.a1 },
    { x: 452, w: 170, title: "Conv 2", sub: "196 · 3×3 · pool 2", fill: C.a1Pale, line: C.a1 },
    { x: 662, w: 170, title: "Conv 3", sub: "400 · 3×3 · pool 2", fill: C.a1Pale, line: C.a1 },
    { x: 872, w: 150, title: "Features", sub: "400 × 2 × 2 = 1,600", fill: C.paper, line: C.muted },
    { x: 1062, w: 145, title: "Ridge", sub: "CIFAR-10 readout", fill: C.paper, line: C.muted },
  ];
  stages.map((s) => {
    const b = addBox(slide, s.x, 141, s.w, 92, s.fill, s.line, 12);
    addText(slide, s.title, s.x + 10, 157, s.w - 20, 28, { size: 20, bold: true, color: s.line, align: "center" });
    addText(slide, s.sub, s.x + 10, 190, s.w - 20, 32, { size: 14, color: C.muted, align: "center" });
    return b;
  });
  [204, 414, 624, 834, 1024].forEach((x) =>
    addText(slide, "›", x, 166, 38, 42, { size: 32, bold: true, color: C.muted, align: "center" }),
  );
  addText(slide, "WTA activates the positive top-cutoff winner(s), if any", 176, 265, 928, 40, { size: 23, bold: true, align: "center" });
  addBox(slide, 168, 313, 944, 48, C.bPale, C.b, 10);
  addText(slide, "Deterministic test batch: update matched for all 3 layers within atol 1e−6 and rtol 1e−5", 184, 323, 912, 28, {
    size: 19, bold: true, color: C.b, align: "center",
  });

  addMethodLabel(slide, 90, 416, "A1", C.a1, "Verified direct/local dense", 250);
  addMethodLabel(slide, 370, 416, "A2 conditional", C.a2, "Sparse storage + sparse kernels", 260);
  addMethodLabel(slide, 660, 416, "B", C.b, "Current surrogate/autograd", 230);
  addMethodLabel(slide, 930, 416, "BP", C.bp, "Equal-architecture reference", 260);
  addLine(slide, 90, 520, 1100, 1, C.faint, 1);
  addText(slide, "A2 conditional is an analytical opportunity estimate. The current PyTorch notebook executes dense operations.", 128, 556, 1024, 58, {
    size: 22, bold: true, color: C.warn, align: "center",
  });
  addNotes(slide, 3, "HebbGrad_Github.ipynb; direct_hebb_equivalence_test.py; final_report.md sections 3–5.");
}

// Slide 4
{
  const slide = presentation.slides.add();
  addFrame(slide, "Matched-work compute: one batch of 100", 4);
  const valuesB = batch.map((v) => v / 1e9);
  const chart = slide.charts.add("bar", {
    position: { left: 72, top: 135, width: 760, height: 425 },
    categories: ["A1", "A2 conditional", "B", "BP"],
    series: [{
      name: "Billion operations",
      values: valuesB,
      valuesFormatCode: '0.000" B"',
      fill: C.a1,
      points: [
        { idx: 0, fill: C.a1 }, { idx: 1, fill: C.a2 },
        { idx: 2, fill: C.b }, { idx: 3, fill: C.bp },
      ],
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 70 },
    hasLegend: false,
    xAxis: { visible: false, min: 0, max: 27, majorGridlines: null },
    yAxis: { textStyle: { fill: C.ink, fontSize: 18, bold: true }, line: { fill: "none", width: 0 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 17, bold: true } },
    chartFill: { color: C.bg, transparency: 100000 },
    chartLine: { fill: "none", width: 0 },
    plotAreaFill: { color: C.bg, transparency: 100000 },
    plotAreaLine: { fill: "none", width: 0 },
  });
  styleChart(chart);

  addText(slide, "BP relative to", 905, 150, 250, 28, { size: 18, bold: true, color: C.muted });
  addText(slide, "2.819×", 905, 191, 250, 54, { size: 36, bold: true, color: C.a1 });
  addText(slide, "A1 direct dense", 905, 239, 250, 28, { size: 18, color: C.muted });
  addText(slide, "17.699×", 905, 302, 250, 54, { size: 36, bold: true, color: C.a2 });
  addText(slide, "A2 conditional", 905, 350, 250, 28, { size: 18, color: C.muted });
  addText(slide, "1.415×", 905, 413, 250, 54, { size: 36, bold: true, color: C.b });
  addText(slide, "B surrogate/autograd", 905, 461, 250, 28, { size: 18, color: C.muted });
  addLine(slide, 865, 142, 1, 365, C.faint, 1);
  addBox(slide, 83, 601, 1086, 58, C.a1Pale, C.a1, 10);
  addText(slide, "Direct A1 removes the global backward path. A2 conditional also skips masked higher-layer work.", 105, 616, 1042, 28, {
    size: 21, bold: true, color: C.ink, align: "center",
  });
  addNotes(slide, 4, "results/data/compute_same_work.csv. Values are analytical scalar-operation totals, not energy measurements.");
}

// Slide 5
{
  const slide = presentation.slides.add();
  addFrame(slide, "Full pipeline and persistent state", 5);
  addText(slide, "Method-level compute · 20 epochs", 68, 122, 610, 34, { size: 23, bold: true });
  const computeValuesT = [...method20, bp20].map((v) => v / 1e12);
  const computeChart = slide.charts.add("bar", {
    position: { left: 62, top: 158, width: 620, height: 330 },
    categories: ["A1", "A2 conditional", "B", "BP 20"],
    series: [{
      name: "Trillion operations",
      values: computeValuesT,
      valuesFormatCode: '0.000" T"',
      fill: C.a1,
      points: [
        { idx: 0, fill: C.a1 }, { idx: 1, fill: C.a2 },
        { idx: 2, fill: C.b }, { idx: 3, fill: C.bp },
      ],
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 70 },
    hasLegend: false,
    xAxis: { visible: false, min: 0, max: 275, majorGridlines: null },
    yAxis: { textStyle: { fill: C.ink, fontSize: 16, bold: true }, line: { fill: "none", width: 0 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 14, bold: true } },
    chartFill: { color: C.bg, transparency: 100000 }, chartLine: { fill: "none", width: 0 },
    plotAreaFill: { color: C.bg, transparency: 100000 }, plotAreaLine: { fill: "none", width: 0 },
  });
  styleChart(computeChart);

  addLine(slide, 695, 122, 1, 485, C.faint, 1);
  addText(slide, "Persistent tensor state", 735, 122, 450, 34, { size: 23, bold: true });
  const memoryChart = slide.charts.add("bar", {
    position: { left: 724, top: 158, width: 480, height: 330 },
    categories: ["A1", "A2 conditional", "B", "BP + Adam"],
    series: [{
      name: "MB",
      values: persistentMb,
      valuesFormatCode: '0.000" MB"',
      fill: C.a1,
      points: [
        { idx: 0, fill: C.a1 }, { idx: 1, fill: C.a2 },
        { idx: 2, fill: C.b }, { idx: 3, fill: C.bp },
      ],
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 70 },
    hasLegend: false,
    xAxis: { visible: false, min: 0, max: 17, tickLabelPosition: "none", majorGridlines: null },
    yAxis: { textStyle: { fill: C.ink, fontSize: 15, bold: true }, line: { fill: "none", width: 0 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 13, bold: true } },
    chartFill: { color: C.bg, transparency: 100000 }, chartLine: { fill: "none", width: 0 },
    plotAreaFill: { color: C.bg, transparency: 100000 }, plotAreaLine: { fill: "none", width: 0 },
  });
  styleChart(memoryChart);
  // Artifact Tool currently emits value-axis tick labels for this horizontal
  // chart even when the axis is hidden. Mask only that redundant label strip;
  // the bars and their exact editable data labels remain visible.
  addBox(slide, 724, 455, 480, 44, C.bg, C.bg, 0);

  addBox(slide, 82, 520, 560, 88, C.bpPale, C.bp, 10);
  addText(slide, "BP · 100 epochs", 103, 538, 230, 24, { size: 17, bold: true, color: C.bp });
  addText(slide, `${(bp100 / 1e12).toLocaleString("en-US", { minimumFractionDigits: 3, maximumFractionDigits: 3 })} T operations`, 103, 564, 500, 34, { size: 27, bold: true, color: C.bp });
  addText(slide, "Includes source-appropriate preprocessing and evaluation. ZCA eigendecomposition remains a separate estimate.", 735, 525, 440, 76, { size: 17, color: C.muted });
  addText(slide, "Analytical payloads only. No measured GPU peak memory.", 735, 620, 440, 30, { size: 17, bold: true, color: C.warn });
  addNotes(slide, 5, "results/data/compute_method_level.csv; results/data/memory_persistent_state.csv; final_report.md sections 7–8.");
}

// Slide 6
{
  const slide = presentation.slides.add();
  addFrame(slide, "Accuracy and convergence context", 6);

  addText(slide, "Exact reported observations", 73, 126, 440, 34, { size: 23, bold: true });
  addText(slide, "64.55%", 83, 184, 260, 66, { size: 44, bold: true, color: C.a1 });
  addText(slide, "± 0.4 · 10 runs", 84, 250, 260, 30, { size: 18, color: C.muted });
  addText(slide, "Original Hebbian paper", 84, 288, 340, 30, { size: 20, bold: true });
  addLine(slide, 82, 341, 355, 1, C.faint, 1);
  addText(slide, "64.65%", 83, 370, 260, 66, { size: 44, bold: true, color: C.b });
  addText(slide, "1 saved run", 84, 436, 260, 30, { size: 18, color: C.muted });
  addText(slide, "Repository notebook", 84, 474, 340, 30, { size: 20, bold: true });

  addLine(slide, 500, 123, 1, 440, C.faint, 1);
  addText(slide, "Approximate comparison-paper graph reads", 548, 126, 650, 34, { size: 23, bold: true });
  const hb20 = accuracyLookup("comparison_paper", "HB", 20);
  const bpAcc20 = accuracyLookup("comparison_paper", "BP", 20);
  const hb100 = accuracyLookup("comparison_paper", "HB", 100);
  const bpAcc100 = accuracyLookup("comparison_paper", "BP", 100);
  const accChart = slide.charts.add("bar", {
    position: { left: 535, top: 173, width: 650, height: 350 },
    categories: ["20 epochs", "100 epochs"],
    series: [
      { name: "HB", values: [hb20, hb100], valuesFormatCode: '0"%"', fill: C.a1 },
      { name: "BP", values: [bpAcc20, bpAcc100], valuesFormatCode: '0"%"', fill: C.bp },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 65 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fill: C.ink, fontSize: 16 } },
    xAxis: { textStyle: { fill: C.ink, fontSize: 13 }, line: { style: "solid", fill: C.grid, width: 1 } },
    yAxis: { visible: true, min: 0, max: 75, majorUnit: 15, numberFormatCode: '0"%"', textStyle: { fill: C.muted, fontSize: 13 }, majorGridlines: { style: "solid", fill: C.grid, width: 1 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 15, bold: true } },
    chartFill: { color: C.bg, transparency: 100000 }, chartLine: { fill: "none", width: 0 },
    plotAreaFill: { color: C.bg, transparency: 100000 }, plotAreaLine: { fill: "none", width: 0 },
  });
  styleChart(accChart);
  addBox(slide, 84, 589, 1100, 72, C.warnPale, C.warn, 10);
  addText(slide, "Earlier Hebbian stabilization does not mean 5 Hebbian epochs equal 100 BP epochs in accuracy.", 108, 606, 1052, 42, {
    size: 21, bold: true, color: C.warn, align: "center",
  });
  addNotes(slide, 6, "results/data/accuracy_context.csv; Miconi Table 1; saved notebook output; Gupta et al. Figure 2 approximate graph reads.");
}

// Slide 7
{
  const slide = presentation.slides.add();
  addFrame(slide, "Supported findings and limits", 7);
  addText(slide, "What the model establishes", 76, 130, 510, 36, { size: 25, bold: true, color: C.a1 });
  addText(slide, "A1", 80, 203, 70, 38, { size: 24, bold: true, color: C.a1 });
  addText(slide, "Direct local Instar uses less modeled arithmetic than BP", 160, 199, 430, 60, { size: 21, bold: true });
  addText(slide, "B", 80, 303, 70, 38, { size: 24, bold: true, color: C.b });
  addText(slide, "Dense autograd captures only part of the potential saving", 160, 299, 430, 60, { size: 21, bold: true });
  addText(slide, "A2 conditional", 80, 403, 126, 38, { size: 18, bold: true, color: C.a2 });
  addText(slide, "Largest saving, conditional on real sparse execution", 220, 399, 370, 60, { size: 21, bold: true });
  addText(slide, "STATE", 80, 503, 90, 38, { size: 19, bold: true, color: C.bp });
  addText(slide, "Locality can reduce persistent state and global-gradient traffic", 180, 499, 410, 74, { size: 21, bold: true });

  addLine(slide, 640, 132, 1, 450, C.faint, 1);
  addText(slide, "What the model does not establish", 696, 130, 500, 36, { size: 25, bold: true, color: C.warn });
  const limits = [
    "No measured energy, runtime, or GPU peak memory",
    "BP includes documented assumptions where sources are incomplete",
    "A2 conditional is not the current PyTorch execution",
    "Accuracy protocols and classifiers are not identical",
  ];
  limits.forEach((text, i) => {
    addText(slide, String(i + 1), 702, 203 + i * 92, 36, 36, { size: 19, bold: true, color: C.warn, align: "center" });
    addText(slide, text, 752, 198 + i * 92, 420, 58, { size: 20, bold: true });
  });
  addBox(slide, 110, 618, 1060, 49, C.paper, C.faint, 8);
  addText(slide, "Hardware outcomes still depend on kernels, fusion, memory hierarchy, and sparse-access overhead.", 134, 629, 1012, 26, { size: 18, color: C.muted, align: "center" });
  addNotes(slide, 7, "Frozen final-report conclusions and limitations.");
}

// Slide 8
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  addText(slide, "Conclusion", 78, 62, 550, 58, { size: 38, bold: true, color: C.paper });
  const conclusions = [
    ["01", "Direct local Hebbian learning has a clear modeled resource advantage for this network."],
    ["02", "Implementation matters: A1 improves on B, while A2 conditional needs genuine sparse execution."],
    ["03", "The result concerns resource efficiency, not equal accuracy, measured energy, or universal superiority."],
  ];
  conclusions.forEach(([n, text], i) => {
    const y = 165 + i * 132;
    addText(slide, n, 86, y + 5, 60, 42, { size: 22, bold: true, color: i === 1 ? C.a2 : C.a1 });
    addText(slide, text, 170, y, 905, 88, { size: 27, bold: true, color: C.paper });
    if (i < 2) addLine(slide, 170, y + 103, 900, 1, "#40505F", 1);
  });
  addText(slide, "Q&A", 928, 575, 260, 68, { size: 48, bold: true, color: C.a1, align: "right" });
  addText(slide, "Bayan & Shay", 82, 633, 380, 30, { size: 18, color: "#C6D0D8" });
  addNotes(slide, 8, "Frozen final-report conclusion.");
}

const requirements = {
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [4, 5, 6],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [FONT] };
const expectedSlideSizeEmu = "12192000,6858000";
const stagingDir = path.join(here, ".finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "hebbian-compute-presentation-candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: requirements.requiredNativeChartOwnerSlides,
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.${Date.now()}.validation.json`),
});

console.log(`Created ${FINAL_PPTX}`);
