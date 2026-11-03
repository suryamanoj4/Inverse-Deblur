// Mid-evaluation deck for the deblurring study.
//
// Build (pptxgenjs, react-icons, react, react-dom and sharp must be resolvable):
//   NODE_PATH=~/.cache/deck-tools/node_modules node report/mid_evaluation_deck.js
// Chart values are read from report/generated/*.json, so the deck follows the
// analysis scripts; text figures are quoted from the same files.

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(path.join(
  process.env.HOME,
  ".claude/skills/synced/667ee6da-69a4-4ea5-a086-65a2235b984f_057eb7c7-2170-4660-98c4-bc739d56209c/pptx/scripts/apply_theme.js"
));

const ROOT = path.resolve(__dirname, "..");
const GEN = path.join(ROOT, "report", "generated");
const OUT = path.join(ROOT, "report", "mid_evaluation.pptx");
const numbers = JSON.parse(fs.readFileSync(path.join(GEN, "numbers.json")));
const h3lpips = JSON.parse(fs.readFileSync(path.join(GEN, "h3_lpips.json")));
const texture = JSON.parse(fs.readFileSync(path.join(GEN, "texture.json"))).sets;

// ---- theme ---------------------------------------------------------------
// Ink and slate dominate; rust marks M1 (data-only), teal M2 (physics-aware),
// violet M3, amber is the single highlight.
const THEME = {
  name: "Deblur Study",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2030", lt1: "FFFFFF", dk2: "4A5468", lt2: "F1F3F6",
    accent1: "C8553D", accent2: "2A9D8F", accent3: "E9A23B",
    accent4: "7A8699", accent5: "7A5195", accent6: "5E9C5A",
    hlink: "2A9D8F", folHlink: "7A5195",
  },
};
const HEX = THEME.colors;
const M1 = HEX.accent1, M2 = HEX.accent2, AMBER = HEX.accent3, M3 = HEX.accent5;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Mid evaluation: classical vs. modern deblurring";
pres.author = "Surya Manoj Pentakota, Kevin Paul Reddy Janagari";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;

// ---- layouts ---------------------------------------------------------------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.9,
      fontSize: 42, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.1, w: 11.7, h: 1.6,
      fontSize: 20, color: C.background2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9,
      fontSize: 28, bold: true, color: C.text1, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "Mid evaluation · Inverse-function learning for image deblurring",
      options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent4, isTextBox: true } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4 },
});

// ---- helpers ---------------------------------------------------------------
async function iconPng(Icon, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: "#" + color, size }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

function shadow() {
  return { type: "outer", color: "000000", blur: 6, offset: 2, angle: 90, opacity: 0.12 };
}

function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: opts.fill || C.background2 },
    line: { color: opts.line || opts.fill || C.background2, width: opts.lineWidth || 0.75 },
    shadow: opts.shadow === false ? undefined : shadow(),
    objectName: opts.name,
  });
}

async function iconCircle(slide, Icon, x, y, d, color, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color }, objectName: name });
  const pad = d * 0.24;
  slide.addImage({ data: await iconPng(Icon), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad,
    altText: name || "icon" });
}

function text(slide, value, opts) {
  slide.addText(value, { isTextBox: true, margin: 0, valign: "top", fontSize: 15, color: C.text1, ...opts });
}

function bullets(slide, items, opts) {
  const runs = items.map((item, i) => {
    const parts = Array.isArray(item) ? item : [{ text: item }];
    return parts.map((p, j) => ({
      text: p.text,
      options: { ...(p.options || {}), bullet: j === 0 ? { indent: 16 } : undefined,
        breakLine: j === parts.length - 1 && i < items.length - 1 },
    }));
  }).flat();
  slide.addText(runs, { isTextBox: true, margin: 0, valign: "top", fontSize: 15, color: C.text1,
    paraSpaceAfter: 8, ...opts });
}

function pill(slide, label, x, y, w, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.34, rectRadius: 0.17,
    fill: { color }, line: { color } });
  slide.addText(label, { isTextBox: true, x, y, w, h: 0.34, margin: 0, align: "center", valign: "middle",
    fontSize: 11, bold: true, color: C.background1 });
}

const CHART_TEXT = {
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  legendFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, dataLabelColor: HEX.dk1,
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 12,
  valAxisTitleFontSize: 12, valAxisTitleColor: HEX.dk2,
  valGridLine: { color: "E1E5EA", size: 0.5 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false,
};

function content(title, section) {
  const slide = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  slide.addText(title, { placeholder: "title" });
  return slide;
}

const f2 = (v) => v.toFixed(2);

const METHOD_LABEL = { wiener: "Wiener", rl: "R–L", fista: "FISTA-TV", epll: "ADMM-EPLL", unet: "U-Net", nafnet: "NAFNet" };
// One colour per method across every chart: grays for untuned baselines, teal shades for M2, rust shades for M1.
const METHOD_HEX = { blurred: "1B2030", wiener: "B7BEC9", rl: "7A8699", fista: "8FCFC6", epll: M2, unet: "E3A08F", nafnet: M1 };

function legendRow(slide, keys, x, y, w) {
  const step = w / keys.length;
  keys.forEach((key, i) => {
    const label = key === "blurred" ? "Blurred input" : METHOD_LABEL[key];
    slide.addShape(pres.shapes.OVAL, { x: x + i * step, y: y + 0.08, w: 0.18, h: 0.18,
      fill: { color: METHOD_HEX[key] }, line: { color: METHOD_HEX[key] } });
    text(slide, label, { x: x + i * step + 0.25, y, w: step - 0.3, h: 0.34, fontSize: 12, color: C.text2,
      valign: "middle" });
  });
}

// ---- slides ----------------------------------------------------------------
async function build() {
  // 1. Title ---------------------------------------------------------------
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("Should a deblurring model know the physics?", { placeholder: "title" });
  s.addText([
    { text: "Classical vs. modern inverse-function learning · Mid evaluation", options: { breakLine: true } },
    { text: "Surya Manoj Pentakota (2026701036) · Kevin Paul Reddy Janagari (2026900006) · IIIT Hyderabad",
      options: { fontSize: 15, color: C.accent4 } },
  ], { placeholder: "body" });
  await iconCircle(s, fa.FaEye, 0.8, 0.9, 0.9, M2, "title icon");
  s.addNotes("We are comparing deblurring methods that differ in one thing: whether they use the physics of the blur. " +
    "This mid evaluation covers what is done (two of three arms), how it is built, what went wrong and how we fixed it, " +
    "what is left, and what we expect at the final evaluation.");

  // 2. Agenda --------------------------------------------------------------
  s = content("What this talk covers", "Introduction");
  const agenda = [
    [fa.FaFlagCheckered, "Work completed", "Status, protocol and timeline"],
    [fa.FaProjectDiagram, "Algorithms, models", "M1 networks, M2 solvers, M3 unrolled nets"],
    [fa.FaCogs, "Implementation", "Evaluator, GPU port, statistics"],
    [fa.FaChartBar, "Results so far", "Mismatch, stability, texture, perceptual"],
    [fa.FaTools, "Challenges and fixes", "What broke and how we addressed it"],
    [fa.FaRoute, "Remaining work, plan", "Next steps and expected outcomes"],
  ];
  for (let i = 0; i < agenda.length; i++) {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.6 + row * 2.6;
    card(s, x, y, 3.8, 2.35, { name: `agenda card ${i + 1}` });
    await iconCircle(s, agenda[i][0], x + 0.3, y + 0.3, 0.7, i < 3 ? M2 : M1, `agenda icon ${i + 1}`);
    text(s, `${i + 1}. ${agenda[i][1]}`, { x: x + 0.3, y: y + 1.12, w: 3.3, h: 0.35, fontSize: 17, bold: true });
    text(s, agenda[i][2], { x: x + 0.3, y: y + 1.55, w: 3.3, h: 0.6, fontSize: 14, color: C.text2 });
  }
  s.addNotes("Six parts, matching the evaluation requirements: work completed; algorithms, models and implementation; " +
    "results; challenges; remaining work and future plans; expected outcomes.");

  // 2b. Executive summary -------------------------------------------------
  s = content("The short version: what we found so far", "Introduction");
  {
    const rows = [
      ["Is the deep network better on training kernels?", `Tied in PSNR (${f2(numbers.mismatch.nafnet.P1)} vs ${f2(numbers.mismatch.epll.P1)} dB); NAFNet is better perceptually`, "Tie", AMBER],
      ["Does it survive an unseen blur kernel?", "Not on motion: it loses half its gain; ADMM-EPLL wins every unseen-kernel cell, by up to 6.4 dB", "M2 wins", M2],
      ["Is the gap about not knowing the noise level?", "No: knowing σ is worth at most 0.25 dB to NAFNet; the kernel is the missing information", "Kernel", M2],
      ["Does image texture explain it (H2)?", "No: NAFNet's small edge comes from smooth images, the opposite of H2", "Not H2", M1],
      ["Is the deep network stable (H3)?", "ADMM-EPLL is 2–17× more stable at σ ≥ 5; at σ = 1 under mismatch it is not", "Mostly M2", M2],
    ];
    for (let i = 0; i < rows.length; i++) {
      const y = 1.45 + i * 0.95;
      card(s, 0.6, y, 12.1, 0.8, { name: `summary row ${i + 1}`, shadow: false });
      text(s, rows[i][0], { x: 0.85, y: y + 0.12, w: 4.3, h: 0.56, fontSize: 15, bold: true, valign: "middle" });
      text(s, rows[i][1], { x: 5.3, y: y + 0.12, w: 5.6, h: 0.56, fontSize: 14, color: C.text2, valign: "middle" });
      pill(s, rows[i][2], 11.15, y + 0.23, 1.35, rows[i][3]);
    }
    card(s, 0.6, 6.25, 12.1, 0.6, { fill: C.text1, line: C.text1, shadow: false, name: "headline" });
    text(s, "Headline: neither side simply wins. Physics-awareness is what transfers.",
      { x: 0.85, y: 6.3, w: 11.6, h: 0.5, fontSize: 16, bold: true, color: C.background1, valign: "middle" });
  }
  s.addNotes("If you remember one slide, it is this one. On the kernels the network was trained on, the two arms are tied. " +
    "On kernels it never saw, especially new motion directions, the physics-aware solver wins clearly. The rest of the talk explains how we know.");

  // 3. Problem -------------------------------------------------------------
  s = content("Deblurring is an ill-posed inverse problem", "Introduction");
  card(s, 0.6, 1.5, 6.0, 4.9, { name: "equation card" });
  text(s, "y = A x + ε", { x: 0.9, y: 1.85, w: 5.4, h: 0.9, fontSize: 40, bold: true, fontFace: THEME.headFontFace,
    color: C.text1 });
  bullets(s, [
    "x: sharp image · y: what the camera records",
    "A: blur by a point-spread function (PSF) k",
    "ε: sensor noise, Gaussian with level σ",
    "A's spectrum decays to ~0 at high frequencies, so naive inversion amplifies noise",
  ], { x: 0.9, y: 2.95, w: 5.4, h: 3.2 });
  card(s, 7.0, 1.5, 5.7, 4.9, { fill: C.text1, line: C.text1, name: "question card" });
  await iconCircle(s, fa.FaQuestion, 7.35, 1.85, 0.75, AMBER, "question icon");
  text(s, "The question", { x: 7.35, y: 2.8, w: 5.0, h: 0.5, fontSize: 20, bold: true, color: C.background1 });
  text(s, "Should the restoring function g be built around the physics of A, or learned purely from data?",
    { x: 7.35, y: 3.35, w: 5.0, h: 1.3, fontSize: 18, color: C.background1 });
  text(s, "Published methods differ along exactly this axis but are tested under different protocols. We compare them under one.",
    { x: 7.35, y: 4.75, w: 5.0, h: 1.3, fontSize: 15, color: C.background2 });
  s.addNotes("A sharp image is blurred by a known kind of operator and corrupted by noise. The operator is badly conditioned, " +
    "so inverting it directly blows up the noise. The question is whether a good restorer should use the operator itself, " +
    "or learn everything from example pairs. Existing papers use different data and settings, so their numbers cannot be compared; " +
    "our contribution is running them under one locked protocol.");

  // 3b. What blur looks like ----------------------------------------------
  s = content("What blur does, and why inverting it fails", "Introduction");
  {
    const Q = path.join(GEN, "qualitative");
    const pairs = [
      ["P1_motion_l15_a45_s5", "Motion blur", "15 px at 45°, noise σ = 5"],
      ["P2_extrap_defocus_r7_s5", "Defocus blur", "disk of radius 7 px, noise σ = 5"],
      ["P2_interp_gaussian_sigma1.4_s15", "Gaussian blur + strong noise", "σk = 1.4, noise σ = 15"],
    ];
    for (let i = 0; i < 3; i++) {
      const x = 0.6 + i * 4.1;
      ["sharp", "observed"].forEach((kind, j) => {
        s.addImage({ path: path.join(Q, `${pairs[i][0]}_${kind}.png`), x: x + j * 1.95, y: 1.45, w: 1.8, h: 1.8,
          altText: `${pairs[i][1]}: ${kind === "sharp" ? "sharp original" : "blurred observation"}` });
      });
      text(s, "sharp  →  observed", { x, y: 3.3, w: 3.75, h: 0.3, fontSize: 12, color: C.text2, align: "center" });
      text(s, pairs[i][1], { x, y: 3.6, w: 3.75, h: 0.35, fontSize: 15, bold: true, align: "center" });
      text(s, pairs[i][2], { x, y: 3.92, w: 3.75, h: 0.3, fontSize: 12, color: C.text2, align: "center" });
    }
    card(s, 0.6, 4.45, 12.1, 2.35, { name: "inversion card" });
    const strip = [["observed", "Observed"], ["wiener", "Direct inverse (Wiener)"], ["epll", "With a learned prior"]];
    strip.forEach(([kind, label], j) => {
      s.addImage({ path: path.join(Q, `P1_motion_l15_a45_s5_${kind}.png`), x: 0.85 + j * 1.9, y: 4.65, w: 1.7, h: 1.7,
        altText: `${label}, motion blur example` });
      text(s, label, { x: 0.85 + j * 1.9, y: 6.38, w: 1.75, h: 0.3, fontSize: 11, color: C.text2, align: "center" });
    });
    text(s, "Undoing the blur directly boosts the frequencies the blur destroyed, and the noise with them (middle). Every good method adds a prior: an assumption about what natural images look like (right).",
      { x: 6.75, y: 4.8, w: 5.7, h: 1.4, fontSize: 15 });
    text(s, "The question is where that prior and the blur model come from.",
      { x: 6.75, y: 6.15, w: 5.7, h: 0.45, fontSize: 15, bold: true, color: M2 });
  }
  s.addNotes("Top row: three kinds of blur from our test set. Bottom: if you simply invert the blur, the noise is amplified, " +
    "as in the middle image. Adding a prior gives the clean result on the right. Every method we compare is a different way of combining " +
    "the blur model with a prior.");

  // 4. Arms and hypotheses --------------------------------------------------
  s = content("Three arms and three hypotheses", "Introduction");
  const arms = [
    ["M1 · data-only", "NAFNet, U-Net", "Sees only y. Learns the inverse end to end.", M1, fa.FaBrain],
    ["M2 · classical ML, physics-aware", "ADMM-EPLL, FISTA-TV, Wiener, R–L", "Knows k and σ. Learned shallow prior + solver.", M2, fa.FaCalculator],
    ["M3 · unrolled, physics-aware", "USRNet, ISTA-Net, LISTA", "Knows k and σ. Deep prior inside a solver.", M3, fa.FaLayerGroup],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.45, 3.8, 2.6, { name: `arm card ${i + 1}` });
    await iconCircle(s, arms[i][4], x + 0.3, 1.7, 0.65, arms[i][3], `arm icon ${i + 1}`);
    text(s, arms[i][0], { x: x + 1.1, y: 1.72, w: 2.55, h: 0.65, fontSize: 16, bold: true, color: arms[i][3],
      valign: "middle" });
    text(s, arms[i][1], { x: x + 0.3, y: 2.55, w: 3.3, h: 0.6, fontSize: 14, bold: true });
    text(s, arms[i][2], { x: x + 0.3, y: 3.2, w: 3.3, h: 0.75, fontSize: 14, color: C.text2 });
  }
  const hyps = [
    ["H1", "Physics-awareness pays off as information degrades: more noise, more texture, kernel mismatch"],
    ["H2", "Prior depth matters most on texture: deep priors win on natural texture"],
    ["H3", "Physics-awareness buys stability against tiny adversarial perturbations"],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 4.4 + i * 0.78;
    pill(s, hyps[i][0], 0.6, y + 0.12, 0.8, C.text1);
    text(s, hyps[i][1], { x: 1.6, y: y + 0.08, w: 11.1, h: 0.45, fontSize: 16, valign: "middle" });
  }
  s.addNotes("M3 is the control arm: it shares M2's physics and M1's deep prior, so it separates the two axes. " +
    "Each hypothesis is stated over the arms; today we can test them for M1 versus M2.");

  // 5. Status --------------------------------------------------------------
  pres.addSection({ title: "Work completed" });
  s = content("Status at mid-evaluation: M1 and M2 complete", "Work completed");
  const stats = [
    ["7", "M1 models trained", "NAFNet, U-Net, 5 per-σ specialists", M1],
    ["5", "GMM priors fitted", "K = 1, 4, 16, 64, 200", M2],
    ["27,000", "cells per method", "200 images × 27 PSFs × 5 noise levels", AMBER],
    ["133", "unit tests", "operators, protocol, solvers, models", HEX.dk2],
  ];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.08;
    card(s, x, 1.45, 2.85, 2.3, { name: `stat card ${i + 1}` });
    text(s, stats[i][0], { x: x + 0.25, y: 1.6, w: 2.45, h: 0.95, fontSize: 40, bold: true, color: stats[i][3],
      fontFace: THEME.headFontFace });
    text(s, stats[i][1], { x: x + 0.25, y: 2.6, w: 2.4, h: 0.4, fontSize: 16, bold: true });
    text(s, stats[i][2], { x: x + 0.25, y: 3.0, w: 2.45, h: 0.6, fontSize: 12, color: C.text2 });
  }
  const status = [
    ["Shared foundation: FFT operator, PSF library, protocol, metrics", "done"],
    ["M1: training and full test evaluation", "done"],
    ["M2: priors, λ tuning, full test evaluation, K-sweep", "done"],
    ["Analyses: mismatch, σ-oracle, inverse crime, H2, H3, LPIPS", "done"],
    ["M3: implemented and unit-tested; two bugs to fix", "blocked"],
    ["M3 training and three-arm evaluation", "next"],
  ];
  for (let i = 0; i < status.length; i++) {
    const col = Math.floor(i / 3), row = i % 3;
    const x = 0.6 + col * 6.15, y = 4.15 + row * 0.82;
    const color = status[i][1] === "done" ? HEX.accent6 : status[i][1] === "blocked" ? M1 : AMBER;
    pill(s, status[i][1], x, y + 0.08, 0.95, color);
    text(s, status[i][0], { x: x + 1.15, y, w: 4.8, h: 0.5, fontSize: 14, valign: "middle" });
  }
  s.addNotes("Two of the three arms are trained and fully evaluated. Every method has been scored on the same 27,000 test cells. " +
    "M3 is coded and unit-tested but has two bugs we found while smoke-testing; we deferred it to finish M1 versus M2 first.");

  // 6. Protocol ------------------------------------------------------------
  s = content("One locked protocol for every method", "Work completed");
  const proto = [
    [fa.FaImages, "Data", "BSDS500 luminance · 200 train / 100 val / 200 test"],
    [fa.FaWaveSquare, "Noise grid", "σ ∈ {0, 1, 5, 15, 25} / 255"],
    [fa.FaDice, "Seeded cells", "noise seeded per (image, PSF, σ): every method sees the identical y"],
    [fa.FaLock, "Selection on val only", "fixed 10-image val subset; one checkpoint per model; test never used"],
  ];
  for (let i = 0; i < proto.length; i++) {
    const y = 1.5 + i * 1.25;
    await iconCircle(s, proto[i][0], 0.6, y, 0.7, M2, `protocol icon ${i + 1}`);
    text(s, proto[i][1], { x: 1.55, y: y - 0.02, w: 5.0, h: 0.4, fontSize: 17, bold: true });
    text(s, proto[i][2], { x: 1.55, y: y + 0.38, w: 5.0, h: 0.6, fontSize: 14, color: C.text2 });
  }
  s.addTable([
    [{ text: "PSF set", options: { bold: true, color: C.background1, fill: { color: C.text1 } } },
     { text: "Kernels", options: { bold: true, color: C.background1, fill: { color: C.text1 } } },
     { text: "Role", options: { bold: true, color: C.background1, fill: { color: C.text1 } } }],
    ["P1 (12)", "Gaussian σk 0.7–2.0; motion 9, 15 px at 0/45/90/135°", "training + in-distribution test"],
    ["P2-interp (11)", "Gaussian σk 0.9–1.8; motion 11, 13 px at 22.5/67.5/112.5/157.5°", "unseen parameters"],
    ["P2-extrap (4)", "defocus disks, radius 3–9 px", "unseen family"],
  ], { x: 6.9, y: 1.5, w: 5.8, colW: [1.35, 2.75, 1.7], fontSize: 13, color: C.text1,
    border: { type: "solid", pt: 0.5, color: "D5DAE1" }, fill: { color: C.background1 }, valign: "middle",
    rowH: [0.45, 0.9, 0.9, 0.6] });
  text(s, "Circular FFT convolution, observations clipped to [0, 1]. M1 trains only on P1; P2 measures kernel mismatch.",
    { x: 6.9, y: 4.75, w: 5.8, h: 0.8, fontSize: 14, color: C.text2 });
  s.addNotes("The protocol is what makes the comparison fair. Same images, same kernels, same noisy observations, selection on " +
    "validation only. P1 is what M1 trains on; P2-interp has unseen settings of the same kernel types, and P2-extrap is a kernel " +
    "type nobody trains on.");

  // 6b. Kernel gallery ----------------------------------------------------
  s = content("Training kernels vs. unseen kernels", "Work completed");
  {
    const K = path.join(GEN, "deck");
    const sets = [
      ["P1", "P1 · training kernels (12)", "4 Gaussian widths and 8 motion blurs: lengths 9 and 15 px at 0°, 45°, 90°, 135°. M1 learns from these.", 1.45, 1.75, 5.04, HEX.dk2],
      ["P2_interp", "P2-interp · unseen settings (11)", "Same families, new parameters: Gaussian widths in between, motion of 11 and 13 px at 22.5°, 67.5°, 112.5°, 157.5°.", 3.45, 1.75, 5.04, M1],
      ["P2_extrap", "P2-extrap · unseen family (4)", "Defocus disks of radius 3, 5, 7, 9 px: a kind of blur no model trains on.", 5.45, 1.1, 4.03, M1],
    ];
    for (const [key, title, body, y, h, w, color] of sets) {
      s.addImage({ path: path.join(K, `kernels_${key}.png`), x: 0.6, y, w, h, altText: `${title} kernel pictures` });
      text(s, title, { x: 6.0, y: y + 0.1, w: 6.7, h: 0.4, fontSize: 18, bold: true, color });
      text(s, body, { x: 6.0, y: y + 0.55, w: 6.7, h: 1.0, fontSize: 15, color: C.text2 });
    }
  }
  s.addNotes("Each square is one blur kernel: the pattern a single bright point is smeared into. P1 is what the network trains on. " +
    "P2-interp keeps the same kinds of blur but changes the settings, for example new motion directions. P2-extrap is a new kind of blur, " +
    "defocus. Physics-aware methods are given the correct kernel in every case; the data-only network has to cope without it.");

  // 7. M1 ------------------------------------------------------------------
  pres.addSection({ title: "Algorithms and implementation" });
  s = content("M1: data-only networks learn y → x directly", "Algorithms and implementation");
  const m1 = [
    ["NAFNet", "primary", "17.1 M parameters", "U-Net of NAFBlocks: SimpleGate, simplified channel attention, no nonlinear activations",
      "29.67 dB val · 2.8 h"],
    ["U-Net", "architecture baseline", "7.8 M parameters", "Plain encoder–decoder with skip connections",
      "28.43 dB val · 0.5 h"],
  ];
  for (let i = 0; i < 2; i++) {
    const x = 0.6 + i * 4.05;
    card(s, x, 1.45, 3.8, 3.4, { name: `m1 card ${i + 1}` });
    await iconCircle(s, fa.FaBrain, x + 0.3, 1.7, 0.6, M1, `m1 icon ${i + 1}`);
    text(s, m1[i][0], { x: x + 1.05, y: 1.68, w: 2.5, h: 0.4, fontSize: 20, bold: true });
    text(s, m1[i][1], { x: x + 1.05, y: 2.08, w: 2.5, h: 0.3, fontSize: 12, color: M1, bold: true });
    text(s, m1[i][2], { x: x + 0.3, y: 2.6, w: 3.3, h: 0.35, fontSize: 15, bold: true });
    text(s, m1[i][3], { x: x + 0.3, y: 3.0, w: 3.3, h: 1.0, fontSize: 14, color: C.text2 });
    text(s, m1[i][4], { x: x + 0.3, y: 4.25, w: 3.3, h: 0.35, fontSize: 14, bold: true, color: M1 });
  }
  card(s, 8.75, 1.45, 3.95, 3.4, { fill: C.text1, line: C.text1, name: "recipe card" });
  text(s, "Training recipe", { x: 9.05, y: 1.7, w: 3.4, h: 0.4, fontSize: 17, bold: true, color: C.background1 });
  bullets(s, [
    "One model on P1 × all noise levels",
    "128² patches; k and σ drawn uniformly per patch",
    "AdamW, L2 loss, batch 16, lr 2×10⁻⁴",
    "100k iterations, bf16, RTX 3090 Ti",
  ], { x: 9.05, y: 2.2, w: 3.45, h: 2.5, fontSize: 14, color: C.background1, paraSpaceAfter: 6 });
  card(s, 0.6, 5.15, 12.1, 1.45, { name: "specialists card" });
  await iconCircle(s, fa.FaSlidersH, 0.9, 5.45, 0.6, AMBER, "specialists icon");
  text(s, "Ablation: five per-σ specialists", { x: 1.75, y: 5.35, w: 10.6, h: 0.4, fontSize: 16, bold: true });
  text(s, "NAFNet fine-tuned 10k iterations at one noise level each, to measure what knowing σ would be worth to M1.",
    { x: 1.75, y: 5.8, w: 10.6, h: 0.6, fontSize: 14, color: C.text2 });
  s.addNotes("M1 never sees the kernel or the noise level. NAFNet is a recent state-of-the-art restoration network; the U-Net is " +
    "there to check whether results depend on the architecture. Both are trained once over all training kernels and noise levels, " +
    "and the checkpoint is chosen on validation only.");

  // 8. M2 ------------------------------------------------------------------
  s = content("M2: physics-aware solvers with learned or fixed priors", "Algorithms and implementation");
  text(s, "x̂ = argmin ½‖Ax − y‖² + λ R(x)     (true k and σ are given)",
    { x: 0.6, y: 1.35, w: 12.1, h: 0.5, fontSize: 18, bold: true, color: M2, fontFace: THEME.headFontFace });
  const m2 = [
    ["Wiener", "Closed form in Fourier: X̂ = K̄Y / (|K|² + λ). Untuned baseline.", false],
    ["Richardson–Lucy", "Multiplicative EM for a Poisson likelihood, 30 iterations. Untuned baseline.", false],
    ["FISTA-TV", "Accelerated proximal gradient with a total-variation prior; TV prox by ADMM. 100 × 30 iterations.", false],
    ["ADMM-EPLL  (primary)", "Half-quadratic splitting with a 200-component GMM patch prior. Data step in closed form (FFT); prior step = per-patch Wiener filter of the most likely component; denoiser strength annealed 49/255 → σ.", true],
  ];
  for (let i = 0; i < 4; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.15, y = 2.1 + row * 2.05;
    const primary = m2[i][2];
    card(s, x, y, 5.95, 1.8, { fill: primary ? C.text1 : C.background2, line: primary ? C.text1 : C.background2,
      name: `m2 card ${i + 1}` });
    text(s, m2[i][0], { x: x + 0.3, y: y + 0.2, w: 5.4, h: 0.4, fontSize: 17, bold: true,
      color: primary ? AMBER : C.text1 });
    text(s, m2[i][1], { x: x + 0.3, y: y + 0.65, w: 5.4, h: 1.05, fontSize: 14,
      color: primary ? C.background1 : C.text2 });
  }
  card(s, 0.6, 6.2, 12.1, 0.6, { fill: C.background1, line: "D5DAE1", shadow: false, name: "tuning note" });
  text(s, "λ tuned per noise level on val: 10 multiples (2⁻⁵ … 2⁴) of a default → M2 has five tuned scalars per method. GMM fitted by EM on 100k patches (7 min).",
    { x: 0.85, y: 6.27, w: 11.6, h: 0.45, fontSize: 13, color: C.text2, valign: "middle" });
  s.addNotes("All M2 solvers solve the same MAP problem with the true blur and noise level. ADMM-EPLL is the primary method: " +
    "its prior is a Gaussian mixture over 8x8 patches learned from training images. FISTA-TV uses a hand-crafted total-variation prior. " +
    "Wiener and Richardson-Lucy are classical untuned baselines. Only five regularisation weights per method are tuned.");

  // 9. M3 ------------------------------------------------------------------
  s = content("M3: unrolled networks put a deep prior in the solver", "Algorithms and implementation");
  const m3 = [
    ["USRNet", "primary", "4.5 M", "8 unrolled HQS iterations: closed-form FFT data step + ResU-Net prior; an MLP maps σ to per-iteration weights"],
    ["ISTA-Net", "ablation", "0.19 M", "Unrolled ISTA with a learned convolutional transform and soft-thresholding"],
    ["LISTA", "ablation", "117", "Unrolled ISTA with a shared 1×1 analysis/synthesis pair: the smallest prior"],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.5, 3.8, 3.55, { name: `m3 card ${i + 1}` });
    await iconCircle(s, fa.FaLayerGroup, x + 0.3, 1.75, 0.6, M3, `m3 icon ${i + 1}`);
    text(s, m3[i][0], { x: x + 1.05, y: 1.73, w: 2.5, h: 0.4, fontSize: 20, bold: true });
    text(s, m3[i][1], { x: x + 1.05, y: 2.13, w: 2.5, h: 0.3, fontSize: 12, bold: true, color: M3 });
    text(s, m3[i][2], { x: x + 0.3, y: 2.6, w: 3.3, h: 0.7, fontSize: 32, bold: true, color: M3,
      fontFace: THEME.headFontFace });
    text(s, "parameters", { x: x + 0.3, y: 3.25, w: 3.3, h: 0.3, fontSize: 12, color: C.text2 });
    text(s, m3[i][3], { x: x + 0.3, y: 3.65, w: 3.3, h: 1.3, fontSize: 13, color: C.text2 });
  }
  card(s, 0.6, 5.35, 12.1, 1.3, { fill: C.text1, line: C.text1, name: "m3 status card" });
  await iconCircle(s, fa.FaWrench, 0.9, 5.6, 0.75, AMBER, "m3 status icon");
  text(s, "Status: implemented and unit-tested, not yet trained", { x: 1.95, y: 5.55, w: 10.4, h: 0.4, fontSize: 16,
    bold: true, color: C.background1 });
  text(s, "Same training regime as M1 (P1 × all noise levels; batch 8, lr 1e-4, ≤ 30k iterations), but the network also receives the true PSF spectrum and σ.",
    { x: 1.95, y: 5.95, w: 10.4, h: 0.6, fontSize: 13, color: C.background2 });
  s.addNotes("M3 combines the two: a closed-form physics step and a learned deep prior, iterated. It is what lets us separate " +
    "'knows the physics' from 'has a deep prior'. The code is in place; training is the main remaining work.");

  // 10. Implementation -----------------------------------------------------
  s = content("Techniques and implementation details", "Algorithms and implementation");
  const tech = [
    [fa.FaTable, "Unified evaluator", "Scores any arm on identical cells plus the unrestored input, so every row has a paired gain. Seeded subsets and sharding."],
    [fa.FaMicrochip, "GPU port of M2", "FISTA-TV and ADMM-EPLL rewritten in PyTorch, operation for operation; max difference 10⁻¹³ / 2×10⁻⁷ vs NumPy."],
    [fa.FaBalanceScale, "Paired statistics", "Image as the unit (n = 200); Wilcoxon signed-rank with Holm correction over 30 tests."],
    [fa.FaShieldAlt, "Stability attack (H3)", "20-step PGD on NAFNet in an L2 ball; the same perturbed y is given to ADMM-EPLL (transfer attack)."],
    [fa.FaSearch, "Unbiased examples", "Qualitative images are the median case of each cell, crops chosen by detail, never by eye."],
    [fa.FaRedo, "Reproducible", "Protocol manifest + git commit in every run; every table generated by a script from per-cell CSVs."],
  ];
  for (let i = 0; i < tech.length; i++) {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.45 + row * 2.7;
    card(s, x, y, 3.8, 2.45, { name: `tech card ${i + 1}` });
    await iconCircle(s, tech[i][0], x + 0.3, y + 0.3, 0.6, i === 1 ? AMBER : M2, `tech icon ${i + 1}`);
    text(s, tech[i][1], { x: x + 1.05, y: y + 0.35, w: 2.6, h: 0.5, fontSize: 16, bold: true, valign: "middle" });
    text(s, tech[i][2], { x: x + 0.3, y: y + 1.05, w: 3.3, h: 1.3, fontSize: 13, color: C.text2 });
  }
  s.addNotes("The evaluator is shared by all arms, so numbers pair one to one. The GPU port of the classical solvers made the " +
    "M2 evaluation feasible. Statistics treat the image as the unit, so the 27 kernels of one image are not counted as independent samples.");

  // 10b. How to read the results ------------------------------------------
  pres.addSection({ title: "Results so far" });
  s = content("How we measure: the metrics in plain language", "Results so far");
  {
    const metrics = [
      [fa.FaBullseye, "PSNR (dB, higher is better)", "Pixel accuracy. +1 dB is a clearly visible improvement; it can favour slightly blurry results.", M2],
      [fa.FaShapes, "SSIM (0–1, higher is better)", "Local structural similarity: closer to how an image looks than PSNR, still simple.", M2],
      [fa.FaEye, "LPIPS (lower is better)", "A learned perceptual distance: images that look alike score low even if pixels differ.", M2],
      [fa.FaUndo, "Data consistency (lower)", "Re-blur the output with the true kernel: does it reproduce the observation? It should leave about σ.", AMBER],
      [fa.FaEquals, "“Blurred” baseline", "The unrestored observation. A method below this line is worse than doing nothing.", AMBER],
      [fa.FaBalanceScale, "Paired tests", "Is a difference consistent over all 200 images? Wilcoxon signed-rank, Holm-corrected.", AMBER],
    ];
    for (let i = 0; i < metrics.length; i++) {
      const col = i % 3, row = Math.floor(i / 3);
      const x = 0.6 + col * 4.1, y = 1.45 + row * 2.6;
      card(s, x, y, 3.8, 2.35, { name: `metric card ${i + 1}` });
      await iconCircle(s, metrics[i][0], x + 0.3, y + 0.3, 0.6, metrics[i][3], `metric icon ${i + 1}`);
      text(s, metrics[i][1], { x: x + 1.05, y: y + 0.3, w: 2.6, h: 0.6, fontSize: 15, bold: true, valign: "middle" });
      text(s, metrics[i][2], { x: x + 0.3, y: y + 1.05, w: 3.3, h: 1.2, fontSize: 14, color: C.text2 });
    }
  }
  s.addNotes("A quick guide before the numbers. PSNR measures pixel accuracy, LPIPS how similar images look to a person. " +
    "Data consistency asks whether the output explains what the camera recorded. 'Blurred' is the do-nothing baseline. " +
    "Noise levels are always quoted on the 0–255 scale.");

  // 10c. PSNR vs noise (study report Figure 1) ------------------------------
  s = content("PSNR vs noise: networks sink on unseen kernels", "Results so far");
  {
    const keys = ["blurred", "wiener", "rl", "fista", "epll", "unet", "nafnet"];
    const sets = [["P1", "P1 · training kernels"], ["P2_interp", "P2-interp · unseen settings"], ["P2_extrap", "P2-extrap · unseen family"]];
    sets.forEach(([set, title], i) => {
      const x = 0.6 + i * 4.1;
      text(s, title, { x, y: 1.4, w: 3.9, h: 0.4, fontSize: 15, bold: true, align: "center" });
      const data = keys.map((key) => ({
        name: key === "blurred" ? "Blurred input" : METHOD_LABEL[key],
        labels: ["1", "5", "15", "25"],
        values: [1, 5, 15, 25].map((sig) => +(key === "blurred" ? numbers.blurred[`${set}/${sig}`] : numbers.mean[key][`${set}/${sig}`]).toFixed(2)),
      }));
      s.addChart(pres.charts.LINE, data, {
        x, y: 1.8, w: 3.9, h: 4.3, chartColors: keys.map((k) => METHOD_HEX[k]), lineSize: 2, lineDataSymbol: "circle",
        lineDataSymbolSize: 6, showLegend: false, valAxisMinVal: 15, valAxisMaxVal: 33, valAxisMajorUnit: 3,
        showCatAxisTitle: true, catAxisTitle: "noise σ (×1/255)", showValAxisTitle: i === 0, valAxisTitle: "PSNR (dB)",
        showTitle: false, ...CHART_TEXT, catAxisTitleFontSize: 12, catAxisTitleColor: HEX.dk2,
      });
    });
    legendRow(s, keys, 0.9, 6.25, 11.8);
  }
  s.addNotes("Each line is a method; higher is better; the black line is the unrestored image. On the training kernels (left) the teal M2 " +
    "solvers and the rust M1 networks sit together. On unseen kernels (middle, right) the networks fall almost onto the black baseline at low noise, " +
    "while ADMM-EPLL and FISTA-TV keep their margin. At high noise everything converges.");

  // 11. Main result chart ---------------------------------------------------
  s = content("Tied on training kernels; M2 wins under mismatch", "Results so far");
  const mm = numbers.mismatch;
  const series = [["NAFNet (M1)", "nafnet"], ["U-Net (M1)", "unet"], ["FISTA-TV (M2)", "fista"], ["ADMM-EPLL (M2)", "epll"]];
  s.addChart(pres.charts.BAR, series.map(([name, key]) => ({
    name, labels: ["P1 (training)", "P2-interp", "P2-extrap"], values: mm[key].gain.map((v) => +v.toFixed(2)),
  })), {
    x: 0.6, y: 1.4, w: 7.6, h: 5.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [M1, "E3A08F", "8FCFC6", M2], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.0", showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMaxVal: 6,
    showValAxisTitle: true, valAxisTitle: "Gain over blurred input (dB, mean σ > 0)",
    showTitle: false, ...CHART_TEXT,
  });
  const p1Naf = numbers.mismatch.nafnet.P1, p1Epll = numbers.mismatch.epll.P1;
  const stat = [
    [`${f2(p1Naf)} vs ${f2(p1Epll)}`, "dB on P1: NAFNet vs ADMM-EPLL, a tie"],
    ["+6.4 dB", "largest ADMM-EPLL win (P2-interp, σ = 1); all P2 differences p < 10⁻⁴"],
    ["½", "of NAFNet's gain lost on unseen kernels; ADMM-EPLL keeps almost all"],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.5 + i * 1.75;
    card(s, 8.55, y, 4.15, 1.5, { name: `result stat ${i + 1}` });
    text(s, stat[i][0], { x: 8.8, y: y + 0.15, w: 3.7, h: 0.6, fontSize: 28, bold: true,
      color: i === 0 ? C.text1 : i === 1 ? M2 : M1, fontFace: THEME.headFontFace });
    text(s, stat[i][1], { x: 8.8, y: y + 0.78, w: 3.7, h: 0.65, fontSize: 13, color: C.text2 });
  }
  s.addNotes("Gain over the blurred input, averaged over noise levels. On the training kernels NAFNet and ADMM-EPLL are tied. " +
    "On unseen kernels the networks lose about half their gain while the physics-aware solvers keep it. The per-sigma specialists " +
    "gain at most 0.25 dB, so the networks are missing the kernel, not the noise level. The failure is concentrated on unseen motion directions.");

  // 11b. Where M1 fails -----------------------------------------------------
  s = content("Where M1 fails, and why we trust M2", "Results so far");
  {
    const fam = numbers.family;
    const groups = [["P1/gaussian", "Gaussian (train)"], ["P1/motion", "Motion (train)"], ["P2_interp/gaussian", "Gaussian (new)"],
      ["P2_interp/motion", "Motion (new)"], ["P2_extrap/defocus", "Defocus (new)"]];
    text(s, "PSNR by kind of blur (mean over σ > 0)", { x: 0.6, y: 1.4, w: 6.6, h: 0.4, fontSize: 15, bold: true });
    s.addChart(pres.charts.BAR, ["blurred", "epll", "nafnet"].map((key) => ({
      name: key === "blurred" ? "Blurred input" : METHOD_LABEL[key], labels: groups.map((g) => g[1]),
      values: groups.map((g) => +fam[key][g[0]].toFixed(1)),
    })), {
      x: 0.6, y: 1.8, w: 6.6, h: 4.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 50,
      chartColors: [METHOD_HEX.blurred, M2, M1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
      showLegend: true, legendPos: "b", valAxisMinVal: 18, valAxisMaxVal: 30, showTitle: false, ...CHART_TEXT,
      dataLabelFontSize: 10, catAxisLabelFontSize: 11,
    });
    const res = numbers.residual_255;
    text(s, "Data consistency at σ = 1 (lower is better)", { x: 7.6, y: 1.4, w: 5.1, h: 0.4, fontSize: 15, bold: true });
    s.addChart(pres.charts.BAR, ["epll", "nafnet"].map((key) => ({
      name: METHOD_LABEL[key], labels: ["P1", "P2-interp", "P2-extrap"],
      values: ["P1", "P2_interp", "P2_extrap"].map((set) => +res[key][`${set}/1`].toFixed(2)),
    })), {
      x: 7.6, y: 1.8, w: 5.1, h: 3.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
      chartColors: [M2, M1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
      showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMaxVal: 5, showValAxisTitle: true,
      valAxisTitle: "residual RMSE (×1/255)", showTitle: false, ...CHART_TEXT,
    });
    text(s, "A consistent restoration leaves about the noise level (1). On unseen kernels NAFNet leaves 3–4×: its output, blurred again, does not match what was recorded.",
      { x: 7.6, y: 5.2, w: 5.1, h: 1.0, fontSize: 13, color: C.text2 });
    text(s, "Unseen Gaussian widths cost NAFNet 0.7 dB; unseen motion directions cost it 3.6 dB, leaving it just above the blurred input.",
      { x: 0.6, y: 6.1, w: 6.6, h: 0.7, fontSize: 13, color: C.text2 });
  }
  s.addNotes("Breaking results down by kind of blur shows where the network fails: it handles new Gaussian widths, but not new motion directions. " +
    "The right chart explains why we trust the solvers: if you blur their output again with the true kernel you get back what the camera saw. " +
    "NAFNet's output does not pass that check on unseen kernels.");

  // 12. Qualitative ---------------------------------------------------------
  s = content("Unseen motion: NAFNet does not undo the blur", "Results so far");
  const q = "P2_interp_motion_l13_a67.5_s1";
  const tiles = [["sharp", "Sharp", null], ["observed", "Observed", "27.27"], ["fista", "FISTA-TV", "33.42"],
    ["epll", "ADMM-EPLL", "34.23"], ["nafnet", "NAFNet", "26.34"]];
  for (let i = 0; i < tiles.length; i++) {
    const x = 0.6 + i * 2.47;
    s.addImage({ path: path.join(GEN, "qualitative", `${q}_${tiles[i][0]}.png`), x, y: 1.5, w: 2.3, h: 2.3,
      altText: `${tiles[i][1]} crop, P2-interp motion 13 px at 67.5 degrees, sigma 1/255` });
    text(s, tiles[i][1], { x, y: 3.9, w: 2.3, h: 0.35, fontSize: 15, bold: true, align: "center" });
    const color = tiles[i][0] === "epll" ? M2 : tiles[i][0] === "nafnet" ? M1 : C.text2;
    text(s, tiles[i][2] ? `${tiles[i][2]} dB` : "reference", { x, y: 4.25, w: 2.3, h: 0.35, fontSize: 14,
      align: "center", color, bold: !!tiles[i][2] });
  }
  card(s, 0.6, 4.95, 12.1, 1.75, { name: "qualitative note" });
  bullets(s, [
    "Motion blur of 13 px at 67.5°, an angle never seen in training; noise σ = 1/255",
    "NAFNet adds streaks and scores below the unrestored input; both physics-aware solvers remove the smear (+6⁠–⁠7 dB)",
    "Image chosen as the median case of this cell for EPLL − NAFNet, not hand-picked; PSNR on the full image",
  ], { x: 0.9, y: 5.15, w: 11.5, h: 1.45, fontSize: 14 });
  s.addNotes("This is the typical case, not the best case: the image with the median difference between ADMM-EPLL and NAFNet for this " +
    "cell. The network was trained on eight motion directions and does not generalise to a new one; the solvers use the true kernel.");

  // 12b. Other three cases ------------------------------------------------
  s = content("Three more cases, chosen without looking", "Results so far");
  {
    const Q = path.join(GEN, "qualitative");
    const cols = [["sharp", "Sharp"], ["observed", "Observed"], ["wiener", "Wiener"], ["fista", "FISTA-TV"], ["epll", "ADMM-EPLL"], ["nafnet", "NAFNet"]];
    const cases = [
      ["P1_motion_l15_a45_s5", "In distribution", "P1 motion 15 px, 45°, σ = 5", [null, 19.95, 22.34, 26.24, 28.26, 28.43]],
      ["P2_extrap_defocus_r7_s5", "Unseen family", "defocus, radius 7 px, σ = 5", [null, 17.17, 19.14, 19.49, 20.12, 18.48]],
      ["P2_interp_gaussian_sigma1.4_s15", "High noise", "Gaussian σk = 1.4, σ = 15", [null, 23.01, 23.19, 27.72, 28.84, 29.14]],
    ];
    cols.forEach(([, label], j) => {
      text(s, label, { x: 2.6 + j * 1.68, y: 1.35, w: 1.55, h: 0.3, fontSize: 13, bold: true, align: "center" });
    });
    cases.forEach(([stem, title, detail, psnr], i) => {
      const y = 1.7 + i * 1.72;
      text(s, title, { x: 0.6, y: y + 0.35, w: 1.9, h: 0.4, fontSize: 15, bold: true });
      text(s, detail, { x: 0.6, y: y + 0.75, w: 1.9, h: 0.6, fontSize: 11, color: C.text2 });
      const best = Math.max(...psnr.slice(2));
      cols.forEach(([kind, label], j) => {
        const x = 2.6 + j * 1.68;
        s.addImage({ path: path.join(Q, `${stem}_${kind}.png`), x, y, w: 1.4, h: 1.4, altText: `${title}: ${label}` });
        const v = psnr[j];
        text(s, v === null ? "PSNR (dB)" : v.toFixed(2), { x: x - 0.05, y: y + 1.4, w: 1.5, h: 0.28, fontSize: 12,
          align: "center", bold: v === best, color: v === best ? (kind === "nafnet" ? M1 : M2) : C.text2 });
      });
    });
  }
  s.addNotes("Same selection rule as before: each row is the median image for its condition, not a favourite. In distribution, NAFNet and " +
    "ADMM-EPLL look alike. On defocus, an unseen kind of blur, every method struggles and the solvers recover more edges. At high noise, " +
    "NAFNet gives the cleanest result: knowing the kernel matters least when noise dominates.");

  // 12c. Ablations: sigma oracle and K-sweep (study report Figure 3) --------
  s = content("Knowing σ barely helps; bigger priors saturate", "Results so far");
  {
    text(s, "Specialist for one σ vs. the single NAFNet", { x: 0.6, y: 1.4, w: 6.0, h: 0.4, fontSize: 15, bold: true });
    const sets = [["P1", "P1"], ["P2_interp", "P2-interp"], ["P2_extrap", "P2-extrap"]];
    s.addChart(pres.charts.BAR, sets.map(([key, label]) => ({
      name: label, labels: ["σ = 1", "σ = 5", "σ = 15", "σ = 25"],
      values: [1, 5, 15, 25].map((sig) => +numbers.specialists[`${key}/${sig}`].toFixed(2)),
    })), {
      x: 0.6, y: 1.8, w: 6.0, h: 3.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
      chartColors: [M1, "E3A08F", "F1CFC6"], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.00;-0.00",
      showLegend: true, legendPos: "b", valAxisMinVal: -0.5, valAxisMaxVal: 0.5, showValAxisTitle: true,
      valAxisTitle: "PSNR change (dB)", showTitle: false, ...CHART_TEXT, dataLabelFontSize: 10,
    });
    text(s, "At most +0.25 dB: the network is not missing the noise level. It is missing the kernel.",
      { x: 0.6, y: 5.85, w: 6.0, h: 0.8, fontSize: 14, color: C.text2 });
    text(s, "ADMM-EPLL vs. GMM size K (50 test images)", { x: 7.0, y: 1.4, w: 5.7, h: 0.4, fontSize: 15, bold: true });
    const ks = ["1", "4", "16", "64", "200"];
    s.addChart(pres.charts.LINE, sets.map(([, label], i) => ({
      name: label, labels: ks.map((k) => `K = ${k}`), values: ks.map((k) => +numbers.ksweep[k][i].toFixed(2)),
    })), {
      x: 7.0, y: 1.8, w: 5.7, h: 3.9, chartColors: [M2, "5FBDB0", "A9DCD5"], lineSize: 2.5, lineDataSymbol: "circle",
      lineDataSymbolSize: 7, showLegend: true, legendPos: "b", valAxisMinVal: 23, valAxisMaxVal: 28,
      showValAxisTitle: true, valAxisTitle: "PSNR (dB), mean σ > 0", showTitle: false, ...CHART_TEXT,
    });
    text(s, "1 → 4 components: +1.8 dB; 64 → 200: +0.02 dB. The mismatch gap stays the same for every K.",
      { x: 7.0, y: 5.85, w: 5.7, h: 0.8, fontSize: 14, color: C.text2 });
  }
  s.addNotes("Left: if we give NAFNet the noise level by training one specialist per level, it gains at most a quarter of a dB, and nothing " +
    "on unseen kernels. Right: making the classical prior richer helps at first and then flattens out, and it never changes the robustness to " +
    "unseen kernels. Robustness comes from the known blur model, not from prior capacity.");

  // 13. H3 chart -----------------------------------------------------------
  s = content("H3: M2 resists NAFNet's perturbations at σ ≥ 5", "Results so far");
  const h3 = h3lpips.h3;
  const sig = ["1", "5", "15", "25"];
  s.addChart(pres.charts.BAR, [
    { name: "NAFNet (M1)", labels: sig.map((v) => `σ = ${v}`), values: sig.map((v) => +h3.P2_interp[v].drop.m1_nafnet[1].toFixed(2)) },
    { name: "ADMM-EPLL (M2)", labels: sig.map((v) => `σ = ${v}`), values: sig.map((v) => +h3.P2_interp[v].drop.m2_admm_epll[1].toFixed(2)) },
  ], {
    x: 0.6, y: 1.4, w: 7.4, h: 5.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 70,
    chartColors: [M1, M2], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    showLegend: true, legendPos: "b", valAxisMinVal: 0, showValAxisTitle: true,
    valAxisTitle: "PSNR drop (dB), P2-interp, ε = 1% of ‖y‖", showTitle: false, ...CHART_TEXT,
  });
  const h3cards = [
    ["2–17× smaller", "drop for ADMM-EPLL at σ ≥ 5/255, in 99–100% of cells", M2],
    ["σ = 1: not stable", "perturbation ≈ noise; weakly regularised EPLL amplifies it (holds on P1, reverses on P2)", M1],
    ["Transfer attack", "η is NAFNet's worst case, not EPLL's: a white-box M2 attack is remaining work", HEX.dk2],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.5 + i * 1.75;
    card(s, 8.35, y, 4.35, 1.5, { name: `h3 card ${i + 1}` });
    text(s, h3cards[i][0], { x: 8.6, y: y + 0.15, w: 3.9, h: 0.5, fontSize: 22, bold: true, color: h3cards[i][2],
      fontFace: THEME.headFontFace });
    text(s, h3cards[i][1], { x: 8.6, y: y + 0.7, w: 3.9, h: 0.75, fontSize: 13, color: C.text2 });
  }
  s.addNotes("We optimise a small perturbation to break NAFNet, then give the same perturbed image to ADMM-EPLL. " +
    "From sigma 5 up, the solver barely notices it. At sigma 1 the perturbation is as large as the noise, and the solver, tuned for " +
    "almost no regularisation, amplifies it. Because the attack targets NAFNet, this is not a worst case for M2.");

  // 14. H2 + LPIPS ---------------------------------------------------------
  s = content("Two surprises: texture (H2) and perceptual quality", "Results so far");
  const terc = ["low", "medium", "high"];
  s.addChart(pres.charts.BAR, [{
    name: "NAFNet − ADMM-EPLL", labels: terc.map((t) => `${t} texture`),
    values: terc.map((t) => +(texture.P1[t].nafnet - texture.P1[t].epll).toFixed(2)),
  }], {
    x: 0.6, y: 1.75, w: 5.9, h: 4.3, barDir: "col", chartColors: [M1], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.00;-0.00", showLegend: false, valAxisMinVal: -0.1,
    valAxisMaxVal: 0.2, showValAxisTitle: true, valAxisTitle: "PSNR difference on P1 (dB)", showTitle: false,
    ...CHART_TEXT,
  });
  text(s, "H2 not supported: NAFNet's lead comes from smooth images (ρ = −0.54, p < 10⁻¹⁶); no texture effect under mismatch",
    { x: 0.6, y: 6.1, w: 5.9, h: 0.75, fontSize: 13, color: C.text2 });
  text(s, "Mean LPIPS (lower is better)", { x: 7.0, y: 1.5, w: 5.7, h: 0.4, fontSize: 17, bold: true });
  const lp = h3lpips.lpips;
  const lrows = [["NAFNet", "nafnet", M1], ["U-Net", "unet", "E3A08F"], ["ADMM-EPLL", "admm_epll", M2], ["FISTA-TV", "fista_tv", "8FCFC6"]];
  const header = [{ text: "", options: { fill: { color: C.text1 } } }].concat(
    ["P1", "P2-interp", "P2-extrap"].map((h) => ({ text: h, options: { bold: true, color: C.background1, fill: { color: C.text1 }, align: "center" } })));
  const rows = [header].concat(lrows.map(([label, key]) => {
    const cells = ["P1", "P2_interp", "P2_extrap"].map((set) => {
      const values = lp[set].mean, best = Math.min(...Object.values(values));
      return { text: values[key].toFixed(3), options: { align: "center", bold: values[key] === best,
        color: values[key] === best ? C.accent2 : C.text1 } };
    });
    return [{ text: label, options: { bold: true } }].concat(cells);
  }));
  s.addTable(rows, { x: 7.0, y: 2.0, w: 5.7, colW: [1.65, 1.35, 1.35, 1.35], fontSize: 14, color: C.text1,
    rowH: 0.5, border: { type: "solid", pt: 0.5, color: "D5DAE1" }, valign: "middle" });
  bullets(s, [
    "In distribution the networks are perceptually best, even where PSNR is tied",
    "Under mismatch FISTA-TV is perceptually best: EPLL wins PSNR, TV keeps sharper edges",
  ], { x: 7.0, y: 4.75, w: 5.7, h: 1.5, fontSize: 14 });
  s.addNotes("Two results went against expectations. First, the deep network's small in-distribution advantage comes from smooth " +
    "images, the opposite of H2. Second, LPIPS ranks things differently from PSNR: NAFNet is perceptually best on the training kernels, " +
    "and on unseen kernels the TV solver is best, even though ADMM-EPLL has the higher PSNR. The physics-aware arm wins under mismatch by both metrics.");

  // 14b. Inverse crime ------------------------------------------------------
  s = content("A trap we caught: the σ = 0 inverse crime", "Results so far");
  {
    const methods = ["wiener", "fista", "epll", "nafnet"];
    s.addChart(pres.charts.BAR, [["Exact σ = 0 data", 0, "B7BEC9"], ["Rounded to 8 bits", 1, AMBER]].map(([name, idx]) => ({
      name, labels: methods.map((m) => METHOD_LABEL[m]),
      values: methods.map((m) => +numbers.sigma0[`${m}/P1`][idx].toFixed(1)),
    })), {
      x: 0.6, y: 1.45, w: 7.2, h: 5.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
      chartColors: ["B7BEC9", AMBER], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
      showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMaxVal: 110, showValAxisTitle: true,
      valAxisTitle: "PSNR on P1 at σ = 0 (dB)", showTitle: false, ...CHART_TEXT,
    });
    const notes = [
      ["Why 100 dB?", "Test images were blurred with exactly the operator the solvers invert, with no noise: inversion becomes trivially exact."],
      ["The check", "Rounding the same images to 8 bits (an error of 0.0011) makes Wiener collapse to 12.6 dB; ADMM-EPLL loses 7 dB; NAFNet is unaffected."],
      ["What we did", "σ = 0 is excluded from every average and claim. Future protocol: quantize observations or add a noise floor."],
    ];
    notes.forEach(([title, body], i) => {
      const y = 1.5 + i * 1.75;
      card(s, 8.2, y, 4.5, 1.5, { name: `crime card ${i + 1}` });
      text(s, title, { x: 8.45, y: y + 0.15, w: 4.0, h: 0.4, fontSize: 16, bold: true, color: i === 2 ? M2 : AMBER });
      text(s, body, { x: 8.45, y: y + 0.55, w: 4.0, h: 0.9, fontSize: 13, color: C.text2 });
    });
  }
  s.addNotes("An 'inverse crime' is testing on data made with exactly the model you invert. Without noise the Wiener filter scores 100 dB, " +
    "which is meaningless. Rounding to 8 bits, as any real camera does, exposes it. This is why the noiseless condition is excluded from our claims.");

  // 15. Scorecard ----------------------------------------------------------
  s = content("Hypothesis scorecard so far (M1 vs M2)", "Results so far");
  const score = [
    ["H1", "Gap grows with kernel mismatch", "Supported", "tied on P1 → +2 to +6 dB for M2 on P2", HEX.accent6],
    ["H1", "Gap grows with noise", "Not supported", "M2's advantage shrinks as noise grows", M1],
    ["H1", "Gap grows with texture", "Weak", "in distribution only", AMBER],
    ["H2", "Deep prior wins on texture", "Not supported", "NAFNet's lead comes from smooth images", M1],
    ["H3", "Physics-aware arm resists perturbations", "Supported at σ ≥ 5", "2–17× smaller drop; weak or reversed at σ = 1", HEX.accent6],
  ];
  for (let i = 0; i < score.length; i++) {
    const y = 1.5 + i * 0.98;
    card(s, 0.6, y, 12.1, 0.82, { name: `score row ${i + 1}`, shadow: false });
    pill(s, score[i][0], 0.85, y + 0.24, 0.7, C.text1);
    text(s, score[i][1], { x: 1.8, y: y + 0.18, w: 4.5, h: 0.46, fontSize: 16, bold: true, valign: "middle" });
    pill(s, score[i][2], 6.4, y + 0.24, 2.1, score[i][4]);
    text(s, score[i][3], { x: 8.75, y: y + 0.18, w: 3.8, h: 0.46, fontSize: 14, color: C.text2, valign: "middle" });
  }
  text(s, "M3 is needed to test H1 and H2 as stated (deep prior with physics); the white-box M2 attack completes H3.",
    { x: 0.6, y: 6.45, w: 12.1, h: 0.4, fontSize: 13, color: C.text2 });
  s.addNotes("Kernel mismatch is the solid, headline result. The noise and texture parts of H1 and all of H2 did not hold for M1 versus M2. " +
    "H3 holds from sigma 5 up under a transfer attack. These are honest negative results, and M3 will test the hypotheses as originally stated.");

  // 16. Challenges ---------------------------------------------------------
  pres.addSection({ title: "Challenges" });
  s = content("Challenges and how we addressed them", "Challenges");
  const ch = [
    [fa.FaBug, "ADMM-EPLL barely deblurred", "Penalty schedule started too high: 21.4 → 21.9 dB even without noise. Switched to the DPIR schedule: 35.2 dB.", M2],
    [fa.FaTachometerAlt, "M2 too slow on CPU", "50–70 s per image; days for the test set. GPU port verified against NumPy: 33–50× faster.", M2],
    [fa.FaExclamationTriangle, "σ = 0 inverse crime", "Same operator for data and inversion → Wiener 100 dB. 8-bit check: 12.6 dB. σ = 0 excluded from claims.", AMBER],
    [fa.FaRulerHorizontal, "λ optimum at grid edge", "Initial ¼–4× grid too narrow. Widened to 2⁻⁵–2⁴; all optima for σ > 0 now inside.", AMBER],
    [fa.FaServer, "Shared, busy hardware", "GPU and 6 physical cores shared. One GPU job at a time, lowest priority, wait for free memory.", HEX.dk2],
    [fa.FaWrench, "M3 bugs (open)", "USRNet pads to ×4 but needs ×8, and returns the wrong output; ISTA-Net/LISTA start far below the blurred input.", M1],
  ];
  for (let i = 0; i < ch.length; i++) {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.45 + row * 2.7;
    card(s, x, y, 3.8, 2.45, { name: `challenge card ${i + 1}` });
    await iconCircle(s, ch[i][0], x + 0.3, y + 0.3, 0.6, ch[i][3], `challenge icon ${i + 1}`);
    text(s, ch[i][1], { x: x + 1.05, y: y + 0.3, w: 2.6, h: 0.65, fontSize: 15, bold: true, valign: "middle" });
    text(s, ch[i][2], { x: x + 0.3, y: y + 1.05, w: 3.3, h: 1.3, fontSize: 13, color: C.text2 });
  }
  s.addNotes("The EPLL bug was the most important: with the original schedule the solver hardly moved away from the blurred image. " +
    "The GPU port turned a multi-day CPU job into about a day. The noiseless condition turned out to be an inverse crime, which we " +
    "demonstrated with quantization and excluded. M3's bugs were found by smoke tests before any long run was wasted.");

  // 17. Remaining work + plan ----------------------------------------------
  pres.addSection({ title: "Next steps" });
  s = content("Remaining work and plan to the final evaluation", "Next steps");
  const phases = [
    ["Days 1–2", "Fix and train M3", "USRNet padding and output; skip connections for ISTA-Net/LISTA; fix 3 tests; train all three (6⁠–⁠12 h GPU)", M3],
    ["Day 3", "Evaluate all arms", "Full test set, paired tests, mismatch, texture, LPIPS, qualitative; H3 with M3", M2],
    ["Days 4–5", "Robustness", "White-box attack on M2; σ = 0 protocol fix (quantized observations); 2 more NAFNet seeds", AMBER],
    ["Days 6–7", "Write-up", "Final report and presentation with the completed scorecard", M1],
  ];
  s.addShape(pres.shapes.LINE, { x: 1.0, y: 2.35, w: 11.3, h: 0, line: { color: "C9CFD8", width: 2 } });
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.08;
    s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: 2.1, w: 0.5, h: 0.5, fill: { color: phases[i][3] },
      line: { color: C.background1, width: 2 }, objectName: `phase dot ${i + 1}` });
    text(s, String(i + 1), { x: x + 0.15, y: 2.1, w: 0.5, h: 0.5, fontSize: 14, bold: true, align: "center",
      valign: "middle", color: C.background1 });
    text(s, phases[i][0], { x: x + 0.15, y: 1.5, w: 2.6, h: 0.4, fontSize: 13, bold: true, color: phases[i][3] });
    card(s, x, 2.95, 2.85, 2.65, { name: `phase card ${i + 1}` });
    text(s, phases[i][1], { x: x + 0.25, y: 3.15, w: 2.4, h: 0.4, fontSize: 16, bold: true });
    text(s, phases[i][2], { x: x + 0.25, y: 3.6, w: 2.4, h: 1.9, fontSize: 13, color: C.text2 });
  }
  card(s, 0.6, 5.9, 12.1, 0.8, { fill: C.background1, line: "D5DAE1", shadow: false, name: "stretch note" });
  text(s, "Optional, time permitting: family-balanced kernel sampling for M1 · LPIPS on the full test set · GoPro real blur · DeepDeblur, K‑SVD",
    { x: 0.85, y: 6.0, w: 11.6, h: 0.6, fontSize: 13, color: C.text2, valign: "middle" });
  s.addNotes("M3 is the critical path: fix two bugs, add skip connections so training starts from the identity, then train. " +
    "Because the evaluator and analysis scripts already exist, M3 drops into every table. Robustness work and the write-up follow.");

  // 18. Expected outcomes --------------------------------------------------
  s = content("Expected outcomes for the final evaluation", "Next steps");
  const outcomes = [
    [fa.FaColumns, "A complete three-arm comparison", "PSNR, SSIM, LPIPS, consistency, mismatch and stability for M1, M2, M3 on identical observations, with paired statistics", M2],
    [fa.FaLayerGroup, "M3 combines M1's prior with M2's robustness", "Expect a P2 drop close to M2's (≈ −0.35 dB, not −2.9 dB) and P1 accuracy at or above NAFNet. If M3 degrades like M1, the learned prior overrides the physics", M3],
    [fa.FaShieldAlt, "A fuller stability test", "M3 more stable than M1 under transfer attack; white-box M2 attack shows whether M2's advantage survives its own worst case", AMBER],
    [fa.FaLightbulb, "A diagnosis for a follow-up method", "Physics-aware network with a locally adaptive prior (GMM-routed experts or spatially varying λ)", M1],
  ];
  for (let i = 0; i < outcomes.length; i++) {
    const y = 1.45 + i * 1.33;
    await iconCircle(s, outcomes[i][0], 0.6, y + 0.1, 0.75, outcomes[i][3], `outcome icon ${i + 1}`);
    text(s, outcomes[i][1], { x: 1.6, y: y + 0.02, w: 11.1, h: 0.4, fontSize: 17, bold: true });
    text(s, outcomes[i][2], { x: 1.6, y: y + 0.45, w: 11.1, h: 0.75, fontSize: 14, color: C.text2 });
  }
  s.addNotes("The main expected outcome is the full three-arm comparison. The key prediction is that M3 keeps M2's robustness to " +
    "unseen kernels while matching M1 in distribution; if it does not, that is itself an informative result. We also expect a fairer " +
    "stability test and a clear motivation for the follow-up method.");

  // 19. Closing ------------------------------------------------------------
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Next steps" });
  s.addText("Knowing the physics is what transfers", { placeholder: "title" });
  s.addText([
    { text: "On training kernels a 17 M-parameter network and a GMM with five tuned scalars are tied; on unseen kernels the physics-aware solver wins by up to 6.4 dB.",
      options: { breakLine: true } },
    { text: "Questions?", options: { fontSize: 24, bold: true, color: C.accent3 } },
  ], { placeholder: "body" });
  await iconCircle(s, fa.FaEye, 0.8, 0.9, 0.9, M2, "closing icon");
  s.addNotes("Summary: in distribution, the deep network and the classical solver are on par. Under kernel mismatch, using the " +
    "physics is what transfers. M3 will show whether a deep prior inside a physics-aware solver gets the best of both.");

  // 20. Appendix: glossary -------------------------------------------------
  pres.addSection({ title: "Appendix" });
  s = content("Appendix: glossary", "Appendix");
  {
    const terms = [
      ["PSF / kernel k", "the blur pattern a single point is smeared into"],
      ["P1 / P2", "training kernels / unseen kernels (interp: new settings; extrap: new family)"],
      ["σ", "noise level, on the 0–255 scale"],
      ["PSNR / SSIM", "pixel accuracy (dB) / structural similarity (0–1); higher is better"],
      ["LPIPS", "learned perceptual distance; lower is better"],
      ["Data consistency", "RMSE of re-blurred output vs. observation"],
      ["EPLL / GMM", "patch prior: a Gaussian mixture over 8×8 image patches"],
      ["TV / FISTA", "total-variation prior / accelerated proximal-gradient solver"],
      ["ADMM / HQS", "splitting methods that alternate a data step and a prior step"],
      ["Unrolled network", "a fixed number of solver iterations with learned parts (M3)"],
      ["PGD", "projected gradient ascent: finds the worst small perturbation"],
      ["Transfer attack", "perturbation optimised against one model, applied to another"],
      ["Inverse crime", "testing on data made with exactly the inverted model and no noise"],
      ["K-sweep / σ-oracle", "vary GMM components / specialist trained for one noise level"],
    ];
    const half = Math.ceil(terms.length / 2);
    [terms.slice(0, half), terms.slice(half)].forEach((chunk, c) => {
      s.addTable(chunk.map(([t, d]) => [{ text: t, options: { bold: true } }, d]), {
        x: 0.6 + c * 6.15, y: 1.45, w: 5.95, colW: [1.75, 4.2], fontSize: 13, color: C.text1, rowH: 0.7,
        border: { type: "solid", pt: 0.5, color: "D5DAE1" }, valign: "middle", fill: { color: C.background1 },
      });
    });
  }
  s.addNotes("Reference slide for questions; not presented.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log(`wrote ${OUT}`);
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
