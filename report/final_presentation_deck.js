// Final-presentation deck for the deblurring study.
//
// Build (pptxgenjs, react-icons, react, react-dom and sharp must be resolvable):
//   NODE_PATH=~/.cache/deck-tools/node_modules node report/final_presentation_deck.js
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
const OUT = path.join(ROOT, "report", "final_presentation.pptx");
const numbers = JSON.parse(fs.readFileSync(path.join(GEN, "numbers.json")));
const h3lpips = JSON.parse(fs.readFileSync(path.join(GEN, "h3_lpips.json")));
const texture = JSON.parse(fs.readFileSync(path.join(GEN, "texture.json"))).sets;
// Phase 2 numbers (learning curve, M3 on v1) from scripts/analyze_phase2.py.
const phase2 = JSON.parse(fs.readFileSync(path.join(GEN, "phase2.json")));
// Phase 3 numbers (placement ladder, kernel-swap probe, mismatch ablation) from scripts/analyze_phase3.py.
const phase3 = JSON.parse(fs.readFileSync(path.join(GEN, "phase3.json")));
const spectral = JSON.parse(fs.readFileSync(path.join(GEN, "spectral_loss.json")));
const stats3 = JSON.parse(fs.readFileSync(path.join(GEN, "phase3_stats.json")));

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
pres.title = "Final presentation: where the kernel enters a deblurring network";
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
    { text: { text: "Final presentation · Inverse-function learning for image deblurring",
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
const SETS = ["P1", "P2_interp", "P2_extrap"];

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
  s.addText("Where the kernel enters decides what transfers", { placeholder: "title" });
  s.addText([
    { text: "Classical vs. modern inverse-function learning for non-blind deblurring · final presentation", options: { breakLine: true } },
    { text: "Surya Manoj Pentakota (2026701036) · Kevin Paul Reddy Janagari (2026900006) · IIIT Hyderabad",
      options: { fontSize: 15, color: C.accent4 } },
  ], { placeholder: "body" });
  await iconCircle(s, fa.FaEye, 0.8, 0.9, 0.9, M2, "title icon");
  s.addNotes("We compared deblurring methods that differ in one thing: whether and how they use the physics of the blur. " +
    "The study ran in three phases: diagnose where data-only networks fail, test whether giving them the kernel fixes it, " +
    "and find out why the way the kernel enters matters more than whether it is given.");

  // 2. Agenda --------------------------------------------------------------
  s = content("What this talk covers", "Introduction");
  const agenda = [
    [fa.FaFlagCheckered, "Problem and protocol", "One locked protocol for three arms"],
    [fa.FaProjectDiagram, "Methods", "M1 networks, M2 solvers, M3 unrolled nets"],
    [fa.FaChartBar, "Phase 1: diagnose", "Data-only nets fail on unseen kernels"],
    [fa.FaLayerGroup, "Phase 2: add M3", "USRNet, robustness checks, real blur"],
    [fa.FaSearch, "Phase 3: explain", "Placement ladder, kernel-swap probe, mismatch"],
    [fa.FaRoute, "Conclusions", "Claims, what is new, limitations"],
  ];
  for (let i = 0; i < agenda.length; i++) {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.6 + row * 2.6;
    card(s, x, y, 3.8, 2.35, { name: `agenda card ${i + 1}` });
    await iconCircle(s, agenda[i][0], x + 0.3, y + 0.3, 0.7, i < 3 ? M2 : M1, `agenda icon ${i + 1}`);
    text(s, `${i + 1}. ${agenda[i][1]}`, { x: x + 0.3, y: y + 1.12, w: 3.3, h: 0.35, fontSize: 17, bold: true });
    text(s, agenda[i][2], { x: x + 0.3, y: y + 1.55, w: 3.3, h: 0.6, fontSize: 14, color: C.text2 });
  }
  s.addNotes("Six parts: the problem and the protocol that makes the comparison fair, the three arms of methods, then one section " +
    "per phase of the study, and the conclusions with their limits.");

  // 2b. Executive summary -------------------------------------------------
  s = content("The short version: what we found", "Introduction");
  {
    const L0 = phase3.ladder, SD0 = phase3.seeds;
    const ext = (m) => (SD0[m] ? SD0[m].mean : L0[m]).P2_extrap;
    const rows = [
      ["Data-only net vs. classical solver, training kernels", `Tied in PSNR (${f2(numbers.mismatch.nafnet.P1)} vs ${f2(numbers.mismatch.epll.P1)} dB); NAFNet better perceptually`, "Tie", AMBER],
      ["…on kernels it never saw", "Protocol v1: NAFNet loses half its gain; ADMM-EPLL wins every unseen-kernel cell, by up to 6.4 dB", "M2 wins", M2],
      ["Deep prior inside the solver (USRNet)", `Best on the synthetic grid (${f2(SD0.usrnet_full.mean.P1)} / ${f2(SD0.usrnet_full.mean.P2_extrap)} dB); real photos need boundary-aware training`, "M3 wins", M3],
      ["Just give the network the kernel?", `Worse on unseen families than no kernel: ${f2(ext("nafnet_film") - ext("nafnet"))} dB (FiLM), ${f2(ext("nafnet_concat") - ext("nafnet"))} dB (input maps)`, "Hurts", M1],
      ["Why?", "Kernel-swap probe: conditioned nets use the kernel as a lookup key; a data step makes it physics", "Mechanism", M2],
    ];
    for (let i = 0; i < rows.length; i++) {
      const y = 1.45 + i * 0.95;
      card(s, 0.6, y, 12.1, 0.8, { name: `summary row ${i + 1}`, shadow: false });
      text(s, rows[i][0], { x: 0.85, y: y + 0.12, w: 4.3, h: 0.56, fontSize: 15, bold: true, valign: "middle" });
      text(s, rows[i][1], { x: 5.3, y: y + 0.12, w: 5.6, h: 0.56, fontSize: 14, color: C.text2, valign: "middle" });
      pill(s, rows[i][2], 11.15, y + 0.23, 1.35, rows[i][3]);
    }
    card(s, 0.6, 6.25, 12.1, 0.6, { fill: C.text1, line: C.text1, shadow: false, name: "headline" });
    text(s, "Knowing the kernel is not enough: it has to enter through a data step.",
      { x: 0.85, y: 6.3, w: 11.6, h: 0.5, fontSize: 16, bold: true, color: C.background1, valign: "middle" });
  }
  s.addNotes("If you remember one slide, it is this one. On training kernels the data-only network and the classical solver tie. " +
    "On unseen kernels the solver wins. Putting a deep prior inside the solver, as USRNet does, is best on the synthetic grid; " +
    "on real photos it first needs to see realistic image boundaries in training. " +
    "But simply handing the kernel to a network as an input makes unseen blur worse, and the kernel-swap probe shows why.");

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
  s = content("What we built: three arms, three phases, one protocol", "Work completed");
  const stats = [
    ["3 × 9", "arms × methods", "2 data-only nets, 4 solvers, 3 unrolled nets, plus 4 placement variants", M1],
    ["~50", "trained networks", "seeds, ablations, specialists, learning curve, mismatch training", M3],
    ["60,000", "test cells / method", "protocol v2: 200 images × 60 kernels × 5 noise levels", AMBER],
    ["165", "unit tests", "operators, protocol, solvers, models, samplers", HEX.dk2],
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
    ["Phase 1", "M1 vs M2 on protocol v1: mismatch, σ-oracle, inverse crime, H2, H3, LPIPS"],
    ["Phase 2", "M3 trained; protocol v2 with new kernel families; learning curve, DIV2K, seeds"],
    ["Phase 2", "Real blur: Levin captures, Sun benchmark, recorded kernels"],
    ["Phase 3", "Pre-registered placement ladder on one NAFNet backbone"],
    ["Phase 3", "Kernel-swap probe, spectral-loss and linear-filter analyses"],
    ["Phase 3", "2×2 mismatch-training ablation; paired statistics on every claim"],
  ];
  for (let i = 0; i < status.length; i++) {
    const col = Math.floor(i / 3), row = i % 3;
    const x = 0.6 + col * 6.15, y = 4.15 + row * 0.82;
    pill(s, status[i][0], x, y + 0.08, 0.95, { "Phase 1": M1, "Phase 2": M3, "Phase 3": M2 }[status[i][0]]);
    text(s, status[i][1], { x: x + 1.15, y, w: 4.8, h: 0.5, fontSize: 14, valign: "middle" });
  }
  s.addNotes("Everything was scored by one shared evaluator on identical noisy observations. Phase 1 compared data-only networks with " +
    "classical solvers; Phase 2 added the unrolled arm and a harder protocol; Phase 3 explained and fixed the failures Phase 2 found.");

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
    "defocus. Physics-aware methods are given the correct kernel in every case; the data-only network has to cope without it. " +
    "This is protocol v1, used in Phase 1; the next slide shows the harder v2 protocol.");

  // 6c. Protocol v2 --------------------------------------------------------
  s = content("Protocol v2: distinct families and three unseen ones", "Work completed");
  {
    s.addImage({ path: path.join(GEN, "v2_kernels.png"), x: 0.6, y: 1.45, w: 7.6, h: 7.6 * 828 / 1592,
      altText: "Protocol v2 kernels: training, interpolation and extrapolation sets" });
    const rows = [
      ["P1 (24)", "Gaussian, anisotropic Gaussian, motion, camera shake: training families", HEX.dk2],
      ["P2-interp (24)", "Same four families, unseen parameters", M1],
      ["P2-extrap (12)", "Defocus disks, rings, ghosts: families no model trains on", M1],
      ["P2-real", "Recorded kernels (Levin et al.) blurring BSDS500 test images", AMBER],
    ];
    rows.forEach(([t, b, c], i) => {
      const y = 1.5 + i * 1.2;
      text(s, t, { x: 8.6, y, w: 4.1, h: 0.4, fontSize: 17, bold: true, color: c });
      text(s, b, { x: 8.6, y: y + 0.42, w: 4.1, h: 0.7, fontSize: 14, color: C.text2 });
    });
    text(s, "Phases 2 and 3 use v2: 60 kernels × 5 noise levels × 200 test images. Real captures: Levin (32 blurred photos), Sun (640).",
      { x: 0.6, y: 6.3, w: 12.1, h: 0.5, fontSize: 13, color: C.text2 });
  }
  s.addNotes("v1 had too few kernel families, so a single unseen family could not separate methods well. v2 has four training families " +
    "and three unseen ones with very different spectra: defocus and rings zero out whole frequency bands, ghosts barely lose any. " +
    "Real recorded kernels and real captured photos test whether the synthetic conclusions carry over.");

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
  text(s, "Trained like M1, but the network also receives the true kernel and σ", { x: 1.95, y: 5.55, w: 10.4, h: 0.4, fontSize: 16,
    bold: true, color: C.background1 });
  text(s, "P1 × all noise levels; batch 8, lr 1e-4; USRNet trained to convergence (3 seeds). Phase 3 adds placement variants on NAFNet's budget.",
    { x: 1.95, y: 5.95, w: 10.4, h: 0.6, fontSize: 13, color: C.background2 });
  s.addNotes("M3 combines the two: a closed-form physics step and a learned deep prior, iterated. It is what lets us separate " +
    "'knows the physics' from 'has a deep prior'.");

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
  pres.addSection({ title: "Phase 1 results" });
  s = content("How we measure: the metrics in plain language", "Phase 1 results");
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
  s = content("PSNR vs noise: networks sink on unseen kernels", "Phase 1 results");
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
  s = content("Tied on training kernels; M2 wins under mismatch", "Phase 1 results");
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
  s = content("Where M1 fails, and why we trust M2", "Phase 1 results");
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
  s = content("Unseen motion: NAFNet does not undo the blur", "Phase 1 results");
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
  s = content("Three more cases, chosen without looking", "Phase 1 results");
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
  s = content("Knowing σ barely helps; bigger priors saturate", "Phase 1 results");
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
  s = content("H3: M2 resists NAFNet's perturbations at σ ≥ 5", "Phase 1 results");
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
  s = content("Two surprises: texture (H2) and perceptual quality", "Phase 1 results");
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
  s = content("A trap we caught: the σ = 0 inverse crime", "Phase 1 results");
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
  s = content("Phase 1 hypothesis scorecard (M1 vs M2)", "Phase 1 results");
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
  text(s, "Phase 2 adds M3 (deep prior with physics) to test H1 and H2 as stated; Phase 3 asks why the kernel helps only in some forms.",
    { x: 0.6, y: 6.45, w: 12.1, h: 0.4, fontSize: 13, color: C.text2 });
  s.addNotes("Kernel mismatch is the solid, headline result. The noise and texture parts of H1 and all of H2 did not hold for M1 versus M2. " +
    "H3 holds from sigma 5 up under a transfer attack. These are honest negative results, and they motivated Phase 2.");


  // 15b. Phase 2: learning curve ------------------------------------------
  pres.addSection({ title: "Phase 2 results" });
  s = content("Reviewer check: NAFNet is not data-hungry", "Phase 2 results");
  {
    const pm = phase2.mean;
    const runs = [["nafnet_real50", "50 images"], ["nafnet_real100", "100 images"], ["nafnet", "200 images (Phase 1)"]];
    text(s, "NAFNet test PSNR by number of training images (mean σ > 0)", { x: 0.6, y: 1.4, w: 7.6, h: 0.4, fontSize: 15, bold: true });
    s.addChart(pres.charts.BAR, runs.map(([key, name]) => ({
      name, labels: ["P1 (training)", "P2-interp", "P2-extrap"], values: SETS.map((set) => +pm[key][set].toFixed(2)),
    })), {
      x: 0.6, y: 1.8, w: 7.6, h: 4.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
      chartColors: ["F2C9BE", "E3A08F", M1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
      showLegend: true, legendPos: "b", valAxisMinVal: 20, valAxisMaxVal: 29, showValAxisTitle: true,
      valAxisTitle: "PSNR (dB)", showTitle: false, ...CHART_TEXT,
    });
    const spread = Math.max(...runs.map(([k]) => pm[k].P1)) - Math.min(...runs.map(([k]) => pm[k].P1));
    const stat = [
      [`${f2(spread)} dB`, "P1 spread across 50, 100 and 200 images: 4× the data changes nothing", M1],
      [`${f2(pm.nafnet.p2_drop)} dB`, `P2 drop at 200 images; ${f2(pm.nafnet_real50.p2_drop)} at 50. Mismatch, not data`, M1],
      ["Gate: closed", "Flat curve: synthetic data not justified (DIV2K control pending)", AMBER],
    ];
    for (let i = 0; i < 3; i++) {
      const y = 1.5 + i * 1.75;
      card(s, 8.55, y, 4.15, 1.5, { name: `curve stat ${i + 1}` });
      text(s, stat[i][0], { x: 8.8, y: y + 0.15, w: 3.7, h: 0.6, fontSize: 28, bold: true, color: stat[i][2], fontFace: THEME.headFontFace });
      text(s, stat[i][1], { x: 8.8, y: y + 0.78, w: 3.7, h: 0.65, fontSize: 13, color: C.text2 });
    }
  }
  s.addNotes("The reviewer suggested NAFNet was data-hungry. We retrained it on nested subsets of 50 and 100 of the 200 training images, " +
    "same protocol and iterations. Test PSNR moves by less than 0.1 dB on the training kernels and the unseen-kernel drop is the same. " +
    "More data of the same kind does not help, so the go/no-go gate for generating synthetic images is closed; the DIV2K control run will confirm it.");

  // 15c. Phase 2: M3 --------------------------------------------------------
  s = content("USRNet (M3): best accuracy, M2's robustness", "Phase 2 results");
  {
    const pg = phase2.gain;
    const series = [["NAFNet (M1)", "nafnet", M1], ["ADMM-EPLL (M2)", "epll", M2], ["USRNet (M3)", "usrnet", M3],
      ["ISTA-Net (M3)", "ista_net", "A88BC0"], ["LISTA (M3)", "lista", "D3C4E0"]];
    text(s, "Gain over blurred input (dB, mean σ > 0)", { x: 0.6, y: 1.4, w: 7.6, h: 0.4, fontSize: 15, bold: true });
    s.addChart(pres.charts.BAR, series.map(([name, key]) => ({
      name, labels: ["P1 (training)", "P2-interp", "P2-extrap"], values: SETS.map((set) => +pg[key][set].toFixed(2)),
    })), {
      x: 0.6, y: 1.8, w: 7.6, h: 4.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 50,
      chartColors: series.map((x) => x[2]), showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
      showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMaxVal: 6, showTitle: false, ...CHART_TEXT,
      dataLabelFontSize: 9,
    });
    const t = phase2.tests, pm = phase2.mean;
    const stat = [
      [`+${f2(t["usrnet-epll/P1"].mean_diff_db)} dB`, `USRNet over ADMM-EPLL on P1; +${f2(t["usrnet-epll/P2_interp"].mean_diff_db)} / +${f2(t["usrnet-epll/P2_extrap"].mean_diff_db)} on P2 (Holm p < 10⁻³⁰)`, M3],
      [`${f2(pm.usrnet.p2_drop)} dB`, `USRNet P2 drop, like ADMM-EPLL (${f2(pm.epll.p2_drop)}); NAFNet ${f2(pm.nafnet.p2_drop)}`, M3],
      ["Prior matters", "ISTA-Net and LISTA are robust but weaker: light priors lose to the GMM", AMBER],
    ];
    for (let i = 0; i < 3; i++) {
      const y = 1.5 + i * 1.75;
      card(s, 8.55, y, 4.15, 1.5, { name: `m3 stat ${i + 1}` });
      text(s, stat[i][0], { x: 8.8, y: y + 0.15, w: 3.7, h: 0.6, fontSize: 28, bold: true, color: stat[i][2], fontFace: THEME.headFontFace });
      text(s, stat[i][1], { x: 8.8, y: y + 0.78, w: 3.7, h: 0.65, fontSize: 13, color: C.text2 });
    }
  }
  s.addNotes("The three unrolled networks were fixed, trained on the same 200 images and kernels, and evaluated on the same 27,000 cells. " +
    "USRNet is the best method overall: about a quarter of a dB above ADMM-EPLL on every kernel set, and its drop on unseen kernels is the " +
    "same small drop as the classical solver, not NAFNet's 2.8 dB. Using the kernel inside the network is what transfers. ISTA-Net and LISTA " +
    "also keep their accuracy on unseen kernels, but their small learned priors are weaker in absolute terms.");

  // 15d. Phase 3 ------------------------------------------------------------
  pres.addSection({ title: "Phase 3 results" });
  const L = phase3.ladder, SD = phase3.seeds, BM = phase3.benchmarks, PR = phase3.probe;
  const LU = phase3.ladder_unet || {};
  // Seed mean where seeds exist, else the single run.
  const R3 = (m) => (SD[m] ? SD[m].mean : L[m]);
  const nSeeds = (m) => (SD[m] ? SD[m].n : 1);
  const pm = (m, set) => (nSeeds(m) > 1 ? `${f2(R3(m)[set])} ± ${SD[m].std[set].toFixed(2)}` : f2(R3(m)[set]));
  const holmP = (name) => {
    const t = stats3.families.ladder.find((x) => x.name === name);
    return t ? t.p_holm : null;
  };
  const pText = (p) => (p == null ? "–" : p < 1e-3 ? `p < 10⁻${Math.floor(-Math.log10(p))}` : `p = ${p.toFixed(2)}`);
  const TABLE = { fontSize: 13, color: C.text1, border: { type: "solid", pt: 0.5, color: "D5DAE1" }, valign: "middle",
    fill: { color: C.background1 }, margin: 0.06 };
  const hdr = (cells) => cells.map((t) => ({ text: t, options: { bold: true, fill: { color: C.background2 } } }));
  const cell = (v, opts = {}) => ({ text: v == null ? "–" : (typeof v === "number" ? f2(v) : v), options: opts });

  // P3-1: framing and pre-registered predictions
  s = content("Phase 3: diagnose → explain → fix → validate", "Phase 3 results");
  {
    text(s, "Predictions were written down before any Phase 3 training (vault/Phase 3 Pre-registration.md).",
      { x: 0.6, y: 1.35, w: 12.1, h: 0.4, fontSize: 14, color: C.text2 });
    const rows = [
      ["Placement", "Unrolled NAFNet: no drop on unseen families, ≥ one-step K-NAFNet everywhere",
        `${f2(R3("nafnet_unrolled").P2_extrap)} dB extrap vs ${f2(R3("nafnet_unrolled").P1)} P1; above K-NAFNet on every set`, "Holds", M2],
      ["Placement", "One FFT data step (K-NAFNet) removes the unseen-family drop (< 0.5 dB)",
        `Drop is still ${f2(R3("knafnet").P1 - R3("knafnet").P2_extrap)} dB: one step is not enough`, "Fails", M1],
      ["Mismatch training", "USRNet gains ≥ 0.5 dB on real-kernel benchmarks, loses ≤ 0.2 dB on the grid",
        `Levin real +${f2(L.usrnet_mm.levin_real.all - L.usrnet_full.levin_real.all)}, Sun +${f2(L.usrnet_mm.sun - BM.usrnet.sun)}; grid −${f2(L.usrnet_full.P1 - L.usrnet_mm.P1)}`, "Holds", M2],
      ["Control", "On circular Sun (exact forward model) USRNet ≥ EPLL",
        `EPLL still wins (${f2(BM.epll.sun_circular)} vs ${f2(BM.usrnet.sun_circular)}): boundaries do not explain it`, "Fails", M1],
      ["Explanation", "NAFNet applies one fixed filter whatever the kernel",
        `Shared filter explains R² ${phase3.linear_filter.nafnet.shared_r2.toFixed(2)}, per-kernel ${phase3.linear_filter.nafnet.per_kernel_r2.toFixed(2)}: it adapts`, "Fails", M1],
    ];
    s.addTable([hdr(["Experiment", "Prediction", "Result", "Verdict"]),
      ...rows.map(([e, p, r, v, c]) => [cell(e, { bold: true }), cell(p), cell(r), cell(v, { bold: true, color: c, align: "center" })])],
      { x: 0.6, y: 1.85, w: 12.1, colW: [1.8, 4.6, 4.6, 1.1], rowH: 0.78, ...TABLE });
  }
  s.addNotes("Phase 2 diagnosed two failures: data-only networks do not transfer to unseen kernel families, and USRNet loses to " +
    "EPLL on provided real blur. Phase 3 tries to explain and fix them. We wrote the predictions down first; three of five failed " +
    "and are reported as failed. The ones that matter for the main claim hold.");

  // P3-2: placement ladder
  s = content("Where the kernel enters, not whether it does", "Phase 3 results");
  {
    const rungs = [["No kernel (NAFNet)", "nafnet"], ["Kernel as input maps", "nafnet_concat"], ["FiLM in every block", "nafnet_film"],
      ["FiLM + one FFT data step", "knafnet"], ["4-stage unrolled", "nafnet_unrolled"], ["USRNet (reference)", "usrnet_full"]];
    text(s, "One NAFNet backbone, data and budget; only the kernel's entry changes (PSNR, σ > 0, mean over seeds)",
      { x: 0.6, y: 1.35, w: 12.1, h: 0.4, fontSize: 14, bold: true });
    s.addChart(pres.charts.BAR, [["P1 (training families)", "P1"], ["P2-extrap (unseen families)", "P2_extrap"]].map(([name, key]) => ({
      name, labels: rungs.map((r) => r[0]), values: rungs.map((r) => +R3(r[1])[key].toFixed(2)),
    })), {
      x: 0.6, y: 1.75, w: 6.9, h: 5.0, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 45,
      chartColors: ["B7BEC9", M3], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
      showLegend: true, legendPos: "b", valAxisMinVal: 15, valAxisMaxVal: 29, catAxisOrientation: "maxMin",
      showTitle: false, ...CHART_TEXT,
    });
    const vsNaf = { nafnet_concat: "nafnet_concat - nafnet / P2_extrap", nafnet_film: "nafnet_film - nafnet / P2_extrap",
      knafnet: "knafnet - nafnet / P2_extrap", nafnet_unrolled: "nafnet_unrolled - nafnet / P2_extrap",
      usrnet_full: "nafnet_unrolled - usrnet_full / P2_extrap" };
    s.addTable([hdr(["Variant", "P2-extrap (dB)", "Seeds", "Δ vs no kernel"]),
      ...rungs.map(([name, m]) => {
        const d = R3(m).P2_extrap - R3("nafnet").P2_extrap;
        return [cell(name, { bold: true, fontSize: 12 }), cell(pm(m, "P2_extrap"), { fontSize: 12 }), cell(String(nSeeds(m)), { align: "center", fontSize: 12 }),
          cell(m === "nafnet" ? "–" : `${d >= 0 ? "+" : ""}${f2(d)}`, { bold: true, fontSize: 12, color: d < 0 ? M1 : M2 })];
      })],
      { x: 7.75, y: 1.85, w: 4.95, colW: [1.85, 1.35, 0.7, 1.05], rowH: 0.5, ...TABLE });
    const pc = holmP(vsNaf.nafnet_concat), pf = holmP(vsNaf.nafnet_film), pu = holmP(vsNaf.usrnet_full);
    bullets(s, [
      `Both conditioned variants are below no kernel on unseen families (Holm ${pText(Math.max(pc, pf))}, 200 test images)`,
      `Unrolled vs USRNet on P2-extrap: ${pText(pu)}, a tie: the placement, not USRNet's architecture, carries the gain`,
    ], { x: 7.75, y: 5.5, w: 4.95, h: 1.3, fontSize: 12, color: C.text2 });
  }
  s.addNotes("The ladder keeps everything fixed except where the kernel enters. Giving the network the kernel as input maps or FiLM " +
    "buys about 0.6 dB on training families and costs 1.2 and several dB on unseen families: worse than no kernel at all. " +
    "A single FFT data step turns the kernel into a help; four unrolled stages lose nothing on unseen families relative to USRNet and tie with it, so USRNet's " +
    "advantage is the placement, not its particular architecture. Errors are standard deviations over training seeds; tests pair test images.");

  // P3-2b: U-Net ladder (only once the second-backbone runs exist)
  if (LU.unet && LU.unet_concat && LU.unet_film && LU.kunet) {
    s = content("Same ladder, second backbone: U-Net", "Phase 3 results");
    const rungs = [["No kernel (U-Net)", "unet"], ["Kernel as input maps", "unet_concat"], ["FiLM after every block", "unet_film"],
      ["FiLM + one FFT data step", "kunet"]];
    text(s, "M1 U-Net backbone, NAFNet's training budget; only the kernel's entry changes (PSNR, σ > 0)",
      { x: 0.6, y: 1.35, w: 12.1, h: 0.4, fontSize: 14, bold: true });
    s.addChart(pres.charts.BAR, [["P1 (training families)", "P1"], ["P2-extrap (unseen families)", "P2_extrap"]].map(([name, key]) => ({
      name, labels: rungs.map((r) => r[0]), values: rungs.map((r) => +LU[r[1]][key].toFixed(2)),
    })), {
      x: 0.6, y: 1.75, w: 6.9, h: 5.0, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 45,
      chartColors: ["B7BEC9", "E3A08F"], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
      showLegend: true, legendPos: "b", valAxisMinVal: 15, valAxisMaxVal: 29, catAxisOrientation: "maxMin",
      showTitle: false, ...CHART_TEXT,
    });
    const d = (m) => LU[m].P2_extrap - LU.unet.P2_extrap;
    const dn = (m) => R3(m).P2_extrap - R3("nafnet").P2_extrap;
    s.addTable([hdr(["Δ unseen families", "U-Net", "NAFNet"]),
      ...[["Input maps", "unet_concat", "nafnet_concat"], ["FiLM", "unet_film", "nafnet_film"], ["FiLM + data step", "kunet", "knafnet"]]
        .map(([name, u, n]) => [cell(name, { bold: true }),
          cell(`${d(u) >= 0 ? "+" : ""}${f2(d(u))}`, { bold: true, color: d(u) < 0 ? M1 : M2 }),
          cell(`${dn(n) >= 0 ? "+" : ""}${f2(dn(n))}`, { color: dn(n) < 0 ? M1 : M2 })])],
      { x: 7.75, y: 1.85, w: 4.95, colW: [2.15, 1.4, 1.4], rowH: 0.55, ...TABLE });
    const same = ["concat", "film", "kunet"].every((k) => {
      const u = k === "kunet" ? "kunet" : `unet_${k}`, n = k === "kunet" ? "knafnet" : `nafnet_${k}`;
      return Math.sign(d(u)) === Math.sign(dn(n));
    });
    text(s, same ? "Every rung moves in the same direction on both backbones: the effect is not specific to NAFNet."
      : "The rungs do not all move the same way on U-Net: the effect depends on the backbone (see table).",
      { x: 7.75, y: 4.3, w: 4.95, h: 1.2, fontSize: 14, bold: true, color: same ? M2 : M1 });
    text(s, "One seed per U-Net variant.", { x: 7.75, y: 5.6, w: 4.95, h: 0.4, fontSize: 12, color: C.text2 });
    s.addNotes("A reviewer's first question is whether the ladder is a NAFNet quirk. We repeated it on the plain U-Net from M1, " +
      "with the same three ways of adding the kernel and the same training budget. The table compares the change on unseen families " +
      "for each rung on both backbones.");
  }

  // P3-3: kernel-swap probe
  s = content("Kernel-swap probe: the kernel becomes a lookup key", "Phase 3 results");
  {
    const methods = [["Wiener (physics)", "wiener"], ["NAFNet (no kernel)", "nafnet"], ["Input maps", "nafnet_concat"],
      ["FiLM every block", "nafnet_film"], ["K-NAFNet (1 step)", "knafnet"], ["Unrolled NAFNet", "nafnet_unrolled"], ["USRNet", "usrnet_full"]];
    const e = (m, c) => PR[m][`P2_extrap/${c}`];
    const gain = (m) => e(m, "true") - Math.max(e(m, "nearest"), e(m, "random"), e(m, "delta"));
    text(s, "Unseen families: blur with the true kernel, then tell the method a true, nearest-training or no-blur kernel",
      { x: 0.6, y: 1.35, w: 12.1, h: 0.4, fontSize: 14, bold: true });
    s.addTable([hdr(["Method", "True", "Nearest training", "No blur", "Worth of true kernel"]),
      ...methods.map(([name, m]) => {
        const g = gain(m);
        return [cell(name, { bold: true }), cell(e(m, "true")), cell(e(m, "nearest")), cell(e(m, "delta")),
          cell((g >= 0 ? "+" : "") + f2(g) + " dB", { bold: true, color: g < 0 ? M1 : g > 1 ? M2 : HEX.dk2 })];
      })],
      { x: 0.6, y: 1.85, w: 8.2, colW: [2.2, 1.4, 1.6, 1.5, 1.5], rowH: 0.5, ...TABLE });
    card(s, 9.1, 1.85, 3.6, 4.2, { name: "probe takeaway" });
    text(s, "Reading it", { x: 9.35, y: 2.0, w: 3.1, h: 0.4, fontSize: 16, bold: true, color: AMBER });
    bullets(s, [
      `FiLM scores higher with a wrong kernel than the true one (${f2(e("nafnet_film", "true"))} vs ${f2(e("nafnet_film", "delta"))}): no physics model can do that`,
      "In distribution the same nets lose ~8 dB with a random kernel: they rely on it as a key",
      "Through a data step the true kernel is worth +3 to +7 dB, like Wiener",
    ], { x: 9.35, y: 2.45, w: 3.15, h: 3.5, fontSize: 13, color: C.text2 });
    text(s, "Worth of the true kernel = true − best wrong kernel (nearest, random or delta). 50 test images × 12 unseen kernels × σ ∈ {1, 5, 15, 25}.",
      { x: 0.6, y: 6.55, w: 12.1, h: 0.35, fontSize: 11, color: C.text2 });
  }
  s.addNotes("The observations are always made with the true kernel; only the kernel we tell the method changes. Wiener behaves like " +
    "physics: any wrong kernel costs 3 to 5 dB. The FiLM network on unseen families does better with no-blur or the nearest training " +
    "kernel than with the truth, so it uses the kernel as a key into learned mappings. A data step restores physical use of the kernel.");

  // P3-3b: spectral loss
  s = content("The kernel helps where the blur keeps information", "Phase 3 results");
  {
    const ek = spectral.extrap_kernels;
    const series = [["FiLM", "nafnet_film", M1], ["K-NAFNet (1 step)", "knafnet", AMBER], ["Unrolled NAFNet", "nafnet_unrolled", M3]];
    text(s, "Each point is one unseen kernel: PSNR gain over no-kernel NAFNet vs. how much of the spectrum the blur destroys",
      { x: 0.6, y: 1.35, w: 12.1, h: 0.4, fontSize: 14, bold: true });
    s.addChart(pres.charts.SCATTER, [
      { name: "X", values: ek.map((k) => +(100 * k.loss).toFixed(1)) },
      ...series.map(([name, key]) => ({ name, values: ek.map((k) => +k[`gain_${key}`].toFixed(2)) })),
    ], {
      x: 0.6, y: 1.75, w: 7.6, h: 5.0, chartColors: series.map((x) => x[2]), lineSize: 0, lineDataSymbol: "circle",
      lineDataSymbolSize: 9, showLegend: true, legendPos: "b", showValAxisTitle: true, valAxisTitle: "gain over NAFNet (dB)",
      showCatAxisTitle: true, catAxisTitle: "spectrum with |K| < 0.1 (%)", catAxisTitleColor: HEX.dk2, catAxisTitleFontSize: 12,
      showTitle: false, ...CHART_TEXT,
    });
    const rho = spectral.spearman_gain_vs_loss;
    s.addTable([hdr(["Method", "Spearman ρ", "p"]),
      ...[["Input maps", "nafnet_concat"], ["FiLM", "nafnet_film"], ["K-NAFNet", "knafnet"], ["Unrolled", "nafnet_unrolled"], ["USRNet", "usrnet_full"]]
        .map(([name, m]) => [cell(name, { bold: true }), cell(rho[m].rho.toFixed(2), { align: "center" }),
          cell(rho[m].p < 1e-3 ? rho[m].p.toExponential(0) : rho[m].p.toFixed(2), { align: "center", color: C.text2 })])],
      { x: 8.55, y: 1.85, w: 4.15, colW: [1.75, 1.2, 1.2], rowH: 0.45, ...TABLE });
    text(s, "Ghosts destroy almost nothing and gain most; wide defocus and rings zero out bands that no kernel can bring back.",
      { x: 8.55, y: 4.7, w: 4.15, h: 1.0, fontSize: 13, color: C.text2 });
    text(s, `Caveat: spectral loss also grows with kernel size (ρ = ${spectral.spearman_loss_vs_kernel_size.rho.toFixed(2)}); n = 12 kernels.`,
      { x: 8.55, y: 5.75, w: 4.15, h: 0.8, fontSize: 12, color: C.text2 });
  }
  s.addNotes("This explains the size of the gain, kernel by kernel. Where the blur keeps most frequencies, for example ghosts, " +
    "a data step recovers 8 to 12 dB. Where the blur zeroes out frequency bands, the kernel cannot bring them back and every method " +
    "relies on its prior. Spectral loss is correlated with kernel size, so we do not separate the two.");

  // P3-4: real blur and mismatch training
  s = content("Real captures: simulate the boundary too", "Phase 3 results");
  {
    const has66 = L.usrnet_mm_boundary66k && L.usrnet_mm_boundary66k.levin_real;
    const rows = [["USRNet (standard)", L.usrnet_full, BM.usrnet.sun, BM.usrnet.sun_circular],
      ["+ realistic boundaries", L.usrnet_mm_boundary, L.usrnet_mm_boundary.sun, L.usrnet_mm_boundary.sun_circular],
      ...(has66 ? [["  same, stopped at 66k", L.usrnet_mm_boundary66k, L.usrnet_mm_boundary66k.sun, L.usrnet_mm_boundary66k.sun_circular]] : []),
      ["+ kernel noise", L.usrnet_mm_knoise, L.usrnet_mm_knoise.sun, L.usrnet_mm_knoise.sun_circular],
      ["+ both", L.usrnet_mm, L.usrnet_mm.sun, L.usrnet_mm.sun_circular]];
    text(s, "2×2 ablation of mismatch-aware training (half of the batches): PSNR in dB", { x: 0.6, y: 1.35, w: 12.1, h: 0.4, fontSize: 14, bold: true });
    s.addTable([hdr(["USRNet trained with", "Grid P1", "Grid extrap", "Levin real", "Levin, 2 largest", "Sun", "Sun circular"]),
      ...rows.map(([name, g, sun, sc], i) => [cell(name, { bold: true }), cell(g.P1), cell(g.P2_extrap), cell(g.levin_real.all),
        cell(g.levin_real.big2, { bold: i > 0, color: name.includes("boundar") || name.includes("66k") ? M2 : C.text1 }), cell(sun), cell(sc)]),
      [cell("EPLL / FISTA (classical)", { color: C.text2 }), cell("–"), cell("–"),
        cell(`${f2(BM.epll.levin_real)} / ${f2(BM.fista.levin_real)}`, { color: C.text2 }), cell("–"),
        cell(`${f2(BM.epll.sun)} / ${f2(BM.fista.sun)}`, { color: C.text2 }),
        cell(`${f2(BM.epll.sun_circular)} / ${f2(BM.fista.sun_circular)}`, { color: C.text2 })]],
      { x: 0.6, y: 1.85, w: 12.1, colW: [2.6, 1.25, 1.35, 1.6, 2.3, 1.4, 1.6], rowH: has66 ? 0.46 : 0.55, ...TABLE });
    const lengthNote = has66
      ? `Stopped at the same 66k iterations as the other runs, boundaries still give ${f2(L.usrnet_mm_boundary66k.levin_real.big2)} dB on the largest kernels: the gain is not extra training`
      : "Caveat: early stopping let the boundary run train longer (120k vs 66k iterations)";
    bullets(s, [
      `Realistic boundaries alone fix the collapse on Levin's largest kernels (${f2(L.usrnet_full.levin_real.big2)} → ${f2(L.usrnet_mm_boundary.levin_real.big2)} dB) at no cost on the grid`,
      "Kernel noise adds a smaller gain and costs ~0.1 dB in distribution; both together give the best Levin real score",
      `Sun gains equally with circular blur, so its gap to EPLL is not about boundaries (Sun is itself synthetic). ${lengthNote}`,
    ], { x: 0.6, y: has66 ? 5.45 : 5.35, w: 12.1, h: 1.5, fontSize: 13, color: C.text2 });
  }
  s.addNotes("Nan and Ji (CVPR 2020) already train with kernel noise and edge-tapering together. Our ablation separates them: " +
    "simulating realistic image boundaries is what fixes real captures with large kernels, and it costs nothing in distribution. " +
    "With it, the learned solver beats both classical solvers on Levin's real captures. This part is mostly known, and we cite it as such.");

  // P3-5: novelty
  s = content("What is new here, and what is not", "Phase 3 results");
  {
    const cols = [
      ["Likely new (not found in our review)", M2, [
        "Kernel conditioning (input maps, FiLM) makes unseen-family deblurring worse than no kernel",
        "Kernel-swap probe: the true kernel can score below a wrong one, so the net uses it as a key",
        "Controlled placement ladder on one backbone; unrolling with a generic prior ties with USRNet",
        "Worth of the kernel tracks its spectral loss (Spearman ρ ≈ −0.9 to −0.99)",
        "Held-out-family protocol (defocus, ring, ghost) with paired statistics",
      ]],
      ["Known (cited, not claimed)", HEX.dk2, [
        "Kernel maps / SFT / FiLM conditioning: SRMD 2018, SFTMD 2019, Kaufman & Fattal 2020",
        "Data step and unrolling: USRNet 2020, DPIR 2021, DWDN 2020",
        "End-to-end nets transfer worse to new blur: Chen et al. IJCV 2022, Blur2Blur 2024",
        "Kernel-noise + edgetaper training: Nan & Ji, CVPR 2020",
        "FFT solvers fail on Levin's largest kernels: Eboli et al., ECCV 2020",
      ]],
    ];
    cols.forEach(([title, color, items], i) => {
      const x = 0.6 + i * 6.15;
      card(s, x, 1.45, 5.95, 4.95, { name: `novelty card ${i + 1}` });
      text(s, title, { x: x + 0.25, y: 1.6, w: 5.5, h: 0.45, fontSize: 16, bold: true, color });
      bullets(s, items, { x: x + 0.25, y: 2.15, w: 5.45, h: 4.15, fontSize: 13, color: C.text2 });
    });
  }
  s.addNotes("The mechanisms are all known; what is new is the controlled measurement and the counter-intuitive result that " +
    "conditioning hurts out of distribution, plus the probe that explains why. Absence in our review is not proof, and we say so.");

  // 16. Challenges ---------------------------------------------------------
  pres.addSection({ title: "Challenges" });
  s = content("Challenges and how we addressed them", "Challenges");
  const ch = [
    [fa.FaBug, "ADMM-EPLL barely deblurred", "Penalty schedule started too high: 21.4 → 21.9 dB even without noise. Switched to the DPIR schedule: 35.2 dB.", M2],
    [fa.FaTachometerAlt, "M2 too slow on CPU", "50–70 s per image; days for the test set. GPU port verified against NumPy: 33–50× faster.", M2],
    [fa.FaExclamationTriangle, "σ = 0 inverse crime", "Same operator for data and inversion → Wiener 100 dB. 8-bit check: 12.6 dB. σ = 0 excluded from claims.", AMBER],
    [fa.FaRulerHorizontal, "Real blur is not circular", "FFT solvers assume wrap-around borders; real photos do not. Edgetaper at test time, linear blur in mismatch training.", AMBER],
    [fa.FaServer, "Shared, busy hardware", "GPU and 6 physical cores shared. One GPU job at a time, lowest priority, chained job queues.", HEX.dk2],
    [fa.FaWrench, "M3 bugs (fixed)", "USRNet now pads to ×8 and returns the prior output; ISTA-Net/LISTA phases are residual and start as gradient steps.", M3],
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
    "demonstrated with quantization and excluded. Real photos broke the circular-boundary assumption, which shaped the mismatch experiments.");

  // 17. Conclusions --------------------------------------------------------
  pres.addSection({ title: "Conclusions" });
  s = content("Conclusions", "Conclusions");
  {
    const ext = (m) => R3(m).P2_extrap;
    const claims = [
      [fa.FaColumns, "Physics transfers, data alone does not", `Data-only nets tie classical solvers on training kernels and lose on unseen ones; USRNet is best on the synthetic grid (${f2(R3("usrnet_full").P1)} / ${f2(R3("usrnet_full").P2_extrap)} dB), not on real photos without boundary-aware training`, M2],
      [fa.FaLayerGroup, "Placement, not presence, of the kernel", `Input maps ${f2(ext("nafnet_concat") - ext("nafnet"))} dB and FiLM ${f2(ext("nafnet_film") - ext("nafnet"))} dB on unseen families; one data step +${f2(ext("knafnet") - ext("nafnet"))}; unrolling +${f2(ext("nafnet_unrolled") - ext("nafnet"))}, tied with USRNet`, M3],
      [fa.FaSearch, "Mechanism: a lookup key vs. a forward model", "Conditioned nets can score higher with a wrong kernel; through a data step the true kernel is worth +3 to +7 dB, like Wiener", AMBER],
      [fa.FaLightbulb, "For real captures, simulate realistic boundaries", "Boundary-aware training fixes Levin's largest kernels at no grid cost (known in spirit: Nan & Ji 2020)", M1],
    ];
    for (let i = 0; i < claims.length; i++) {
      const y = 1.45 + i * 1.33;
      await iconCircle(s, claims[i][0], 0.6, y + 0.1, 0.75, claims[i][3], `conclusion icon ${i + 1}`);
      text(s, claims[i][1], { x: 1.6, y: y + 0.02, w: 11.1, h: 0.4, fontSize: 17, bold: true });
      text(s, claims[i][2], { x: 1.6, y: y + 0.45, w: 11.1, h: 0.75, fontSize: 14, color: C.text2 });
    }
  }
  s.addNotes("Four conclusions. Physics-aware methods transfer to unseen blur and data-only ones do not. Within networks, how the kernel " +
    "enters decides the outcome: as a conditioning input it hurts, through a data step it helps. The probe shows why. And on real photos " +
    "a learned solver also needs realistic boundaries in training.");

  // 18. Limitations ---------------------------------------------------------
  s = content("Limitations and future work", "Conclusions");
  {
    const lim = [
      "Our own reimplementations, trained on grayscale BSDS500 (200 images); no official pretrained weights",
      LU.kunet ? "Two backbones (NAFNet, U-Net); the U-Net ladder has one seed per variant" : "One backbone for the placement ladder (NAFNet)",
      `Seeds: ${nSeeds("nafnet_concat")} for input maps, ${nSeeds("nafnet_film")} for FiLM, ${nSeeds("knafnet")} for K-NAFNet, ${nSeeds("nafnet_unrolled")} for the unrolled net`,
      "Spectral loss and kernel size are correlated over only 12 unseen kernels",
      "Non-blind only: the true kernel is given; H3 uses a transfer attack, not a white-box attack on M2",
    ];
    const fut = [
      "Blind deblurring: does an estimated kernel behave like the 'nearest' kernel in the probe?",
      "Leave-one-family-out training to map which families transfer to which",
      "Colour images and larger training sets (DIV2K) for the ladder",
      "Official USRNet / DPIR weights as an external reference point",
    ];
    [["Limitations", M1, lim], ["Future work", M2, fut]].forEach(([title, color, items], i) => {
      const x = 0.6 + i * 6.15;
      card(s, x, 1.45, 5.95, 5.0, { name: `${title} card` });
      text(s, title, { x: x + 0.25, y: 1.6, w: 5.5, h: 0.45, fontSize: 17, bold: true, color });
      bullets(s, items, { x: x + 0.25, y: 2.15, w: 5.45, h: 4.2, fontSize: 14, color: C.text2 });
    });
  }
  s.addNotes("These are the limits we know of. The most interesting next step is blind deblurring: an estimated kernel is a slightly " +
    "wrong kernel, which is exactly the situation the kernel-swap probe simulates.");

  // 19. Closing ------------------------------------------------------------
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusions" });
  s.addText("Where the kernel enters decides what transfers", { placeholder: "title" });
  s.addText([
    { text: "Feeding the kernel to a network makes it a lookup key that fails on unseen blur; only a data-consistency step makes the network use it as physics.",
      options: { breakLine: true } },
    { text: "Thank you · Questions?", options: { fontSize: 24, bold: true, color: C.accent3 } },
  ], { placeholder: "body" });
  await iconCircle(s, fa.FaEye, 0.8, 0.9, 0.9, M2, "closing icon");
  s.addNotes("Summary: knowing the kernel is not enough; it has to enter through the physics. On real captures, the learned solver " +
    "also needs to see realistic boundaries in training.");

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
