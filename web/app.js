"use strict";

/* =====================================================================
   Reference data: the project's own, measured results
   ===================================================================== */
const GRADES = [
  { name: "Healthy", desc: "No radiographic features of osteoarthritis." },
  { name: "Doubtful", desc: "Doubtful joint space narrowing with possible osteophytic lipping." },
  { name: "Minimal", desc: "Definite osteophytes and possible joint space narrowing." },
  { name: "Moderate", desc: "Multiple osteophytes, definite joint space narrowing and mild sclerosis." },
  { name: "Severe", desc: "Large osteophytes, marked joint space narrowing and severe sclerosis." },
];

const TEST = {
  metrics: [
    { label: "Balanced accuracy", value: 0.644, ci: [0.621, 0.666], note: "Average recall across the five grades. The target metric.", primary: true },
    { label: "Quadratic weighted kappa", value: 0.753, ci: [0.730, 0.775], note: "Agreement with the true grade, penalizing distant errors more." },
    { label: "Accuracy", value: 0.542, note: "Share of X-rays graded exactly right." },
    { label: "Macro F1", value: 0.604, note: "Unweighted mean of per-grade F1 scores." },
  ],
  perClass: [ // precision, recall, f1, support
    [0.803, 0.440, 0.568, 639], [0.237, 0.436, 0.307, 296], [0.589, 0.579, 0.584, 447],
    [0.713, 0.803, 0.755, 223], [0.690, 0.961, 0.803, 51],
  ],
  confusion: [
    [281, 296, 57, 5, 0], [49, 129, 103, 14, 1], [20, 114, 259, 51, 3],
    [0, 5, 21, 179, 18], [0, 0, 0, 2, 49],
  ],
};
const SPLITS = {
  Training: [2286, 1046, 1516, 757, 173],
  Validation: [328, 153, 212, 106, 27],
  Test: [639, 296, 447, 223, 51],
};
const EXPERIMENTS = [
  ["Always predict grade 0", "Reference floor", 0.200, 0.000],
  ["Baseline CNN", "Four convolution blocks, no augmentation", 0.591, 0.556],
  ["Augmentation", "Flips, rotations, shifts, zoom, brightness", 0.613, 0.721],
  ["Class weights", "Square-root inverse-frequency weights", 0.603, 0.723],
  ["Wider network", "1.5 times the filters", 0.615, 0.713],
  ["Ordinal output", "CORAL cumulative head", 0.555, 0.725],
  ["No preprocessing", "Raw X-rays, same model", 0.589, 0.693],
  ["Ensemble with rarity correction", "Final pipeline", 0.647, 0.712, true],
];

/* =====================================================================
   Helpers
   ===================================================================== */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p) => (p > 0 && p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`);
const gradeVar = (g) => `var(--g${g})`;
const fmtDate = (iso) => new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
const localDay = (iso) => {
  const d = new Date(iso);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};
const newId = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`);

function badge(g) {
  return `<span class="badge" style="--gc:${gradeVar(g)}">Grade ${g}<span class="badge-name">${GRADES[g].name}</span></span>`;
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 3200);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("This file isn't a readable image. Upload a PNG or JPG."));
    img.src = src;
  });
}

async function toJpeg(src, maxSide, quality = 0.85) {
  const img = await loadImage(src);
  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
  canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", quality);
}

/* ---------- In-browser engine: models load once, then everything runs locally ---------- */
const Engine = {
  cfg: null,
  members: null,
  loading: null,
  load() {
    if (!this.loading) {
      this.loading = (async () => {
        if (!window.tf) throw new Error("TensorFlow.js couldn't load. Check your internet connection and refresh the page.");
        const res = await fetch("models/pipeline.json");
        if (!res.ok) throw new Error("The model configuration couldn't be downloaded. Refresh the page to try again.");
        const cfg = await res.json();
        await tf.ready();
        const members = await Promise.all(cfg.members.map((spec) => KneeModel.loadMember(tf, "models/", spec)));
        KneeModel.analyze(tf, members, cfg, new Uint8Array(224 * 224), false);   // warm-up compiles GPU shaders
        this.cfg = cfg;
        this.members = members;
        return this;
      })();
      this.loading.catch(() => { this.loading = null; });
    }
    return this.loading;
  },
};

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

function pixelsToDataUrl(rgba) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  canvas.getContext("2d").putImageData(new ImageData(rgba, 224, 224), 0, 0);
  return canvas.toDataURL("image/png");
}

function grayToDataUrl(gray) {
  const rgba = new Uint8ClampedArray(224 * 224 * 4);
  for (let i = 0; i < gray.length; i++) { rgba.fill(gray[i], i * 4, i * 4 + 3); rgba[i * 4 + 3] = 255; }
  return pixelsToDataUrl(rgba);
}

/* =====================================================================
   Persistent history: IndexedDB in this browser (survives refreshes)
   ===================================================================== */
const DB = {
  db: null,
  open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open("kneegrade", 1);
      req.onupgradeneeded = () => {
        const store = req.result.createObjectStore("analyses", { keyPath: "id" });
        store.createIndex("createdAt", "createdAt");
      };
      req.onsuccess = () => { this.db = req.result; resolve(); };
      req.onerror = () => reject(req.error);
    });
  },
  store(mode) { return this.db.transaction("analyses", mode).objectStore("analyses"); },
  wrap(req) { return new Promise((resolve, reject) => { req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error); }); },
  async all() {
    if (!this.db) return [];
    const rows = await this.wrap(this.store("readonly").getAll());
    return rows.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  },
  get(id) { return this.db ? this.wrap(this.store("readonly").get(id)) : Promise.resolve(null); },
  put(rec) { return this.db ? this.wrap(this.store("readwrite").put(rec)) : Promise.reject(new Error("Storage unavailable")); },
  del(id) { return this.wrap(this.store("readwrite").delete(id)); },
  clear() { return this.wrap(this.store("readwrite").clear()); },
};

/* =====================================================================
   Sidebar: click to navigate, Ctrl+B (or Cmd+B) to hide and show
   ===================================================================== */
const Sidebar = {
  mobile: () => window.matchMedia("(max-width: 900px)").matches,
  isOpen() {
    const app = $("#app");
    return this.mobile() ? app.classList.contains("nav-open") : !app.classList.contains("nav-collapsed");
  },
  set(open) {
    const app = $("#app");
    if (this.mobile()) {
      app.classList.toggle("nav-open", open);
    } else {
      app.classList.toggle("nav-collapsed", !open);
      localStorage.setItem("kneegrade.sidebar", open ? "open" : "closed");
    }
    $("#sidebar-toggle").setAttribute("aria-expanded", String(open));
  },
  toggle() { this.set(!this.isOpen()); },
  init() {
    if (!this.mobile() && localStorage.getItem("kneegrade.sidebar") === "closed") this.set(false);
    if (this.mobile()) $("#sidebar-toggle").setAttribute("aria-expanded", "false");
    $("#sidebar-toggle").addEventListener("click", () => this.toggle());
    $("#backdrop").addEventListener("click", () => this.set(false));
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "b") {
        e.preventDefault();
        this.toggle();
      }
    });
  },
};

/* =====================================================================
   Router: one page per hash
   ===================================================================== */
const PAGES = {
  dashboard: { title: "Dashboard", render: renderDashboard },
  analyze: { title: "New analysis", render: () => {} },
  history: { title: "History", render: renderHistory },
  insights: { title: "Model insights", render: renderInsights },
  about: { title: "About", render: renderAbout },
};

function route() {
  let page = location.hash.replace(/^#\/?/, "").split("?")[0];
  if (!PAGES[page]) page = "dashboard";
  $$(".view").forEach((v) => { v.hidden = v.dataset.view !== page; });
  $$(".nav-item").forEach((a) => {
    const on = a.dataset.page === page;
    a.classList.toggle("active", on);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  $("#page-title").textContent = PAGES[page].title;
  document.title = `${PAGES[page].title} | KneeGrade`;
  $(".topbar-cta").hidden = page === "analyze";
  if (Sidebar.mobile()) Sidebar.set(false);
  window.scrollTo(0, 0);
  PAGES[page].render();
}

/* =====================================================================
   Shared result view (New analysis page and the History detail modal)
   ===================================================================== */
function severityScale(probs, grade) {
  return `<div class="scale" role="img" aria-label="Model probability for each grade">${probs.map((p, g) => `
    <div class="scale-col${g === grade ? " is-pred" : ""}" style="--gc:${gradeVar(g)}">
      <div class="scale-track"><div class="scale-fill" style="height:${Math.max(p * 100, 1.5)}%"></div></div>
      <div class="scale-pct">${pct(p)}</div>
      <div class="scale-label"><b>${g}</b> ${GRADES[g].name}</div>
    </div>`).join("")}</div>`;
}

function renderResult(el, rec) {
  const g = GRADES[rec.grade];
  const hasHeat = Boolean(rec.heatmap);
  const adjusted = rec.grade !== rec.topGrade
    ? `<p class="note">The tallest bar is grade ${rec.topGrade} (${GRADES[rec.topGrade].name}). The final grade corrects for how rare grade ${rec.grade} is in the training data, which improves performance across all grades. <a href="#/about">How grading works</a></p>`
    : "";
  const inverted = rec.invertedCorrected
    ? `<p class="note quiet">This X-ray appeared inverted (dark bone on a light background), so it was flipped back before grading.</p>`
    : "";
  const share = hasHeat && rec.jointShare != null ? ` ${Math.round(rec.jointShare * 100)}% of the heatmap falls on the joint region.` : "";

  el.innerHTML = `
    <div class="result-head" style="--gc:${gradeVar(rec.grade)}">
      <div class="result-grade"><span class="result-num">${rec.grade}</span><span class="result-kl">KL grade</span></div>
      <div><h3>${g.name}</h3><p>${g.desc}</p></div>
    </div>
    <h4 class="sub">Model probability for each grade</h4>
    ${severityScale(rec.probabilities, rec.grade)}
    ${adjusted}${inverted}
    <div class="viewer">
      <div class="viewer-bar">
        <div class="seg" role="tablist" aria-label="Image shown">
          <button type="button" role="tab" data-show="pre" aria-selected="true">Preprocessed</button>
          <button type="button" role="tab" data-show="orig" aria-selected="false">Original</button>
        </div>
        ${hasHeat
          ? `<label class="opacity">Heatmap<input type="range" min="0" max="100" value="55" data-opacity aria-label="Heatmap opacity"></label>`
          : `<span class="viewer-hint">No heatmap for this analysis</span>`}
      </div>
      <div class="viewer-stage">
        <img class="v-base" src="${rec.preprocessed}" alt="Preprocessed X-ray">
        ${hasHeat ? `<img class="v-heat" src="${rec.heatmap}" alt="" style="opacity:.55">` : ""}
      </div>
      ${hasHeat ? `<p class="viewer-legend"><span class="ramp" aria-hidden="true"></span><span>Low to high influence on the predicted grade.${share}</span></p>` : ""}
    </div>
    <dl class="meta">
      <div><dt>File</dt><dd>${esc(rec.fileName)}</dd></div>
      <div><dt>Analyzed</dt><dd>${fmtDate(rec.createdAt)}</dd></div>
      <div><dt>Model probability of this grade</dt><dd>${pct(rec.probabilities[rec.grade])}</dd></div>
    </dl>`;

  const base = $(".v-base", el);
  const heat = $(".v-heat", el);
  const slider = $("[data-opacity]", el);
  $$(".seg button", el).forEach((btn) => btn.addEventListener("click", () => {
    const original = btn.dataset.show === "orig";
    $$(".seg button", el).forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
    base.src = original ? (rec.original || rec.preprocessed) : rec.preprocessed;
    base.alt = original ? "Original X-ray" : "Preprocessed X-ray";
    if (heat) heat.hidden = original;
    if (slider) slider.disabled = original;
  }));
  if (slider) slider.addEventListener("input", () => { heat.style.opacity = slider.value / 100; });
}

/* =====================================================================
   New analysis
   ===================================================================== */
const Analysis = { file: null, url: null, busy: false };

function showError(message) { const el = $("#analyze-error"); el.textContent = message; el.hidden = false; }
function hideError() { $("#analyze-error").hidden = true; }

function resetResult() {
  $("#result").innerHTML = `
    <div class="result-empty"><div>
      <div class="scale-ghost">${[0, 1, 2, 3, 4].map((g) => `<span style="--gc:${gradeVar(g)}"></span>`).join("")}</div>
      <strong>Your result will appear here</strong>
      <span>Choose an X-ray, then select Analyze X-ray.</span>
    </div></div>`;
}

function selectFile(file) {
  if (!file) return;
  if (location.hash !== "#/analyze") location.hash = "#/analyze";
  hideError();
  if (!/^image\/(png|jpeg)$/.test(file.type)) { showError("Unsupported file type. Upload a PNG or JPG image."); return; }
  if (file.size > 10 * 1024 * 1024) { showError("This file is larger than 10 MB. Upload a smaller image."); return; }
  if (Analysis.url) URL.revokeObjectURL(Analysis.url);
  Analysis.file = file;
  Analysis.url = URL.createObjectURL(file);
  $("#preview-img").src = Analysis.url;
  $("#preview-name").textContent = file.name;
  $("#preview").hidden = false;
  $("#analyze-drop").hidden = true;
  $("#analyze-btn").disabled = false;
  $("#stepper").hidden = true;
  resetResult();
}

function setStep(name, state) {
  const li = $(`#stepper [data-step="${name}"]`);
  if (li) li.dataset.state = state;
}

async function runAnalysis() {
  if (!Analysis.file || Analysis.busy) return;
  Analysis.busy = true;
  hideError();
  const btn = $("#analyze-btn");
  btn.disabled = true;
  btn.textContent = "Analyzing";
  $$("#stepper li").forEach((li) => { li.dataset.state = "pending"; });
  $("#stepper").hidden = false;
  const wantHeatmap = $("#opt-heatmap").checked;
  let current = "read";

  try {
    setStep("read", "active");
    const raw = await KneePreprocess.decodeToGray224(Analysis.file);
    const original = await toJpeg(Analysis.url, 448);
    setStep("read", "done");

    current = "grade";
    setStep("grade", "active");
    await Engine.load();
    await nextFrame();
    const pixels = KneePreprocess.preprocess(raw);
    const res = KneeModel.analyze(tf, Engine.members, Engine.cfg, pixels, false);
    setStep("grade", "done");

    const rec = {
      id: newId(),
      createdAt: new Date().toISOString(),
      fileName: Analysis.file.name,
      grade: res.grade,
      topGrade: res.topGrade,
      probabilities: res.probabilities.map((p) => Math.round(p * 1e4) / 1e4),
      adjustedScores: res.adjustedScores.map((p) => Math.round(p * 1e4) / 1e4),
      invertedCorrected: KneePreprocess.polarityScore(raw) < 0,
      preprocessed: grayToDataUrl(pixels),
      original,
      heatmap: null,
      jointShare: null,
    };
    renderResult($("#result"), rec);

    if (wantHeatmap) {
      current = "heatmap";
      setStep("heatmap", "active");
      await nextFrame();
      const withCam = KneeModel.analyze(tf, Engine.members, Engine.cfg, pixels, true);
      rec.heatmap = pixelsToDataUrl(KneeModel.heatmapRGBA(withCam.cam));
      rec.jointShare = withCam.jointShare;
      setStep("heatmap", "done");
      renderResult($("#result"), rec);
    } else {
      setStep("heatmap", "skipped");
    }

    current = "save";
    setStep("save", "active");
    rec.thumbnail = await toJpeg(rec.preprocessed, 96, 0.8);
    await DB.put(rec);
    setStep("save", "done");
    toast("Saved to history");
  } catch (err) {
    setStep(current, "error");
    showError(current === "save"
      ? "The result couldn't be saved to history. Your browser may be blocking storage for this site."
      : err.message);
  } finally {
    Analysis.busy = false;
    btn.disabled = false;
    btn.textContent = "Analyze X-ray";
  }
}

async function loadExamples() {
  try {
    const items = await (await fetch("data/examples.json")).json();
    if (!items.length) { $(".examples").hidden = true; return; }
    $("#examples").innerHTML = items.map((it) => `
      <button type="button" class="example" data-url="${esc(it.url)}" data-name="${esc(it.name)}">
        <img src="${esc(it.url)}" alt="" loading="lazy">
        <span>${it.label == null ? "Example" : `Dataset label: grade ${it.label}`}</span>
      </button>`).join("");
  } catch {
    $(".examples").hidden = true;
  }
}

function wireAnalyze() {
  const input = $("#file-input");
  input.addEventListener("change", () => { selectFile(input.files[0]); input.value = ""; });
  $$("[data-dropzone]").forEach((zone) => {
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("drag");
      selectFile(e.dataTransfer.files[0]);
    });
  });
  $("#change-file").addEventListener("click", () => input.click());
  $("#analyze-btn").addEventListener("click", runAnalysis);
  $("#examples").addEventListener("click", async (e) => {
    const card = e.target.closest(".example");
    if (!card) return;
    try {
      const blob = await (await fetch(card.dataset.url)).blob();
      selectFile(new File([blob], card.dataset.name, { type: "image/png" }));
    } catch {
      showError("The example couldn't be loaded. Try again.");
    }
  });
  resetResult();
}

/* =====================================================================
   Dashboard
   ===================================================================== */
function recordButton(r) {
  return `<button type="button" class="rec" data-open="${r.id}">
    <img src="${r.thumbnail}" alt="">
    <span class="rec-main"><span class="rec-name">${esc(r.fileName)}</span><span class="rec-date">${fmtDate(r.createdAt)}</span></span>
    ${badge(r.grade)}
  </button>`;
}

function distBars(counts) {
  const max = Math.max(1, ...counts);
  return `<div class="dist">${counts.map((c, g) => `
    <div class="dist-row" style="--gc:${gradeVar(g)}">
      <span class="dist-label"><b>${g}</b>${GRADES[g].name}</span>
      <span class="dist-track"><span class="dist-fill" style="width:${(c / max) * 100}%"></span></span>
      <span class="dist-val">${c}</span>
    </div>`).join("")}</div>`;
}

async function renderDashboard() {
  const recs = await DB.all();
  const counts = [0, 0, 0, 0, 0];
  recs.forEach((r) => { counts[r.grade] += 1; });
  const weekAgo = Date.now() - 7 * 24 * 3600 * 1000;
  const week = recs.filter((r) => Date.parse(r.createdAt) >= weekAgo).length;
  const common = recs.length ? counts.indexOf(Math.max(...counts)) : null;

  $("#dash-stats").innerHTML = [
    ["Analyses saved", recs.length, "In this browser"],
    ["Last 7 days", week, week === 1 ? "Analysis this week" : "Analyses this week"],
    ["Most frequent grade", common == null ? "None yet" : `Grade ${common}`, common == null ? "Grade an X-ray to start" : GRADES[common].name],
    ["Grades 3 and 4", counts[3] + counts[4], "Moderate or severe results"],
  ].map(([label, value, sub]) => `
    <div class="stat"><span class="stat-label">${label}</span><span class="stat-value">${value}</span><span class="stat-sub">${sub}</span></div>`).join("");

  $("#dash-recent").innerHTML = recs.length
    ? `<div class="rec-list">${recs.slice(0, 5).map(recordButton).join("")}</div>`
    : `<div class="empty"><strong>No analyses yet</strong>Drop an X-ray above to grade your first one. Results stay here after you refresh.</div>`;

  $("#dash-dist").innerHTML = recs.length
    ? distBars(counts)
    : `<div class="empty">The spread of grades across your analyses will appear here.</div>`;
}

/* =====================================================================
   History
   ===================================================================== */
const History = { recs: [] };

async function renderHistory() {
  History.recs = await DB.all();
  applyFilters();
}

function filteredRecords() {
  const q = $("#f-search").value.trim().toLowerCase();
  const grade = $("#f-grade").value;
  const from = $("#f-from").value;
  const to = $("#f-to").value;
  const sort = $("#f-sort").value;
  const rows = History.recs.filter((r) =>
    (!q || r.fileName.toLowerCase().includes(q)) &&
    (grade === "" || r.grade === Number(grade)) &&
    (!from || localDay(r.createdAt) >= from) &&
    (!to || localDay(r.createdAt) <= to));
  const byDate = (a, b) => a.createdAt.localeCompare(b.createdAt);
  const sorters = {
    new: (a, b) => byDate(b, a),
    old: byDate,
    "grade-desc": (a, b) => b.grade - a.grade || byDate(b, a),
    "grade-asc": (a, b) => a.grade - b.grade || byDate(b, a),
  };
  return rows.sort(sorters[sort] || sorters.new);
}

function applyFilters() {
  const rows = filteredRecords();
  const total = History.recs.length;
  $("#h-count").textContent = total ? `Showing ${rows.length} of ${total} ${total === 1 ? "analysis" : "analyses"}` : "";
  $("#h-export").disabled = rows.length === 0;
  $("#h-clear").disabled = total === 0;

  if (!total) {
    $("#h-table").innerHTML = `<div class="empty"><strong>Your history is empty</strong>Every analysis you run is saved here automatically and stays after you refresh.<br><a class="btn primary" href="#/analyze">Start a new analysis</a></div>`;
    return;
  }
  if (!rows.length) {
    $("#h-table").innerHTML = `<div class="empty"><strong>No analyses match these filters</strong><button type="button" class="btn" id="reset-filters">Reset filters</button></div>`;
    $("#reset-filters").addEventListener("click", () => {
      ["#f-search", "#f-grade", "#f-from", "#f-to"].forEach((s) => { $(s).value = ""; });
      $("#f-sort").value = "new";
      applyFilters();
    });
    return;
  }
  $("#h-table").innerHTML = `<div class="table-wrap"><table class="table">
    <thead><tr><th>X-ray</th><th>File</th><th>Analyzed</th><th>Grade</th><th class="num">Model probability</th><th>Heatmap</th><th><span class="sr-only">Actions</span></th></tr></thead>
    <tbody>${rows.map((r) => `
      <tr class="clickable" data-open="${r.id}">
        <td><img class="thumb" src="${r.thumbnail}" alt=""></td>
        <td class="truncate" style="max-width:240px">${esc(r.fileName)}</td>
        <td>${fmtDate(r.createdAt)}</td>
        <td>${badge(r.grade)}</td>
        <td class="num">${pct(r.probabilities[r.grade])}</td>
        <td>${r.heatmap ? "Yes" : "No"}</td>
        <td><div class="row-actions">
          <button type="button" class="icon-btn" data-open="${r.id}" aria-label="View ${esc(r.fileName)}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/></svg></button>
          <button type="button" class="icon-btn del" data-delete="${r.id}" aria-label="Delete ${esc(r.fileName)}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 13h9l1-13"/></svg></button>
        </div></td>
      </tr>`).join("")}</tbody></table></div>`;
}

async function deleteRecord(id) {
  const rec = History.recs.find((r) => r.id === id) || (await DB.get(id));
  if (!rec || !confirm(`Delete the analysis of "${rec.fileName}"? This can't be undone.`)) return false;
  await DB.del(id);
  toast("Analysis deleted");
  return true;
}

function exportCsv() {
  const rows = filteredRecords();
  const header = ["analyzed_at", "file", "kl_grade", "grade_name", "model_probability_of_grade", "p_grade0", "p_grade1", "p_grade2", "p_grade3", "p_grade4", "heatmap", "joint_share"];
  const quote = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = rows.map((r) => [
    r.createdAt, r.fileName, r.grade, GRADES[r.grade].name, r.probabilities[r.grade], ...r.probabilities,
    r.heatmap ? "yes" : "no", r.jointShare ?? "",
  ].map(quote).join(","));
  const blob = new Blob([[header.join(","), ...lines].join("\n")], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `kneegrade-history-${localDay(new Date().toISOString())}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function wireHistory() {
  let timer;
  $("#f-search").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(applyFilters, 150); });
  ["#f-grade", "#f-from", "#f-to", "#f-sort"].forEach((s) => $(s).addEventListener("change", applyFilters));
  $("#h-export").addEventListener("click", exportCsv);
  $("#h-clear").addEventListener("click", async () => {
    if (!confirm("Delete every saved analysis in this browser? This can't be undone.")) return;
    await DB.clear();
    toast("History cleared");
    renderHistory();
  });
  $("#h-table").addEventListener("click", async (e) => {
    const del = e.target.closest("[data-delete]");
    if (del) {
      e.stopPropagation();
      if (await deleteRecord(del.dataset.delete)) renderHistory();
      return;
    }
    const open = e.target.closest("[data-open]");
    if (open) Modal.open(open.dataset.open);
  });
  $("#dash-recent").addEventListener("click", (e) => {
    const open = e.target.closest("[data-open]");
    if (open) Modal.open(open.dataset.open);
  });
}

/* =====================================================================
   Detail modal
   ===================================================================== */
const Modal = {
  id: null,
  lastFocus: null,
  async open(id) {
    const rec = await DB.get(id);
    if (!rec) return;
    this.id = id;
    this.lastFocus = document.activeElement;
    $("#modal-title").textContent = rec.fileName;
    renderResult($("#modal-body"), rec);
    $("#modal").hidden = false;
    document.body.classList.add("modal-open");
    $("#modal-close").focus();
  },
  close() {
    $("#modal").hidden = true;
    document.body.classList.remove("modal-open");
    this.lastFocus?.focus?.();
  },
};

function wireModal() {
  $("#modal-close").addEventListener("click", () => Modal.close());
  $("#modal").addEventListener("click", (e) => { if (e.target.closest("[data-close]")) Modal.close(); });
  $("#modal-delete").addEventListener("click", async () => {
    if (await deleteRecord(Modal.id)) {
      Modal.close();
      route();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("#modal").hidden) Modal.close();
    else if (Sidebar.mobile() && Sidebar.isOpen()) Sidebar.set(false);
  });
}

/* =====================================================================
   Model insights (static results, plus training curves from the server)
   ===================================================================== */
let insightsRendered = false;

function confusionGrid(m) {
  let html = `<div class="cm"><div></div>${[0, 1, 2, 3, 4].map((g) => `<div class="cm-h">${g}</div>`).join("")}`;
  m.forEach((row, t) => {
    const total = row.reduce((a, b) => a + b, 0);
    html += `<div class="cm-rh">Grade ${t}</div>`;
    row.forEach((n, p) => {
      const share = total ? n / total : 0;
      const bg = `color-mix(in srgb, var(--accent) ${Math.round(share * 90)}%, #F1F5F7)`;
      const fg = share > 0.45 ? "#fff" : "var(--ink)";
      html += `<div class="cm-cell${t === p ? " diag" : ""}" style="background:${bg};color:${fg}" title="True ${t}, predicted ${p}: ${n} X-rays"><b>${n}</b><span>${Math.round(share * 100)}%</span></div>`;
    });
  });
  return `${html}</div>`;
}

async function renderInsights() {
  if (insightsRendered) return;
  insightsRendered = true;

  $("#ins-metrics").innerHTML = TEST.metrics.map((m) => `
    <div class="metric${m.primary ? " primary-metric" : ""}">
      <div class="metric-label">${m.label}</div>
      <div class="metric-value">${m.value.toFixed(3)}</div>
      ${m.ci ? `<div class="metric-ci">95% CI ${m.ci[0].toFixed(3)} to ${m.ci[1].toFixed(3)}</div>` : `<div class="metric-ci">&nbsp;</div>`}
      <div class="metric-note">${m.note}</div>
    </div>`).join("");

  $("#ins-perclass").innerHTML = `<div class="table-wrap"><table class="table compact">
    <thead><tr><th>Grade</th><th class="num">Precision</th><th>Recall</th><th class="num">F1</th><th class="num">X-rays</th></tr></thead>
    <tbody>${TEST.perClass.map(([p, r, f, n], g) => `
      <tr><td>${badge(g)}</td><td class="num">${p.toFixed(2)}</td>
      <td>${r.toFixed(2)}<span class="recall-bar" style="--gc:${gradeVar(g)}"><span style="width:${r * 100}%"></span></span></td>
      <td class="num">${f.toFixed(2)}</td><td class="num">${n}</td></tr>`).join("")}</tbody></table></div>
    <p class="fine">Severe and moderate osteoarthritis are recognized well. Grade 1 ("doubtful") is the weak point: the model catches 44% of grade-1 knees but also grades 46% of healthy knees as 1.</p>`;

  $("#ins-cm").innerHTML = `${confusionGrid(TEST.confusion)}<p class="fine">Almost every error is to a neighbouring grade. No healthy knee was ever graded 4.</p>`;

  $("#ins-exp").innerHTML = `<div class="table-wrap"><table class="table compact">
    <thead><tr><th>Run</th><th>Change</th><th class="num">Balanced acc.</th><th class="num">QWK</th></tr></thead>
    <tbody>${EXPERIMENTS.map(([run, change, ba, q, final]) => `
      <tr class="${final ? "exp-final" : ""}"><td>${run}</td><td class="muted">${change}</td><td class="num">${ba.toFixed(3)}</td><td class="num">${q.toFixed(3)}</td></tr>`).join("")}</tbody></table></div>`;

  const total = SPLITS.Training.map((_, g) => SPLITS.Training[g] + SPLITS.Validation[g] + SPLITS.Test[g]);
  $("#ins-data").innerHTML = `${distBars(total)}
    <div class="table-wrap" style="margin-top:16px"><table class="table compact">
      <thead><tr><th>Split</th>${[0, 1, 2, 3, 4].map((g) => `<th class="num">${g}</th>`).join("")}<th class="num">Total</th></tr></thead>
      <tbody>${Object.entries(SPLITS).map(([name, c]) => `<tr><td>${name}</td>${c.map((n) => `<td class="num">${n.toLocaleString()}</td>`).join("")}<td class="num"><b>${c.reduce((a, b) => a + b, 0).toLocaleString()}</b></td></tr>`).join("")}</tbody>
    </table></div>
    <p class="fine">Grade 4 is only 3% of the data. Always predicting "healthy" would score 40% accuracy but just 20% balanced accuracy.</p>`;

  await renderCurves();
}

async function renderCurves() {
  const note = $("#ch-note");
  if (!window.Chart) { note.textContent = "Charts need an internet connection to load the charting library."; return; }
  let curves;
  try {
    curves = await (await fetch("data/curves.json")).json();
  } catch {
    note.textContent = "Training curves couldn't be loaded.";
    return;
  }
  const a = curves.exp1_augmentation;
  const w = curves.exp3_aug_wider;
  if (!a) { note.textContent = "Training history data wasn't found."; return; }

  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.color = "#586777";
  const epochs = (h) => h.epoch.map((e) => e + 1);
  const line = (label, data, color, dashed = false) => ({ label, data, borderColor: color, backgroundColor: color, borderDash: dashed ? [5, 4] : [], borderWidth: 2, pointRadius: 0, tension: 0.25 });
  const options = (yLabel) => ({
    responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
    plugins: { legend: { position: "bottom", labels: { boxWidth: 12, boxHeight: 2 } } },
    scales: { x: { title: { display: true, text: "Epoch" }, grid: { display: false } }, y: { title: { display: true, text: yLabel }, grid: { color: "#E6EBEF" } } },
  });
  const longest = w && w.epoch.length > a.epoch.length ? epochs(w) : epochs(a);

  new Chart($("#ch-balacc"), {
    type: "line",
    data: { labels: longest, datasets: [line("Baseline + augmentation", a.val_balanced_accuracy, "#0B6E66"), ...(w ? [line("Wider network", w.val_balanced_accuracy, "#D39A22")] : [])] },
    options: options("Balanced accuracy"),
  });
  new Chart($("#ch-loss"), {
    type: "line",
    data: { labels: epochs(a), datasets: [line("Training", a.loss, "#0F2F47"), line("Validation", a.val_loss, "#0B6E66", true)] },
    options: options("Loss"),
  });
  new Chart($("#ch-acc"), {
    type: "line",
    data: { labels: epochs(a), datasets: [line("Training", a.accuracy, "#0F2F47"), line("Validation", a.val_accuracy, "#0B6E66", true)] },
    options: options("Accuracy"),
  });
  note.textContent = "Training and validation accuracy stay close, the effect of augmentation. Each model was checkpointed at its best validation balanced accuracy.";
}

/* =====================================================================
   About
   ===================================================================== */
function renderAbout() {
  const el = $("#about-scale");
  if (el.dataset.done) return;
  el.dataset.done = "1";
  el.innerHTML = `<div class="kl-list">${GRADES.map((g, i) => `<div class="kl-row">${badge(i)}<span>${g.desc}</span></div>`).join("")}</div>`;
}

/* =====================================================================
   Start-up
   ===================================================================== */
async function loadModel() {
  const state = $("#model-state");
  try {
    await Engine.load();
    state.className = "model-state ok";
    state.lastElementChild.textContent = `Model ${Engine.cfg.version} ready, runs on this device`;
    $("#link-model").href = `https://huggingface.co/${Engine.cfg.model_repo}`;
  } catch (err) {
    state.className = "model-state down";
    state.lastElementChild.textContent = "Model couldn't load. Refresh to retry.";
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  Sidebar.init();
  wireAnalyze();
  wireHistory();
  wireModal();
  try {
    await DB.open();
    if (navigator.storage && navigator.storage.persist) navigator.storage.persist();
  } catch {
    toast("History can't be saved: this browser is blocking storage for the site.");
  }
  loadModel();
  loadExamples();
  window.addEventListener("hashchange", route);
  route();
});
