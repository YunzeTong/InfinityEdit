/* =====================================================================
 * InfinityEdit project page — editing-chain config + renderer
 *
 * HOW TO ADD A DEMO:
 *   1. Drop your mp4 files in static/videos/
 *   2. Add an entry to CHAINS below.
 *
 * Each chain = { tag, title, steps: [...] }
 *   tag    : where it renders -> 'teaser' | 'gallery' | 'compare'
 *   title  : row heading
 *   steps  : ordered list of clips. steps[0] is the SOURCE (no instruction).
 *            every later step needs an `instruction` (the edit applied to the
 *            PREVIOUS clip to produce this one).
 *
 * Each step = { src, label?, instruction?, poster? }
 *   src         : path to the mp4
 *   label       : caption under the clip (default: "Source" / "Edit N")
 *   instruction : the edit text shown on the arrow leading INTO this clip
 *   poster      : optional still image shown before the video loads
 * ===================================================================== */

const CHAINS = [
  // ---- TEASER (shows at the top) ----
  {
    tag: "teaser",
    title: "Style → Camera → Weather → Time-of-day",
    steps: [
      { src: "static/videos/placeholder.mp4", label: "Source" },
      { src: "static/videos/placeholder.mp4", instruction: "Van Gogh oil-painting style",
        instructionFull: "Repaint the entire scene in the style of a Van Gogh oil painting, with thick visible brushstrokes, swirling textured skies, and a vivid warm color palette, while preserving the original motion and camera trajectory. (Replace with your real 80–120 word instruction.)" },
      { src: "static/videos/placeholder.mp4", instruction: "Orbit the camera right",
        instructionFull: "Slowly orbit the camera to the right around the subject, revealing the scene from a new angle while keeping the painted appearance and subject motion consistent. (Replace with your real 80–120 word instruction.)" },
      { src: "static/videos/placeholder.mp4", instruction: "Make it snow heavily",
        instructionFull: "Introduce heavy falling snow throughout the scene, accumulating on surfaces and drifting with the wind, while preserving the established style, subjects, and camera movement. (Replace with your real 80–120 word instruction.)" },
      { src: "static/videos/placeholder.mp4", instruction: "Golden-hour sunset lighting",
        instructionFull: "Shift the lighting to a warm golden-hour sunset, with long soft shadows and amber highlights, keeping all prior edits, subjects, and motion intact. (Replace with your real 80–120 word instruction.)" },
    ],
  },

  // ---- GALLERY chains ----
  {
    tag: "gallery",
    title: "Chain 1 — Restyling a continuous shot",
    steps: [
      { src: "static/videos/placeholder.mp4", label: "Source" },
      { src: "static/videos/placeholder.mp4", instruction: "Turn it into a cyberpunk neon style." },
      { src: "static/videos/placeholder.mp4", instruction: "Add heavy rain and wet reflections." },
      { src: "static/videos/placeholder.mp4", instruction: "Convert to a pencil-sketch look." },
    ],
  },
  {
    tag: "gallery",
    title: "Chain 2 — Camera moves on an ongoing scene",
    steps: [
      { src: "static/videos/placeholder.mp4", label: "Source" },
      { src: "static/videos/placeholder.mp4", instruction: "Push the camera in toward the subject." },
      { src: "static/videos/placeholder.mp4", instruction: "Pan left to reveal the background." },
    ],
  },

  // ---- COMPARISON rows (optional; delete the section in index.html if unused) ----
  {
    tag: "compare",
    title: "Helios baseline (prompt-swap)",
    steps: [
      { src: "static/videos/placeholder.mp4", label: "Source" },
      { src: "static/videos/placeholder.mp4", instruction: "Repaint in watercolor style." },
      { src: "static/videos/placeholder.mp4", instruction: "Repaint in watercolor style. (drifts / breaks)" },
    ],
  },
  {
    tag: "compare",
    title: "Ours",
    steps: [
      { src: "static/videos/placeholder.mp4", label: "Source" },
      { src: "static/videos/placeholder.mp4", instruction: "Repaint in watercolor style." },
      { src: "static/videos/placeholder.mp4", instruction: "Repaint in watercolor style. (stable)" },
    ],
  },
];

/* =====================================================================
 * SHORT VIDEO EDITING — a single instruction, grouped by category.
 *
 * SHORT_EDITS = [ { category, cases: [ caseObj, ... ] } ]
 * caseObj = {
 *   title           : sub-label for this case (e.g. "Ghibli")
 *   source, edited  : mp4 paths (source clip and the result after ONE edit)
 *   instruction     : short label shown on the arrow
 *   instructionFull : (optional) full 80–120 word instruction, shown on hover/tap
 * }
 * Add more cases per category by pushing into its `cases` array.
 * ===================================================================== */

const SHORT_EDITS = [
  {
    category: "Style Transfer",
    cases: [
      {
        title: "Ghibli",
        source: "static/videos/placeholder.mp4",
        edited: "static/videos/placeholder.mp4",
        instruction: "Studio Ghibli style",
        instructionFull:
          "Repaint the entire scene in the hand-drawn Studio Ghibli animation style: soft watercolor backgrounds, gentle pastel color palette, rounded character shapes with simple expressive features, and warm diffuse lighting, while preserving the original motion, subject positions, and camera trajectory of the clip. (Replace with your real 80–120 word instruction.)",
      },
    ],
  },
  {
    category: "Entity Transformation",
    cases: [
      {
        title: "Cat → Tiger",
        source: "static/videos/placeholder.mp4",
        edited: "static/videos/placeholder.mp4",
        instruction: "Turn the cat into a tiger",
        instructionFull:
          "Transform the cat in the scene into a full-grown Bengal tiger, keeping its pose, movement, and screen position consistent with the source; give it orange fur with black stripes, a larger muscular body, and realistic fur shading, while leaving the background and camera motion unchanged. (Replace with your real 80–120 word instruction.)",
      },
    ],
  },
  {
    category: "Motion Transfer",
    cases: [
      {
        title: "Add walking motion",
        source: "static/videos/placeholder.mp4",
        edited: "static/videos/placeholder.mp4",
        instruction: "Make the subject start walking",
        instructionFull:
          "Animate the standing subject so that it begins walking forward naturally, with a smooth, physically plausible gait that continues the existing scene; keep the appearance, clothing, lighting, and background identical to the source and only introduce the new locomotion. (Replace with your real 80–120 word instruction.)",
      },
    ],
  },
  {
    category: "Camera Movement Control",
    cases: [
      {
        title: "Orbit right",
        source: "static/videos/placeholder.mp4",
        edited: "static/videos/placeholder.mp4",
        instruction: "Orbit the camera to the right",
        instructionFull:
          "Move the camera in a smooth arc orbiting to the right around the main subject, revealing the scene from a new viewpoint while keeping the subject and scene appearance unchanged; the motion should be steady and continuous, preserving temporal coherence with the preceding frames. (Replace with your real 80–120 word instruction.)",
      },
    ],
  },
];

/* --------------------------- renderer --------------------------- */

const DEFAULT_POSTER = "static/images/clip-placeholder.svg";

function makeVideo(step) {
  const v = document.createElement("video");
  v.src = step.src;
  v.poster = step.poster || DEFAULT_POSTER;
  v.muted = true;
  v.loop = true;
  v.autoplay = true;
  v.playsInline = true;
  v.setAttribute("playsinline", "");
  v.preload = "metadata";
  v.controls = false;
  return v;
}

function makeCard(step, idx) {
  const card = document.createElement("div");
  card.className = "vid-card";
  card.appendChild(makeVideo(step));

  const label = document.createElement("div");
  const isSource = idx === 0;
  label.className = "vid-label" + (isSource ? " source" : "");
  const text = step.label || (isSource ? "Source" : `Edit ${idx}`);
  label.innerHTML = isSource
    ? text
    : `<span class="step-idx">#${idx}</span> ${text}`;
  card.appendChild(label);
  return card;
}

/* Build the instruction chip. `short` is always shown; if `full` is provided it
 * appears in a popover on hover (desktop) or tap (touch). */
function makeInstr(short, full) {
  const chip = document.createElement("div");
  chip.className = "instr";
  const label = (short || "").trim();
  if (full && full.trim() && full.trim() !== label) {
    chip.classList.add("has-full");
    chip.innerHTML =
      `<span class="instr-label">${label}</span>` +
      `<i class="fas fa-circle-info instr-icon"></i>` +
      `<span class="instr-full">${full.trim()}</span>`;
    const pop = chip.querySelector(".instr-full");
    // Popover is position:fixed so it escapes the row's horizontal scroll clip;
    // we compute its coordinates each time it is shown.
    const place = () => {
      pop.style.display = "block"; // force layout so we can measure size
      const r = chip.getBoundingClientRect();
      const w = Math.min(pop.offsetWidth || 280, window.innerWidth - 24);
      let left = r.left + r.width / 2 - w / 2;
      left = Math.max(12, Math.min(left, window.innerWidth - w - 12));
      let top = r.top - pop.offsetHeight - 10;
      if (top < 8) top = r.bottom + 10; // flip below if no room above
      pop.style.left = left + "px";
      pop.style.top = top + "px";
      pop.style.display = ""; // hand visibility back to CSS (:hover / .open)
    };
    chip.addEventListener("mouseenter", place);
    chip.addEventListener("click", (e) => {
      e.stopPropagation();
      document
        .querySelectorAll(".instr.open")
        .forEach((el) => el !== chip && el.classList.remove("open"));
      chip.classList.toggle("open");
      if (chip.classList.contains("open")) place();
    });
  } else {
    chip.textContent = label;
  }
  return chip;
}

function makeArrow(step) {
  const a = document.createElement("div");
  a.className = "chain-arrow";
  a.appendChild(makeInstr(step.instruction, step.instructionFull));
  const glyph = document.createElement("div");
  glyph.className = "arrow-glyph";
  glyph.innerHTML = "&rarr;";
  a.appendChild(glyph);
  return a;
}

function renderChain(chain) {
  const wrap = document.createElement("div");
  wrap.className = "edit-chain";

  if (chain.title) {
    const h = document.createElement("div");
    h.className = "chain-title";
    h.innerHTML = `<span class="chain-tag">CHAIN</span>${chain.title}`;
    wrap.appendChild(h);
  }

  const row = document.createElement("div");
  row.className = "chain-row";
  chain.steps.forEach((step, idx) => {
    if (idx > 0) row.appendChild(makeArrow(step));
    row.appendChild(makeCard(step, idx));
  });
  wrap.appendChild(row);
  return wrap;
}

/* ---- Short (single-instruction) editing, grouped by category ---- */

function renderShortCase(c) {
  const wrap = document.createElement("div");
  wrap.className = "edit-chain short-case";
  if (c.title) {
    const h = document.createElement("div");
    h.className = "chain-title";
    h.innerHTML = `<span class="chain-tag">CASE</span>${c.title}`;
    wrap.appendChild(h);
  }
  const row = document.createElement("div");
  row.className = "chain-row";
  row.appendChild(makeCard({ src: c.source, label: "Source", poster: c.sourcePoster }, 0));
  row.appendChild(makeArrow({ instruction: c.instruction, instructionFull: c.instructionFull }));
  row.appendChild(makeCard({ src: c.edited, label: "Edited", poster: c.editedPoster }, 1));
  wrap.appendChild(row);
  return wrap;
}

function renderShortEdits(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  SHORT_EDITS.forEach((group) => {
    const block = document.createElement("div");
    block.className = "category-block";
    const h = document.createElement("h3");
    h.className = "category-title";
    h.textContent = group.category;
    block.appendChild(h);
    const grid = document.createElement("div");
    grid.className = "cases-grid";
    group.cases.forEach((c) => grid.appendChild(renderShortCase(c)));
    block.appendChild(grid);
    container.appendChild(block);
  });
}

function mount(tag, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  CHAINS.filter((c) => c.tag === tag).forEach((c) =>
    container.appendChild(renderChain(c))
  );
}

document.addEventListener("DOMContentLoaded", () => {
  mount("teaser", "teaser-chain");
  renderShortEdits("short-edits");
  mount("gallery", "chains-gallery");
  mount("compare", "chains-compare");
  // tap anywhere else closes any open instruction popover
  document.addEventListener("click", () =>
    document.querySelectorAll(".instr.open").forEach((el) => el.classList.remove("open"))
  );
});
