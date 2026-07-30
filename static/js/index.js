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
  // ---- LONG VIDEO GENERATION : long-generation/case1 & case2 ----
  {
    tag: "longgen",
    title: "Pianist in an opulent room — Simpsons style, then Labubu toy",
    steps: [
      { src: "static/videos/long-generation/case1/source.mp4", label: "Source" },
      { src: "static/videos/long-generation/case1/edit_1_simpsons%20comic.mp4", instruction: "Simpsons comic style",
        instructionFull: "Apply the simpsons comic transformation to the man sitting at the grand piano. Redraw the man with bright yellow skin and large, bulging round eyes while maintaining his formal vest and white shirt. Modify his facial structure to include the signature overbite and simplify his hands to have only four fingers as they move gracefully over the piano keys. Apply bold black outlines to his entire figure to match the distinctive animation style. Ensure the surrounding ornate wooden paneling, classical sculptures, and the landscape seen through the windows remain in their realistic cinematic style, creating a visual contrast between the cartoon character and the opulent room." },
      { src: "static/videos/long-generation/case1/edit_2_ladudu%20me.mp4", instruction: "Labubu designer toy",
        instructionFull: "Transform the man seated at the grand piano into a Labubu-style designer toy character while maintaining his seated posture and interaction with the keys. Replace his human features with a small, rounded body dressed in a miniature version of the white shirt and formal vest. Give the figure its signature mischievous grin and prominent pointed ears, applying a smooth, playful vinyl texture that reflects the warm glow from the vintage lamp and the soft sunlight from the large windows. The surrounding ornate wooden paneling, classical sculptures, and the view of the lush greenery outside must remain unchanged, ensuring the vinyl character is seamlessly integrated into the opulent, naturally lit room." },
    ],
  },

  {
    tag: "longgen",
    title: "Mycologist in a forest — 3D cartoon doll, then Studio Ghibli",
    steps: [
      { src: "static/videos/long-generation/case2/source.mp4", label: "Source" },
      { src: "static/videos/long-generation/case2/edit_1_cartoon%20doll.mp4", instruction: "3D cartoon doll",
        instructionFull: "Transform the middle-aged man into a stylized 3D cartoon doll while he examines the white mushroom. Replace his human features with a miniature, toy-like face featuring large, expressive round eyes and smooth, polished plastic-like skin. Maintain his original dark shirt and cap, but render them with simplified, bold textures characteristic of a high-quality figurine. His hands, still delicately turning the smooth white mushroom, should appear with softened, rounded proportions. The surrounding lush forest and dappled sunlight remain realistic, creating a contrast against the glossy, exaggerated doll character who retains the original focused pose and steady movements within the greenery." },
      { src: "static/videos/long-generation/case2/edit_2_ghibli.mp4", instruction: "Studio Ghibli style",
        instructionFull: "Convert the scene into a Studio Ghibli anime aesthetic. Transform the dense greenery into a hand-painted backdrop featuring lush, vibrant emerald and soft lime watercolor textures. Stylize the middle-aged man with clean, hand-drawn outlines, softening his dark shirt and cap into muted, warm pastel tones. The white mushroom should gain a gentle, whimsical glow with delicate line-work. Replace the realistic dappled sunlight with soft, golden atmospheric lighting that creates a dreamy, painterly effect across the forest floor. The overall texture must shift from realistic documentary footage to a smooth, cel-shaded look with gentle gradients and a nostalgic, whimsical atmosphere characteristic of traditional Japanese animation." },
    ],
  },

  {
    tag: "longgen",
    title: "European robin in an autumn forest — Irasutoya, then Minecraft",
    steps: [
      { src: "static/videos/long-generation/case3/source.mp4", label: "Source" },
      { src: "static/videos/long-generation/case3/edit_1_irasutoya.mp4", instruction: "Irasutoya illustration",
        instructionFull: "Transform the entire scene into the Irasutoya illustration style. Convert the European robin into a simplified kawaii character by replacing its detailed plumage with flat, pastel orange and grayish-brown sections defined by thick, soft, rounded outlines. The robin’s sharp black eyes and slender beak must become simple, friendly dots and a minimal pointed shape. Flatten the texture of the moss-covered rock into a smooth, rounded green and brown form without realistic moss detail. Simplify the warm-colored autumn trees in the background into soft, cheerful pastel blobs. Remove all complex textures and diffused lighting, replacing them with a bright, flat color palette and minimal shading characteristic of Japanese clip art." },
      { src: "static/videos/long-generation/case3/edit_2_minecraft.mp4", instruction: "Minecraft voxel art",
        instructionFull: "Transform the European robin and its immediate environment into a Minecraft voxel art style. Convert the bird's vibrant orange breast and grayish-brown plumage into distinct, low-resolution cubic blocks, ensuring the sharp black eyes and slender beak are represented by single voxels. The moss-covered rock beneath the bird must be restyled into a grid of textured green and earthy-brown cubes. Simplify the soft, blurred background of warm-colored trees into a pixelated mosaic of large orange and yellow squares. The robin's subtle head turns and tail flicks should be rendered as rigid, blocky movements, maintaining the retro game aesthetic while preserving the bird's recognizable shape and the tranquil autumn forest layout." },
    ],
  },

  // ---- INFINITE SEQUENTIAL EDITING : comparison/case3 (Dolomites) & case1 (library),
  //      our method only. (case2 is now used in the Comparison section below.)
  {
    tag: "infinite",
    title: "Tre Cime di Lavaredo peaks — move up, zoom out, then American comic",
    steps: [
      { src: "static/videos/comparison/case3/ours/source.mp4", label: "Source" },
      { src: "static/videos/comparison/case3/ours/edit_1_move%20up.mp4", instruction: "Move up (tilt)",
        instructionFull: "Execute a smooth upward tilt movement starting from the wide-angle shot of the three peaks. Begin with the rugged rock formations of the central peak centered in the frame, then gradually move the camera vertically to reveal the upper reaches of the mountain range. As the camera ascends, the frame should transition from the lower valleys and the interplay of shadows on the mountain's base to focus on the thick mist clinging to the summits. Conclude the movement by revealing the vast, overcast sky above the peaks, ensuring the final composition captures the patches of sunlight breaking through the clouds to illuminate the highest rock textures and the sharp edges of the range." },
      { src: "static/videos/comparison/case3/ours/edit_2_zoom%20out.mp4", instruction: "Zoom out",
        instructionFull: "Starting from the tight framing on the central peak’s rugged rock formations and its play of light and shadow, execute a smooth zoom out. As the camera retreats, expand the frame to reveal the two flanking peaks of the Tre Cime di Lavaredo and the mist clinging to their vertical faces. Continue the outward movement until the surrounding valleys and the wider mountain range are fully visible within the composition. Ensure the transition reveals the expansive, cloudy sky and the patches of sunlight highlighting the distant terrain, transforming the shot from a detailed study of a single summit into a comprehensive panoramic view of the entire pristine Dolomites landscape." },
      { src: "static/videos/comparison/case3/ours/edit_3_american%20comic.mp4", instruction: "American comic style",
        instructionFull: "Convert the Tre Cime di Lavaredo sequence into a classic American comic book style. Apply thick, black inked outlines to the jagged edges of the three peaks and the surrounding rock formations. Transform the overcast sky into a backdrop of saturated teals and deep purples, utilizing prominent halftone dot textures across the mist clinging to the mountainsides. The subtle sunlight hitting the central peak must be rendered as vibrant, flat yellow patches, while the shadows are replaced with high-contrast, stylized black hatching. Ensure the valleys and distant mountains maintain this graphic look, with every contour of the rugged terrain defined by bold, hand-drawn ink strokes and vivid, saturated color palettes." },
    ],
  },

  {
    tag: "infinite",
    title: "Young man in a library — Labubu toy, Simpsons comic, then 3D cartoon doll",
    steps: [
      { src: "static/videos/comparison/case1/ours/source.mp4", label: "Source" },
      { src: "static/videos/comparison/case1/ours/edit_1_ladudu%20me.mp4", instruction: "Labubu toy",
        instructionFull: "Transform the young man wearing the maroon hoodie into a Labubu-style designer toy character. Replace his human facial features with oversized, wide-set eyes and a mischievous grin showcasing rows of jagged, pointed teeth. His skin and the fabric of the maroon hoodie must take on a stylized texture, blending soft plush fur with smooth, semi-glossy vinyl. As he flips through the pages of the books, ensure the natural light from the large windows glints off his vinyl surfaces and highlights the fuzzy texture of his ears. The character must remain integrated with the realistic wooden bookshelves and the textures of the library environment." },
      { src: "static/videos/comparison/case1/ours/edit_2_simpsons%20comic.mp4", instruction: "Simpsons comic",
        instructionFull: "Transform the young man in the maroon hoodie into a classic Simpsons-style cartoon character. His skin must be changed to a vibrant yellow, and his facial features simplified with large, round eyes and bold black outlines. The maroon hoodie and its logo should become flat, two-dimensional shapes with thick, dark borders. Extend this restyling to the surrounding environment; the tall wooden bookshelves and the multicolored books must lose their realistic textures, replaced by solid blocks of color and heavy line art. Convert the natural light streaming from the windows into flat, cel-shaded highlights that fall across the character and the polished wood, completing the 2D comic transformation." },
      { src: "static/videos/comparison/case1/ours/edit_3_cartoon%20doll.mp4", instruction: "3D cartoon doll",
        instructionFull: "Transform the young man in the maroon hoodie into a stylized 3D cartoon doll. Replace his facial features with soft, rounded contours and large, glossy eyes to create a toy-like appearance. The maroon hoodie should be re-rendered with a soft, felt-like texture while maintaining the visible logo on the chest. Reshape his hands in the close-up shots into simplified, rounded doll hands with a smooth, matte plastic finish as he flips through the pages. Ensure his movements against the tall wooden bookshelves and the natural light from the windows maintain this consistent 3D doll aesthetic, contrasting his toy-like materials with the polished wood and textured book covers." },
    ],
  },

  // ---- COMPARISON : comparison/case2 — same camera-move chain (zoom out -> move up -> zoom in)
  //      applied by our method vs. baselines, grouped by method category (tagLabel).
  //      Prompt-switch baselines are pure T2V (no visual input): they get a scene-prompt
  //      block instead of a source clip and are flagged with a dagger (†).
  {
    tag: "compare",
    tagLabel: "Streaming Edit",
    title: "Ours",
    steps: [
      { src: "static/videos/comparison/case2/ours/source.mp4", label: "Source" },
      { src: "static/videos/comparison/case2/ours/edit_1_zoom%20out.mp4", instruction: "Zoom out",
        instructionFull: "Initiate a smooth zoom-out transition starting from a close-up that focuses on the intricate texture of the tree bark and the ruggedness of the mountain cliffs. Gradually pull the camera back to broaden the frame, revealing the dense forests filled with deep oranges and bright yellows. As the field of view expands, encompass the calm lake and its mirror-like surface reflecting the golden-hued trees. Continue the zoom until the entire expansive landscape is visible, showing the towering mountains against the clear sky with soft, diffused light, effectively transitioning from a detailed focus on the rocky terrain to a comprehensive wide-angle shot of the tranquil autumnal environment." },
      { src: "static/videos/comparison/case2/ours/edit_2_move%20up.mp4", instruction: "Move up",
        instructionFull: "Begin the shot focused on the calm lake surface where the golden trees and rugged mountains are reflected perfectly. Execute a smooth crane movement upwards, transitioning from the mirror-like water to the actual shoreline. As the camera ascends, capture the vivid orange and yellow foliage of the dense forests and the detailed texture of the nearby tree bark. Continue the upward tilt past the rugged mountain cliffs, eventually revealing the expansive clear sky and the soft, diffused light of the late afternoon. This vertical shift must move from the lower reflections of the terrain to the higher, majestic peaks and the open atmosphere above." },
      { src: "static/videos/comparison/case2/ours/edit_3_zoom%20in.mp4", instruction: "Zoom in",
        instructionFull: "Starting from the wide-angle view of the serene autumnal landscape, initiate a smooth and gradual zoom-in toward the dense forest lining the lake's edge. As the camera moves closer, shift the focus from the expansive panoramic view to the specific textures of the golden-hued tree bark and the ruggedness of the mountain cliffs towering above. The frame should tighten to a medium shot, prominently featuring the deep oranges and bright yellows of the autumn foliage. Throughout this transformation, maintain the alignment of the mirror-like reflection on the still lake surface, ensuring the transition captures the fine details of the terrain while preserving the soft, diffused lighting of the clear sky." },
    ],
  },
  {
    tag: "compare",
    tagLabel: "Pure Backbone",
    title: "Helios (Base)",
    steps: [
      { src: "static/videos/comparison/case2/Helios_Base/source.mp4", label: "Source" },
      { src: "static/videos/comparison/case2/Helios_Base/edit_1_zoom%20out.mp4", instruction: "Zoom out" },
      { src: "static/videos/comparison/case2/Helios_Base/edit_2_move%20up.mp4", instruction: "Move up" },
      { src: "static/videos/comparison/case2/Helios_Base/edit_3_zoom%20in.mp4", instruction: "Zoom in" },
    ],
  },
  {
    tag: "compare",
    tagLabel: "Prompt Switch",
    title: "Anchor Forcing †",
    steps: [
      { scene: "Autumn lake — golden trees, mirror-like water, mountains", label: "Scene prompt (T2V)",
        sceneFull: "The video showcases a serene autumnal landscape featuring a calm lake surrounded by golden trees and towering mountains. The scene opens with a wide-angle view, capturing the essence of nature's tranquility. The lake mirrors the surrounding environment flawlessly, creating a mirror-like effect that enhances the tranquil beauty of the scene. Golden-hued trees line the shores, their colors vivid against the rugged mountains. The camera slowly pans across the landscape, highlighting intricate details such as the texture of the tree bark and the ruggedness of the mountain cliffs. Smooth transitions and gentle zooms maintain a consistent pace, allowing viewers to fully absorb the beauty of the scene. The background features a clear sky with soft, diffused light, suggesting either early morning or late afternoon. The terrain is a mix of rocky cliffs and dense forests, with the trees displaying a range of autumn colors from deep oranges to bright yellows. The lake is calm and still, providing a perfect reflective surface for the surrounding scenery. The video has a documentary style, using natural lighting and minimalistic composition to enhance the realism and authenticity of the scene, with smooth, deliberate camera movement and a steady, calming rhythm." },
      { src: "static/videos/comparison/case2/Anchor%20Forcing/edit_1_zoom%20out.mp4", instruction: "Zoom out" },
      { src: "static/videos/comparison/case2/Anchor%20Forcing/edit_2_move%20up.mp4", instruction: "Move up" },
      { src: "static/videos/comparison/case2/Anchor%20Forcing/edit_3_zoom%20in.mp4", instruction: "Zoom in" },
    ],
  },
  {
    tag: "compare",
    tagLabel: "Prompt Switch",
    title: "Infinity-RoPE †",
    steps: [
      { scene: "Autumn lake — golden trees, mirror-like water, mountains", label: "Scene prompt (T2V)",
        sceneFull: "The video showcases a serene autumnal landscape featuring a calm lake surrounded by golden trees and towering mountains. The scene opens with a wide-angle view, capturing the essence of nature's tranquility. The lake mirrors the surrounding environment flawlessly, creating a mirror-like effect that enhances the tranquil beauty of the scene. Golden-hued trees line the shores, their colors vivid against the rugged mountains. The camera slowly pans across the landscape, highlighting intricate details such as the texture of the tree bark and the ruggedness of the mountain cliffs. Smooth transitions and gentle zooms maintain a consistent pace, allowing viewers to fully absorb the beauty of the scene. The background features a clear sky with soft, diffused light, suggesting either early morning or late afternoon. The terrain is a mix of rocky cliffs and dense forests, with the trees displaying a range of autumn colors from deep oranges to bright yellows. The lake is calm and still, providing a perfect reflective surface for the surrounding scenery. The video has a documentary style, using natural lighting and minimalistic composition to enhance the realism and authenticity of the scene, with smooth, deliberate camera movement and a steady, calming rhythm." },
      { src: "static/videos/comparison/case2/InfinityRoPE/edit_1_zoom%20out.mp4", instruction: "Zoom out" },
      { src: "static/videos/comparison/case2/InfinityRoPE/edit_2_move%20up.mp4", instruction: "Move up" },
      { src: "static/videos/comparison/case2/InfinityRoPE/edit_3_zoom%20in.mp4", instruction: "Zoom in" },
    ],
  },
  {
    tag: "compare",
    tagLabel: "In-Place Edit",
    title: "Lucy-Edit",
    steps: [
      { src: "static/videos/comparison/case2/ours/source.mp4", label: "Source" },
      { src: "static/videos/comparison/case2/LucyEdit/lucy_edit_1_zoom%20out.mp4", instruction: "Zoom out" },
      { src: "static/videos/comparison/case2/LucyEdit/lucy_edit_2_move%20up.mp4", instruction: "Move up" },
      { src: "static/videos/comparison/case2/LucyEdit/lucy_edit_3_zoom%20in.mp4", instruction: "Zoom in" },
    ],
  },
  {
    tag: "compare",
    tagLabel: "In-Place Edit",
    title: "SANA-Streaming",
    steps: [
      { src: "static/videos/comparison/case2/ours/source.mp4", label: "Source" },
      { src: "static/videos/comparison/case2/SANA-Streaming/sana_edit_1_zoom%20out.mp4", instruction: "Zoom out" },
      { src: "static/videos/comparison/case2/SANA-Streaming/sana_edit_2_move%20up.mp4", instruction: "Move up" },
      { src: "static/videos/comparison/case2/SANA-Streaming/sana_edit_3_zoom%20in.mp4", instruction: "Zoom in" },
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
        title: "Simpsons Comic",
        source: "static/videos/single-edit/style%20transfer/simpsons%20comic/src.mp4",
        edited: "static/videos/single-edit/style%20transfer/simpsons%20comic/out.mp4",
        instruction: "Simpsons comic style",
        instructionFull:
          "Transform the woman and the partially visible man into Simpsons-style characters by applying bright yellow skin tones and thick, black outlines to all figures. Convert the woman’s vibrant red coat, white sweater, and gold hoop earrings into flat, saturated 2D shapes. The man’s beige jacket and blue hood must be rendered with bold borders and simplified color fills. Flatten the background architectural elements and blurred bokeh lights into stylized, two-dimensional circles of light and simple geometric forms. Ensure the woman’s animated expressions are translated into a flat animation style, emphasizing her large eyes and specific accessories within a bold, comic-inspired aesthetic.",
      },
      {
        title: "American Comic",
        source: "static/videos/single-edit/style%20transfer/american%20comic/src.mp4",
        edited: "static/videos/single-edit/style%20transfer/american%20comic/out.mp4",
        instruction: "American comic style",
        instructionFull:
          "Transform this urban aerial footage into a classic American comic book aesthetic. Apply heavy, bold black ink outlines to the historic steeples, intricate facades, and the modern structures under construction with their cranes. Replace the natural midday lighting with flat, vivid primary colors and introduce prominent halftone dot shading across the meandering river's surface. The pedestrians and vehicles navigating the bustling streets should feature dramatic, high-contrast shadows to enhance their dynamic movement. Simplify the birds flying overhead into sharp graphic silhouettes. Every surface, from the sidewalks to the water, must be rendered using traditional comic book textures like cross-hatching and uniform dot patterns to create a hand-drawn, high-impact visual style.",
      },
    ],
  },
  {
    category: "Entity Transformation",
    cases: [
      {
        title: "Labubu Toy",
        source: "static/videos/single-edit/entity%20transformation/ladudu/src.mp4",
        edited: "static/videos/single-edit/entity%20transformation/ladudu/out.mp4",
        instruction: "Turn the man into a Labubu toy",
        instructionFull:
          "Transform the man into Labubu-style designer toy while maintaining his current attire. The man's gray and white raglan shirt and beige pants should now cover a rounded, plush-vinyl body. Replace his face with large, expressive eyes and a wide mouth featuring jagged teeth, while he continues to hold the red cup. Ensure the golden sunlight interacts with the new vinyl-like surfaces of their skin and features.",
      },
      {
        title: "3D Cartoon Doll",
        source: "static/videos/single-edit/entity%20transformation/cartoon%20doll/src.mp4",
        edited: "static/videos/single-edit/entity%20transformation/cartoon%20doll/out.mp4",
        instruction: "Turn the family into cartoon dolls",
        instructionFull:
          "Transform the four family members into 3D cartoon dolls with smooth, plastic-textured skin and exaggeratedly large, glossy eyes. The father in his blue shirt, the mother in her striped sweater, the child in the cowboy hat, and the child in red must retain their original poses and clothing styles while taking on a miniature toy-like appearance. Simplify their facial features into cute, rounded proportions, ensuring the father's calm expression and the mother's attentive look are captured in this new aesthetic. The fluffy brown and white dog and the sunlit car interior should remain realistic, creating a contrast between the cartoon characters and the natural greenery visible through the windows.",
      },
    ],
  },
  // ---- Motion Transfer: no video uploaded yet. Add one under
  //      static/videos/single-edit/ and uncomment this block.
  // {
  //   category: "Motion Transfer",
  //   cases: [ { title: "...", source: "...", edited: "...", instruction: "...", instructionFull: "..." } ],
  // },
  {
    category: "Camera Movement Control",
    cases: [
      {
        title: "Zoom In",
        source: "static/videos/single-edit/camera%20movement/zoom%20in/src.mp4",
        edited: "static/videos/single-edit/camera%20movement/zoom%20in/out.mp4",
        instruction: "Zoom in on the temple",
        instructionFull:
          "Apply a smooth zoom-in on the central portion of the Luxor Temple complex. Starting from the wide angle that includes the Nile River and modern buildings in the background, progressively magnify the frame toward the massive stone structures and ancient columns. As the edges of the scene, including the peripheral palm trees and distant skyline, fall out of view, the intricate details of the temple walls and the weathered textures of the stone remnants should become increasingly prominent. The final frame should focus closely on the vertical columns, with natural sunlight highlighting their form and the shadows accentuating the depth of the ancient masonry.",
      },
      {
        title: "Move Down",
        source: "static/videos/single-edit/camera%20movement/move%20down/src.mp4",
        edited: "static/videos/single-edit/camera%20movement/move%20down/out.mp4",
        instruction: "Tilt down to the shore",
        instructionFull:
          "Starting with the upper third of the frame showcasing the soft purple and pink twilight sky above the modern glass house, simulate a steady downward tilt that moves the sky out of the frame while bringing the grassy cliff into more prominent view. The movement should transition from the large glass windows and flowing curtains down to the base of the cliff. Conclude the shot by focusing on the lower portion of the scene, specifically where the gentle waves lap against the shore and the house's reflection is clearly visible in the calm, deep blue sea.",
      },
    ],
  },
];

/* --------------------------- renderer --------------------------- */

function makeVideo(step) {
  const v = document.createElement("video");
  v.src = step.src;
  if (step.poster) v.poster = step.poster; // otherwise the browser shows the first frame
  v.controls = true;                        // progress bar + play/seek
  v.playsInline = true;
  v.setAttribute("playsinline", "");
  v.preload = "metadata";                   // load enough to show duration + first frame
  return v;
}

/* A "scene" step has no video: it stands in for the source of a pure-T2V
 * (prompt-switch) baseline, showing a brief scene description with the full
 * scene prompt revealed on hover/tap. */
function makeSceneCard(step) {
  const card = document.createElement("div");
  card.className = "vid-card scene-card";
  const body = document.createElement("div");
  body.className = "scene-body";
  body.appendChild(makeInstr(step.scene, step.sceneFull));
  card.appendChild(body);
  const label = document.createElement("div");
  label.className = "vid-label source";
  label.innerHTML = step.label || "Scene prompt (T2V)";
  card.appendChild(label);
  return card;
}

function makeCard(step, idx) {
  if (step.scene && !step.src) return makeSceneCard(step);
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
  wrap.className = "edit-chain" + (chain.tag ? ` chain-${chain.tag}` : "");

  if (chain.title) {
    const h = document.createElement("div");
    h.className = "chain-title";
    const tagLabel = chain.tagLabel || (chain.tag === "compare" ? "METHOD" : "CHAIN");
    h.innerHTML = `<span class="chain-tag">${tagLabel}</span>${chain.title}`;
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
  renderShortEdits("short-edits");
  mount("longgen", "chains-longgen");
  mount("infinite", "chains-infinite");
  mount("compare", "chains-compare");
  // tap anywhere else closes any open instruction popover
  document.addEventListener("click", () =>
    document.querySelectorAll(".instr.open").forEach((el) => el.classList.remove("open"))
  );
});
