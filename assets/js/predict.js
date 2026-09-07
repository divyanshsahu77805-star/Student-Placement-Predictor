/* ==========================================================================
   Runs the exported XGBoost model (as ONNX) entirely in the browser.
   No server, no network call after the initial asset load.
   ========================================================================== */

const DISPLAY_NAMES = {
  study_hours_per_day: "Study hours / day",
  projects_count: "Projects completed",
  internships: "Internship experience",
  cgpa: "CGPA",
  mock_interviews: "Mock interviews",
  communication_score: "Communication score",
  resume_score: "Resume score",
  hackathons_participated: "Hackathons",
  academic_efficiency: "Academic efficiency (CGPA × study hours)",
  practical_experience: "Practical experience (projects + internships)",
  dsa_overall_score: "DSA overall score",
};

let session = null;
let scalerInfo = null;
let featureImportance = null;
let internshipVal = 0;

const $ = (id) => document.getElementById(id);

// -- Wire up slider labels --
const sliderPairs = [
  ["cgpa", "cgpa-val", (v) => v.toFixed(1)],
  ["study_hours", "study_hours-val", (v) => v.toFixed(1)],
  ["dsa_hours", "dsa_hours-val", (v) => v.toFixed(1)],
  ["dsa_problems", "dsa_problems-val", (v) => Math.round(v)],
  ["projects", "projects-val", (v) => Math.round(v)],
  ["mock", "mock-val", (v) => Math.round(v)],
  ["comm", "comm-val", (v) => v.toFixed(1)],
  ["resume", "resume-val", (v) => v.toFixed(1)],
  ["hackathons", "hackathons-val", (v) => Math.round(v)],
];

function wireSliders() {
  sliderPairs.forEach(([inputId, labelId, fmt]) => {
    const input = $(inputId);
    const label = $(labelId);
    input.addEventListener("input", () => {
      label.textContent = fmt(parseFloat(input.value));
      runPrediction();
    });
  });

  const toggle = $("internship-toggle");
  toggle.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      toggle.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      internshipVal = parseInt(btn.dataset.val, 10);
      runPrediction();
    });
  });
}

function readRawInputs() {
  return {
    cgpa: parseFloat($("cgpa").value),
    study_hours_per_day: parseFloat($("study_hours").value),
    dsa_hours_per_week: parseFloat($("dsa_hours").value),
    dsa_problems_solved: parseFloat($("dsa_problems").value),
    projects_count: parseFloat($("projects").value),
    internships: internshipVal,
    mock_interviews: parseFloat($("mock").value),
    communication_score: parseFloat($("comm").value),
    resume_score: parseFloat($("resume").value),
    hackathons_participated: parseFloat($("hackathons").value),
  };
}

function engineerFeatures(raw) {
  return {
    ...raw,
    academic_efficiency: raw.cgpa * raw.study_hours_per_day,
    practical_experience: raw.projects_count + raw.internships * 2,
    dsa_overall_score: (raw.dsa_hours_per_week * raw.dsa_problems_solved) / 100,
  };
}

function standardize(features) {
  const { feature_order, mean, scale } = scalerInfo;
  return feature_order.map((f, i) => (features[f] - mean[i]) / scale[i]);
}

async function loadAssets() {
  // Run single-threaded: GitHub Pages doesn't send the cross-origin-isolation
  // headers WASM threads need, so we pin this explicitly rather than let the
  // runtime silently try (and fail) to spin up a worker pool.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL("assets/js/ort/", document.baseURI).href;

  const [scalerRes, fiRes] = await Promise.all([
    fetch("model/scaler.json"),
    fetch("model/feature_importance.json"),
  ]);
  scalerInfo = await scalerRes.json();
  featureImportance = await fiRes.json();

  session = await ort.InferenceSession.create("model/model.onnx");

  $("result-empty").style.display = "none";
  $("result-body").style.display = "block";
  runPrediction();
}

function topDrivers(standardized) {
  const { feature_order } = scalerInfo;
  const contributions = feature_order.map((f, i) => {
    const fi = featureImportance[f];
    const contribution = fi.importance * standardized[i] * fi.corr_sign;
    return { name: f, contribution, standardized: standardized[i] };
  });
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return contributions.slice(0, 5);
}

async function runPrediction() {
  if (!session) return;

  const raw = readRawInputs();
  const engineered = engineerFeatures(raw);
  const standardized = standardize(engineered);

  const inputTensor = new ort.Tensor(
    "float32",
    Float32Array.from(standardized),
    [1, standardized.length]
  );

  const feeds = { float_input: inputTensor };
  const results = await session.run(feeds);
  const probs = results.probabilities.data; // [P(not placed), P(placed)]
  const pPlaced = probs[1];

  renderResult(pPlaced, topDrivers(standardized));
}

function renderResult(pPlaced, drivers) {
  const pct = Math.round(pPlaced * 100);
  const label = pPlaced >= 0.5 ? "Likely placed" : "At risk";

  $("verdict-label").textContent = label;
  $("verdict-label").style.color = pPlaced >= 0.5 ? "var(--green-light)" : "var(--red)";
  $("verdict-pct").textContent = `${pct}% placement probability`;
  $("result-fill").style.width = `${pct}%`;

  const list = $("driver-list");
  list.innerHTML = "";
  drivers.forEach((d) => {
    const li = document.createElement("li");
    const positive = d.contribution >= 0;
    li.innerHTML = `
      <span class="dname">${DISPLAY_NAMES[d.name] || d.name}</span>
      <span class="dval ${positive ? "pos" : "neg"}">${positive ? "▲" : "▼"} ${
      d.standardized >= 0 ? "above avg" : "below avg"
    }</span>`;
    list.appendChild(li);
  });
}

wireSliders();
loadAssets().catch((err) => {
  console.error(err);
  $("result-empty").textContent =
    "Couldn't load the model in this browser. Try a recent Chrome, Firefox, or Safari.";
});
