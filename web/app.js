/* Mombe field app: one screen, one action. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const app = $("#app");
const store = {
  get key() { try { return localStorage.getItem("mombe.key") || ""; } catch { return ""; } },
  set key(v) { try { localStorage.setItem("mombe.key", v); } catch { /* private mode */ } },
};

const LABELS = {
  start: "Session opened", resume: "Picked up the new evidence",
  assess_capture: "Checked the photo", identify_muzzle: "Read the noseprint", lookup_animal: "Looked up the register",
  check_continuity: "Checked the take is unbroken", calibrate_scale: "Fixed scale from the board",
  measure_body: "Measured the body", estimate_weight: "Estimated weight", value_animal: "Valued the animal",
  request_capture: "Asked for a new photo", request_approval: "Sent to a person for approval",
  write_registry: "Wrote the record", flag_fraud: "Flagged for investigation", finish: "Finished",
};
const TASKS = [
  ["verify", "Verify", "Who is this animal"],
  ["enrol", "Enrol", "Register a new animal"],
  ["value", "Value", "Weight and value for a loan"],
  ["claim", "Claim", "Insurance claim check"],
];

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { "x-mombe-key": store.key, ...(opts.headers || {}) } });
  if (r.status === 401) { location.hash = "#/key"; throw new Error("key"); }
  return r.json();
}

/* ---- noseprint motif ---------------------------------------------------- */
function drawPrint() {
  const svg = $("#print");
  const w = innerWidth, h = innerHeight;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const cx = w * 0.82, cy = h * 0.86, R = Math.max(w, h) * 0.55;
  let out = "";
  for (let i = 0; i < 420; i++) {
    const a = rnd() * Math.PI * 2, d = Math.sqrt(rnd()) * R;
    const x = cx + Math.cos(a) * d, y = cy + Math.sin(a) * d * 0.8;
    const r = 5 + rnd() * 13 * (1 - d / R * 0.6);
    const rx = r * (0.75 + rnd() * 0.5), ry = r * (0.75 + rnd() * 0.5), rot = rnd() * 180;
    out += `<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="${rx.toFixed(1)}" ry="${ry.toFixed(1)}" transform="rotate(${rot.toFixed(0)} ${x.toFixed(1)} ${y.toFixed(1)})"/>`;
  }
  svg.innerHTML = `<g fill="none" stroke="#b4501c" stroke-opacity=".16" stroke-width="1.2">${out}</g>`;
}

/* ---- screens ------------------------------------------------------------ */
function header(right = "") {
  return `<div class="top"><div class="wordmark">Mom<b>be</b></div>${right}</div>`;
}

function keyScreen() {
  app.innerHTML = `${header()}
    <h1>Officer passcode.</h1>
    <p class="lede">Enter the code your supervisor issued.</p>
    <input id="k" type="password" autocomplete="off" placeholder="Passcode">
    <button class="go" id="save">Continue</button>`;
  $("#save").onclick = () => { store.key = $("#k").value.trim(); location.hash = "#/"; };
}

function startScreen() {
  let task = "verify";
  app.innerHTML = `${header('<a href="#/approve">Approvals</a>')}
    <h1>Every animal carries a signature.</h1>
    <p class="lede">Its nose. Mombe reads it, weighs the body and puts a value on it.</p>
    <div class="tasks">${TASKS.map(([k, t, s]) => `<button class="task" data-k="${k}" aria-pressed="${k === task}">${t}<small>${s}</small></button>`).join("")}</div>
    <div id="fields"></div>
    <button class="go" id="begin">Begin</button>
    <div class="foot">OpenCV 5 on AWS Graviton · Claude on Amazon Bedrock</div>`;
  const fields = () => {
    const f = [`<label>Owner</label><input id="owner" placeholder="Owner's name">`];
    if (task === "value") f.push(`<label>Loan requested (USD)</label><input id="loan" inputmode="decimal" placeholder="500">`);
    if (task === "claim") f.push(`<label>Insured animal</label><input id="insured" placeholder="Animal ID on the policy">`);
    $("#fields").innerHTML = f.join("");
  };
  fields();
  app.querySelectorAll(".task").forEach(b => b.onclick = () => {
    task = b.dataset.k;
    app.querySelectorAll(".task").forEach(x => x.setAttribute("aria-pressed", x === b));
    fields();
  });
  $("#begin").onclick = async () => {
    const body = { task, claimed_owner: $("#owner")?.value || null };
    if ($("#loan")) body.loan_threshold_usd = parseFloat($("#loan").value) || null;
    if ($("#insured")) body.policy_animal = $("#insured").value || null;
    const s = await api("/api/sessions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    location.hash = `#/s/${s.id}/capture`;
  };
}

let stream = null;
async function camera(video) {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1920 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    return true;
  } catch { return false; }
}
function stopCamera() { stream?.getTracks().forEach(t => t.stop()); stream = null; }
function grab(video) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return new Promise(r => c.toBlob(r, "image/jpeg", 0.9));
}

async function captureScreen(sid) {
  const s = await api(`/api/sessions/${sid}`);
  const last = s.instructions[s.instructions.length - 1];
  const needBody = s.task === "value";
  const shots = [], body = [];
  app.innerHTML = `${header()}
    <h2>${last ? last.message : "Fill the oval with the nose."}</h2>
    <div class="finder"><video playsinline muted></video><div class="oval"></div><div class="hint" id="hint">Two photos, nose filling the oval</div></div>
    <div class="shots" id="shots"></div>
    <input type="file" id="file" accept="image/*" capture="environment" multiple hidden>
    <button class="go" id="snap">Take photo</button>
    <button class="ghost" id="pick">Choose photos instead</button>`;
  const video = $("video");
  const live = await camera(video);
  if (!live) $("#snap").textContent = "Open camera";
  const addShot = blob => { shots.push(blob); $("#shots").insertAdjacentHTML("beforeend", `<img src="${URL.createObjectURL(blob)}">`); };
  const done = async () => {
    if (needBody) return walkScreen(sid, shots);
    await send(sid, shots, body);
  };
  $("#snap").onclick = async () => {
    if (!live) return $("#file").click();
    addShot(await grab(video));
    if (shots.length >= 2) { stopCamera(); done(); }
    else $("#hint").textContent = "One more, slightly to the side";
  };
  $("#pick").onclick = () => $("#file").click();
  $("#file").onchange = e => { [...e.target.files].slice(0, 4).forEach(addShot); stopCamera(); done(); };
}

async function walkScreen(sid, shots) {
  const body = [];
  app.innerHTML = `${header()}
    <h2>Walk slowly around the animal.</h2>
    <p class="lede">Board flat against the shoulder. Keep the whole body in view.</p>
    <div class="finder"><video playsinline muted></video><div class="hint" id="hint">Ready</div></div>
    <button class="go" id="rec">Start walk-around</button>`;
  const video = $("video");
  await camera(video);
  let timer = null;
  $("#rec").onclick = async () => {
    if (timer) { clearInterval(timer); stopCamera(); return send(sid, shots, body); }
    $("#rec").textContent = "Finish";
    timer = setInterval(async () => {
      body.push(await grab(video));
      $("#hint").textContent = `${body.length} frames`;
      if (body.length >= 80) $("#rec").click();
    }, 500);
  };
}

async function send(sid, muzzle, body) {
  const fd = new FormData();
  muzzle.forEach((b, i) => fd.append(`muzzle_${i}`, b, `m${i}.jpg`));
  body.forEach((b, i) => fd.append(`body_${i}`, b, `b${i}.jpg`));
  fd.append("meta", JSON.stringify({ device: navigator.userAgent.slice(0, 80) }));
  await api(`/api/sessions/${sid}/captures`, { method: "POST", body: fd });
  location.hash = `#/s/${sid}/think`;
}

function stepItem(e) {
  if (!["observation", "action", "refused", "start", "resume"].includes(e.kind)) return "";
  const r = e.result || {};
  let meta = "";
  if (e.name === "assess_capture") meta = r.issues?.length ? `issues: ${r.issues.join(", ")}` : "clear";
  if (e.name === "identify_muzzle") meta = `${r.decision}${r.animal_id ? " · " + r.animal_id : ""} · ${r.score} verified points`;
  if (e.name === "estimate_weight") meta = `${r.lo}–${r.hi} kg`;
  if (e.name === "value_animal") meta = `$${r.lo}–$${r.hi}`;
  if (e.name === "calibrate_scale") meta = `scale error ${(100 * (r.scale_rel_err || 0)).toFixed(1)}%`;
  if (e.kind === "refused") meta = e.reason;
  const next = e.changed_next_action && e.required_after ? ` → next: ${LABELS[e.required_after] || e.required_after}` : "";
  const cls = e.kind === "refused" ? "refused" : (e.changed_next_action ? "changed" : "");
  return `<li class="${cls}"><span></span><div>${LABELS[e.name || e.kind] || e.name}<div class="meta">${meta}${next}</div></div></li>`;
}

let poller = null;
async function thinkScreen(sid) {
  app.innerHTML = `${header()}<div class="pulse"></div><h2 style="text-align:center">Mombe is looking.</h2><ul class="steps" id="steps"></ul>`;
  const seenInstr = (await api(`/api/sessions/${sid}`)).instructions.length;
  const tick = async () => {
    const [s, tr] = await Promise.all([api(`/api/sessions/${sid}`), api(`/api/sessions/${sid}/trace`)]);
    $("#steps") && ($("#steps").innerHTML = tr.map(stepItem).join(""));
    if (s.status === "waiting" && s.instructions.length > seenInstr) return go(`#/s/${sid}/instruct`);
    if (s.status === "waiting" || s.status === "done") return go(`#/s/${sid}/result`);
  };
  poller = setInterval(tick, 1500);
  tick();
}
function go(h) { clearInterval(poller); location.hash = h; }

async function instructScreen(sid) {
  const s = await api(`/api/sessions/${sid}`);
  const ins = s.instructions[s.instructions.length - 1];
  const [en, sn] = ins.message.split(" / ");
  app.innerHTML = `${header()}
    <div class="instruction">${en}${sn ? `<span class="sn">${sn}</span>` : ""}</div>
    <p class="why">The evidence fell short at <b>${(ins.gate || "").replace("_", " ")}</b>. One more capture settles it.</p>
    <button class="go" id="again">Capture again</button>
    <button class="ghost" onclick="location.hash='#/trace/${sid}'">See the reasoning</button>`;
  $("#again").onclick = () => location.hash = `#/s/${sid}/capture`;
}

async function resultScreen(sid) {
  const s = await api(`/api/sessions/${sid}`);
  const L = s.ledger || {};
  const id = L.identity || {};
  const w = L.weight, v = L.value, ap = L.approval;
  const outcome = (await api(`/api/sessions/${sid}/trace`)).filter(e => e.kind === "action" && e.name === "finish").pop()?.result?.outcome
    || (ap && ap.status === "pending" ? "awaiting approval" : s.status);
  const ev = id.evidence_image ? `/api/sessions/${sid}/artifacts/${id.evidence_image.split(/[\\/]/).pop()}` : null;
  app.innerHTML = `${header()}
    <h2>${{ verified: "Identity confirmed.", enrolled: "Enrolled.", approved: "Approved.", escalated: "Sent for investigation.", not_registered: "Not in the register." }[outcome] || "Waiting on a person."}</h2>
    <div class="card">
      <div class="big">${id.animal_id || (id.decision === "new" ? "New animal" : "—")}</div>
      <span class="badge ${outcome === "approved" || outcome === "verified" || outcome === "enrolled" ? "ok" : ""}">${outcome}</span>
      ${L.registry?.owner ? `<div class="row"><span>Registered owner</span><span>${L.registry.owner}</span></div>` : ""}
      ${id.score != null ? `<div class="row"><span>Verified points</span><span>${id.score}</span></div>` : ""}
      ${w ? `<div class="row"><span>Weight, 90%</span><span>${w.lo}–${w.hi} kg</span></div>` : ""}
      ${v ? `<div class="row"><span>Value</span><span>$${v.lo}–$${v.hi}</span></div>` : ""}
      ${ap ? `<div class="row"><span>${ap.kind.replace("_", " ")}</span><span>${ap.status}</span></div>` : ""}
      ${ev ? `<img class="evidence" src="${ev}" alt="Matched noseprint points">` : ""}
    </div>
    <button class="ghost" onclick="location.hash='#/trace/${sid}'">See the reasoning</button>
    <button class="go" onclick="location.hash='#/'">Next animal</button>`;
  if (ev) $(".evidence").src = await authImg(ev);
}

async function authImg(path) {
  const r = await fetch(path, { headers: { "x-mombe-key": store.key } });
  return URL.createObjectURL(await r.blob());
}

async function traceScreen(sid) {
  const tr = await api(`/api/sessions/${sid}/trace`);
  const changes = tr.filter(e => e.changed_next_action).length;
  app.innerHTML = `${header(`<a href="#/s/${sid}/result">Back</a>`)}
    <h2>How Mombe decided.</h2>
    <p class="lede">${changes} times a vision result changed what Mombe did next. Red dots mark them.</p>
    <ul class="steps">${tr.map(stepItem).join("")}</ul>`;
}

async function approveScreen() {
  const items = await api("/api/approvals?status=pending");
  app.innerHTML = `${header('<a href="#/">Field</a>')}
    <h2>${items.length ? "Waiting for your decision." : "Nothing waiting."}</h2>
    <div class="approvals">${items.map(a => `
      <div class="card" data-id="${a.id}">
        <h3>${a.kind.replace("_", " ")} · ${a.summary?.identity?.animal_id || "unidentified"}</h3>
        ${a.summary?.weight ? `<div class="row"><span>Weight</span><span>${a.summary.weight.lo}–${a.summary.weight.hi} kg</span></div>` : ""}
        ${a.summary?.value ? `<div class="row"><span>Value</span><span>$${a.summary.value.lo}–$${a.summary.value.hi}</span></div>` : ""}
        ${a.summary?.offer_cap_usd ? `<div class="row"><span>Offer capped at</span><span>$${a.summary.offer_cap_usd}</span></div>` : ""}
        ${a.summary?.reason ? `<p class="why">${a.summary.reason}</p>` : ""}
        ${a.summary?.agent_note ? `<p class="why">${a.summary.agent_note}</p>` : ""}
        <div class="split"><button class="ghost" data-d="declined">Decline</button><button class="go" data-d="approved">Approve</button></div>
      </div>`).join("")}</div>`;
  app.querySelectorAll("[data-d]").forEach(b => b.onclick = async () => {
    const id = b.closest("[data-id]").dataset.id;
    await api(`/api/approvals/${id}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ status: b.dataset.d, by: "approver" }) });
    approveScreen();
  });
}

/* ---- router ------------------------------------------------------------- */
function route() {
  clearInterval(poller);
  stopCamera();
  const h = location.hash.slice(2).split("/");
  if (!store.key && h[0] !== "key") return keyScreen();
  if (h[0] === "key") return keyScreen();
  if (h[0] === "approve") return approveScreen();
  if (h[0] === "trace") return traceScreen(h[1]);
  if (h[0] === "s") return ({ capture: captureScreen, think: thinkScreen, instruct: instructScreen, result: resultScreen }[h[2]] || resultScreen)(h[1]);
  startScreen();
}
addEventListener("hashchange", route);
addEventListener("resize", drawPrint);
drawPrint();
route();
