/* Mombe field app: one screen, one action. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const app = $("#app");
const store = {
  get key() { try { return localStorage.getItem("mombe.key") || ""; } catch { return ""; } },
  set key(v) { try { localStorage.setItem("mombe.key", v); } catch { /* private mode */ } },
};
// The approver code lives for the tab only: a shared phone forgets it when the tab closes.
const approver = {
  get code() { try { return sessionStorage.getItem("mombe.approver") || ""; } catch { return ""; } },
  set code(v) { try { v ? sessionStorage.setItem("mombe.approver", v) : sessionStorage.removeItem("mombe.approver"); } catch { /* private mode */ } },
};

const LABELS = {
  start: "Session opened", resume: "Picked up the new evidence",
  assess_capture: "Checked the photo", check_liveness: "Checked the nose is live",
  identify_muzzle: "Read the noseprint", lookup_animal: "Looked up the register",
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
    <p class="lede">Boards hung on both shoulders, helper out of frame. Keep the whole body in view.</p>
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
  if (e.name === "check_liveness") meta = r.live ? "live" : `held back: ${r.failed_cue || "liveness"}`;
  if (e.name === "identify_muzzle") meta = `${r.decision}${r.animal_id ? " · " + r.animal_id : ""} · ${r.score} verified points`;
  if (e.name === "estimate_weight") meta = `${r.lo} to ${r.hi} kg`;
  if (e.name === "value_animal") meta = r.p10_usd != null ? `P10 $${r.p10_usd} · valid to ${r.expires}` : `$${r.lo} to $${r.hi}`;
  if (e.name === "calibrate_scale") meta = `scale error ${(100 * (r.scale_rel_err || 0)).toFixed(1)}%`;
  if (e.kind === "refused") meta = e.reason;
  let label = LABELS[e.name || e.kind] || e.name;
  if (e.name === "finish") {
    const o = r.outcome || "";
    label = { awaiting_capture: "Waiting for the new photo", awaiting_approval: "Waiting for approval", awaiting_review: "Waiting for a reviewer" }[o] || `Finished: ${o.replace("_", " ")}`;
  }
  const visionChange = e.kind === "observation" && e.changed_next_action;
  const next = visionChange && e.required_after ? ` → next: ${LABELS[e.required_after] || e.required_after}` : "";
  const cls = e.kind === "refused" ? "refused" : (visionChange ? "changed" : "");
  return `<li class="${cls}"><span></span><div>${label}<div class="meta">${meta}${next}</div></div></li>`;
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
    <h2>${{ verified: "Identity confirmed.", enrolled: "Enrolled.", approved: "Approved.", escalated: "Sent for investigation.", not_registered: "Not in the register.", price_stale: "Prices need new quotes." }[outcome] || "Waiting on a person."}</h2>
    <div class="card">
      <div class="big">${id.animal_id || (id.decision === "new" ? "New animal" : "—")}</div>
      <span class="badge ${outcome === "approved" || outcome === "verified" || outcome === "enrolled" ? "ok" : ""}">${outcome}</span>
      ${L.registry?.owner ? `<div class="row"><span>Registered owner</span><span>${L.registry.owner}</span></div>` : ""}
      ${id.score != null ? `<div class="row"><span>Verified points</span><span>${id.score}</span></div>` : ""}
      ${w ? `<div class="row"><span>Weight, 90%</span><span>${w.lo} to ${w.hi} kg</span></div>` : ""}
      ${v ? `<div class="row"><span>${v.p10_usd != null ? "P10 value" : "Value"}</span><span>${v.p10_usd != null ? "$" + v.p10_usd : `$${v.lo} to $${v.hi}`}</span></div>` : ""}
      ${v?.expires ? `<div class="row"><span>Valid until</span><span>${v.expires}</span></div>` : ""}
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
  const changes = tr.filter(e => e.kind === "observation" && e.changed_next_action).length;
  app.innerHTML = `${header(`<a href="#/s/${sid}/result">Back</a>`)}
    <h2>How Mombe decided.</h2>
    <p class="lede">${changes} times a vision result changed what Mombe did next. Red dots mark them.</p>
    <ul class="steps">${tr.map(stepItem).join("")}</ul>`;
}

/* ---- approver: the money decision stays with a person -------------------- */
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const usd = n => n == null ? "n/a" : "$" + Math.round(n).toLocaleString("en-US");
const day = s => s ? new Date(s + "T00:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : "";
const chance = p => p == null ? "n/a" : p < 0.001 ? "under 0.1%" : `${(100 * p).toFixed(1)}%`;
const row = (k, v) => `<div class="row"><span>${k}</span><span>${v}</span></div>`;
const KIND = { credit_offer: "Loan", claim_payout: "Claim", adjudicate: "Review" };
const quote = q => q ? `$${Number(q.usd_per_kg).toFixed(2)}/kg · <a href="${esc(q.url)}" rel="noopener" target="_blank">${esc(q.source.split(",")[0])}</a>, ${day(q.effective)}` : "";

function signInCard() {
  return approver.code
    ? `<p class="src">Signed in as approver. <a href="#/approve" id="signout">Sign out</a></p>`
    : `<div class="card"><div class="label">Approver sign-in</div>
        <p class="why">Officers and the agent can ask for a decision. Only an approver can make one.</p>
        <div class="split"><input id="ac" type="password" autocomplete="off" placeholder="Approver code"><button class="go" id="signin">Sign in</button></div></div>`;
}

function wireSignIn(after) {
  if ($("#signin")) $("#signin").onclick = () => { approver.code = $("#ac").value.trim(); after(); };
  if ($("#signout")) $("#signout").onclick = e => { e.preventDefault(); approver.code = ""; after(); };
}

async function approveScreen() {
  const items = await api("/api/approvals?status=pending");
  app.innerHTML = `${header('<a href="#/">Field</a>')}
    <h2>${items.length ? "Waiting for your decision." : "Nothing waiting."}</h2>
    ${signInCard()}
    <div class="approvals">${items.map(a => {
      const s = a.summary || {}, v = s.valuation;
      return `<a class="card pick" href="#/approve/${esc(a.id)}">
        <h3>${KIND[a.kind] || esc(a.kind)} · ${esc(s.identity?.animal_id || "unidentified")}</h3>
        ${v ? row("P10 value", `<b>${usd(v.p10_usd)}</b>`) : ""}
        ${s.loan ? row("Requested", usd(s.loan.requested_usd)) : ""}
        ${s.offer_cap_usd ? row("Offer capped at", usd(s.offer_cap_usd)) : ""}
        ${v?.expires ? row("Valid until", day(v.expires)) : ""}
        ${a.kind === "adjudicate" ? `<p class="why">${esc(s.reason)}</p>` : ""}
      </a>`;
    }).join("")}</div>`;
  wireSignIn(approveScreen);
}

async function approveDetail(id) {
  const a = await api(`/api/approvals/${id}`);
  const s = a.summary || {}, id_ = s.identity || {}, w = s.weight, v = s.valuation, loan = s.loan;
  const owner = s.owner?.registered || s.owner?.claimed;
  const cap = s.offer_cap_usd;
  app.innerHTML = `${header('<a href="#/approve">All</a>')}
    <h2>${KIND[a.kind] || esc(a.kind)} · ${esc(id_.animal_id || "unidentified")}</h2>
    ${v ? `<div class="card">
      <div class="label">P10 value</div>
      <div class="big">${usd(v.p10_usd)}</div>
      ${loan ? row("Requested", usd(loan.requested_usd)) + row(`Lendable at ${Math.round(100 * loan.ltv)}% of P10`, usd(loan.lendable_at_p10_usd)) : ""}
      ${cap ? row("Offer capped at", `<b>${usd(cap)}</b>`) : ""}
      ${row("Valid until", `${day(v.expires)}${v.current === false ? ' <span class="badge">lapsed</span>' : ""}`)}
      <p class="src">${esc(v.expiry_reason).replace(/\d{4}-\d{2}-\d{2}/g, day).replace(/^./, c => c.toUpperCase())}.</p>
    </div>` : `<div class="card"><p class="why">${a.kind === "claim_payout" ? "No valuation on record for this animal." : esc(s.reason || "")}</p></div>`}

    <div class="card">
      <div class="label">Identity</div>
      ${row("Noseprint", `${esc(id_.decision || "unknown")}${id_.score != null ? ` · ${id_.score} verified points` : ""}`)}
      ${owner ? row("Registered owner", esc(owner)) : ""}
      ${s.owner?.claimed && s.owner?.registered && s.owner.claimed !== s.owner.registered ? row("Claimed by", esc(s.owner.claimed)) : ""}
      <ul class="gates">${(s.evidence || []).map(g => {
        const reviewed = g.passed && /accepted by/.test(g.reason || "");
        return `<li class="${reviewed ? "rev" : g.passed ? "ok" : g.passed === false ? "no" : ""}" title="${esc(g.reason)}">${esc(g.gate.replace(/_/g, " "))}${reviewed ? " · reviewed" : ""}</li>`;
      }).join("")}</ul>
      ${id_.evidence_image ? `<button class="ghost" id="ev">Show matched points</button>` : ""}
    </div>

    ${w ? `<div class="card">
      <div class="label">Weight and value</div>
      ${row("Weight, 90%", `${Math.round(w.lo)} to ${Math.round(w.hi)} kg`)}
      ${v?.p10_kg ? row("P10 weight", `${Math.round(v.p10_kg)} kg`) : ""}
      ${v ? row("Market value", usd(v.market_usd)) : ""}
      ${v?.floor_quote ? `<p class="src">Floor ${quote(v.floor_quote)}</p>` : ""}
      ${v?.market_quote && v.market_quote.source !== v.floor_quote?.source ? `<p class="src">Market ${quote(v.market_quote)}</p>` : ""}
    </div>` : ""}

    <div class="card" id="herd" hidden></div>

    ${s.agent_note ? `<p class="why">${esc(s.agent_note)}</p>` : ""}
    ${s.reason && v ? `<p class="why">${esc(s.reason)}</p>` : ""}
    ${a.status === "pending" && !approver.code ? signInCard() : ""}
    ${a.status === "pending"
      ? `<div class="split"><button class="ghost" data-d="declined"${approver.code ? "" : " disabled"}>Decline</button><button class="go" data-d="approved"${approver.code ? "" : " disabled"}>Approve</button></div>`
      : `<p class="why">Decided: ${esc(a.status)} by ${esc(a.decided_by || "")}.</p>`}`;

  app.querySelectorAll("[data-d]").forEach(b => b.onclick = async () => {
    app.querySelectorAll("[data-d]").forEach(x => x.disabled = true);
    const r = await api(`/api/approvals/${id}`, { method: "POST", headers: { "content-type": "application/json", "x-mombe-approver": approver.code }, body: JSON.stringify({ status: b.dataset.d, by: "approver" }) });
    if (r.error) { approver.code = ""; return approveDetail(id); }      // wrong or expired code: sign in again
    location.hash = "#/approve";
  });
  wireSignIn(() => approveDetail(id));
  if ($("#ev")) $("#ev").onclick = async () => {
    const p = `/api/sessions/${a.session_id}/artifacts/${id_.evidence_image.split(/[\\/]/).pop()}`;
    $("#ev").outerHTML = `<img class="evidence" alt="Matched noseprint points" src="${await authImg(p)}">`;
  };
  if (a.kind === "credit_offer" && owner) herdCard(owner, a.session_id);
}

async function herdCard(owner, sid) {
  const h = await api(`/api/herd?owner=${encodeURIComponent(owner)}&session=${encodeURIComponent(sid)}`);
  if (!h.animals) return;
  const el = $("#herd");
  el.hidden = false;
  el.innerHTML = `<div class="label">${esc(owner)}'s herd · ${h.animals} weighed</div>
    ${row("P10, errors together", `<b>${usd(h.p10_conservative_usd)}</b>`)}
    ${row("P10, errors independent", usd(h.p10_independent_usd))}
    ${row("Market value", usd(h.market_usd))}
    <table class="tbl"><thead><tr><th>Loan to value</th><th>Loan</th><th>Shortfall chance</th></tr></thead>
    <tbody>${h.shortfall.map(r => `<tr><td>${Math.round(100 * r.ltv)}%</td><td>${usd(r.loan_usd)}</td><td>${chance(r.p_shortfall.with_price_together)}</td></tr>`).join("")}</tbody></table>
    <p class="src">Loan sized on the herd P10 with errors together. Shortfall: the herd sold at the floor price, with the 90-day price spread and condition loss to expiry.</p>`;
}

/* ---- router ------------------------------------------------------------- */
function route() {
  clearInterval(poller);
  stopCamera();
  const h = location.hash.slice(2).split("/");
  if (!store.key && h[0] !== "key") return keyScreen();
  if (h[0] === "key") return keyScreen();
  if (h[0] === "approve") return h[1] ? approveDetail(h[1]) : approveScreen();
  if (h[0] === "trace") return traceScreen(h[1]);
  if (h[0] === "s") return ({ capture: captureScreen, think: thinkScreen, instruct: instructScreen, result: resultScreen }[h[2]] || resultScreen)(h[1]);
  startScreen();
}
addEventListener("hashchange", route);
addEventListener("resize", drawPrint);
drawPrint();
route();
