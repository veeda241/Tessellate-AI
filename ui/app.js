"use strict";

const $ = (id) => document.getElementById(id);
const signed = (n, dp) => (n > 0 ? "+" : "") + n.toFixed(dp);

/* ── Tabs ── */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
  });
});

/* ── Health + metrics strip ── */
async function loadHealth() {
  const conn = $("conn"), text = $("conn-text");
  try {
    const h = await (await fetch("/healthz")).json();
    text.textContent = h.model_loaded ? "Inference cluster online" : "model not loaded";
    conn.querySelector(".dot").className = "dot " + (h.model_loaded ? "ok" : "bad");
    $("m-version").textContent = (h.model_version || "unknown").slice(0, 10);
    $("m-rmse").textContent = h.holdout_rmse_positions != null ? h.holdout_rmse_positions + " pos" : "—";
    $("m-mae").textContent = h.holdout_mae_positions != null
      ? `${h.holdout_mae_positions} vs ${h.baseline_mae_positions}` : "—";
    $("m-rows").textContent = (h.total_training_rows || "—").toLocaleString ? (h.total_training_rows || 0).toLocaleString() : h.total_training_rows;
    $("m-tracks").textContent = h.track_count ?? "—";
  } catch (e) {
    text.textContent = "server unreachable";
    conn.querySelector(".dot").className = "dot bad";
  }
}

/* ── Circuit dropdown ── */
async function loadTracks() {
  try {
    const { tracks } = await (await fetch("/api/tracks")).json();
    const sel = $("f-track");
    sel.innerHTML = "";
    (tracks.length ? tracks : ["British Grand Prix"]).forEach((t) => {
      const o = document.createElement("option");
      o.value = o.textContent = t;
      if (t === "British Grand Prix") o.selected = true;
      sel.appendChild(o);
    });
  } catch (e) { /* leave empty */ }
}

/* ── Manual evaluation ── */
$("evaluate").addEventListener("click", async () => {
  const payload = {
    track_name: $("f-track").value,
    laps_remaining: +$("f-laps").value,
    current_position: +$("f-pos").value,
    tire_age: +$("f-age").value,
    tire_compound: $("f-compound").value,
    radio_transcript: $("f-radio").value,
    race_control_text: $("f-control").value,
    gap_ahead: +$("f-gapA").value,
    gap_behind: +$("f-gapB").value,
    driver_pace_index: +$("f-pace").value,
  };
  const btn = $("evaluate");
  btn.disabled = true; btn.textContent = "Evaluating…";
  const t0 = performance.now();
  try {
    const res = await (await fetch("/api/predict_counterfactual", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    })).json();
    const rt = performance.now() - t0;
    if (res.detail) { showVerdict("idle", Array.isArray(res.detail) ? "Invalid input — check fields" : res.detail); return; }
    const p = res.counterfactual_projections;
    const box = p.BOX_NOW_EXPECTED_FINISH, stay = p.STAY_OUT_EXPECTED_FINISH;
    const isBox = res.pit_wall_recommendation === "BOX";

    showVerdict(isBox ? "box" : "stay", isBox ? "🛑 BOX — PIT THIS LAP" : "🟢 STAY OUT — HOLD TRACK POSITION");
    $("p-box").textContent = "P" + box;
    $("p-stay").textContent = "P" + stay;
    document.querySelector(".proj.box").classList.toggle("winner", isBox);
    document.querySelector(".proj.stay").classList.toggle("winner", !isBox);
    $("v-gain").textContent = (res.expected_position_gain > 0 ? "+" : "") + res.expected_position_gain + " pos";
    $("v-core").textContent = res.inference_ms + " ms";
    $("v-rt").textContent = Math.round(rt) + " ms";
    $("v-urgency").textContent = res.jev_scout_indicators.tire_urgency_label;

    const risk = res.jev_scout_indicators.safety_car_escalation_risk;
    $("v-sc-num").textContent = risk;
    $("v-sc-bar").style.width = Math.round(risk * 100) + "%";

    const rc = res.rejoin_context || {};
    const gapTxt = payload.gap_ahead.toFixed(1);
    $("v-pitloss").textContent = rc.pit_loss_s == null ? "—" : rc.pit_loss_s.toFixed(1) + " s";
    $("v-rejoin").textContent = rc.rejoin_gap_s == null ? "—" : signed(rc.rejoin_gap_s, 1) + " s";
    $("rejoin-note").textContent = rc.rejoin_gap_s == null ? "" : (
      rc.rejoins_behind_car_ahead
        ? `Gap ahead is ${gapTxt} s, under the ${rc.pit_loss_s.toFixed(1)} s pit loss — the stop rejoins behind that car and its dirty air is priced into BOX NOW.`
        : `Gap ahead of ${gapTxt} s covers the ${rc.pit_loss_s.toFixed(1)} s pit loss — the stop rejoins in clean air, so no place is surrendered.`
    );

    $("fallback-note").textContent = res.track_fallback_used
      ? "Circuit not in track constants — generic traffic prior applied." : "";
  } catch (e) {
    showVerdict("idle", "Evaluation failed: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Evaluate Parallel Universes 🏁";
  }
});

function showVerdict(cls, text) {
  const v = $("verdict");
  v.className = "verdict " + cls;
  v.textContent = text;
}

/* ── Live feed polling + canvas chart ── */
async function pollLive() {
  const dot = $("live-dot");
  try {
    const d = await (await fetch("/api/live_state")).json();
    if (!d.available) {
      dot.className = "dot idle";
      $("live-updated").innerHTML = 'streamer offline — start with <code>python -m api.openf1_streamer</code>';
      return;
    }
    dot.className = "dot " + (d.age_seconds < 4 ? "ok" : "idle");
    const s = d.latest_snapshot || {}, dec = d.latest_decision || {};
    $("live-session").textContent = `${s.session_name || "Session"} — Lap ${s.lap_number ?? "—"}`;
    $("live-updated").textContent = `updated ${d.age_seconds}s ago`;
    $("s-drv").textContent = "Driver #" + (s.driver_number ?? "—");
    $("s-pos").textContent = "P" + (s.current_position ?? "—");
    $("s-age").textContent = (s.tire_age ?? "—");
    $("s-comp").textContent = s.tire_compound ?? "—";
    $("s-laps").textContent = s.laps_remaining ?? "—";
    $("s-lap").textContent = s.lap_number ?? "—";
    $("s-radio").textContent = s.radio_transcript || "—";
    $("s-control").textContent = s.race_control_text || "—";

    const lv = $("live-verdict");
    if (dec.status === "success") {
      const isBox = dec.pit_wall_recommendation === "BOX";
      const rj = dec.rejoin_context ? ` · rejoin ${signed(dec.rejoin_context.rejoin_gap_s, 1)} s` : "";
      lv.className = "verdict small " + (isBox ? "box" : "stay");
      lv.textContent = `${isBox ? "BOX" : "STAY OUT"} · gain ${dec.expected_position_gain} positions${rj} · ${dec.inference_ms} ms`;
    } else {
      lv.className = "verdict small idle";
      lv.textContent = "Engine verdict pending — " + (dec.status || "no decision");
    }
    drawChart(d.rolling_decisions || []);
    renderTable(d.rolling_decisions || []);
  } catch (e) { /* transient, keep last state */ }
}

function drawChart(rows) {
  const cv = $("chart"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, pad = 30, base = H - 30;
  ctx.clearRect(0, 0, W, H);
  if (!rows.length) {
    ctx.fillStyle = "#8b95a7"; ctx.font = "14px Segoe UI"; ctx.textAlign = "center";
    ctx.fillText("Awaiting rolling decisions…", W / 2, H / 2);
    return;
  }
  const gains = rows.map((r) => Number(r.gain) || 0);
  const maxAbs = Math.max(1, ...gains.map(Math.abs));
  const bw = Math.min(26, (W - pad * 2) / rows.length - 4);
  const step = (W - pad * 2) / rows.length;
  // zero baseline
  ctx.strokeStyle = "#262c3a"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, base); ctx.lineTo(W - pad, base); ctx.stroke();
  rows.forEach((r, i) => {
    const g = Number(r.gain) || 0;
    const h = (Math.abs(g) / maxAbs) * (base - 20);
    const x = pad + i * step + step / 2 - bw / 2;
    const isBox = r.call === "BOX";
    ctx.fillStyle = isBox ? "#ff2d20" : "#2ecc71";
    if (g >= 0) ctx.fillRect(x, base - h, bw, h);          // BOX favourable
    else ctx.fillRect(x, base, bw, h);                      // STAY favourable
    ctx.fillStyle = "#8b95a7"; ctx.font = "10px Segoe UI"; ctx.textAlign = "center";
    ctx.fillText(g > 0 ? "+" + g.toFixed(1) : g.toFixed(1), x + bw / 2, g >= 0 ? base - h - 4 : base + h + 11);
  });
}

function renderTable(rows) {
  const tbody = $("dec-table").querySelector("tbody");
  tbody.innerHTML = "";
  rows.slice(-12).reverse().forEach((r) => {
    const tr = document.createElement("tr");
    const cls = r.call === "BOX" ? "call-box" : "call-stay";
    tr.innerHTML = `<td>${r.lap}</td><td>#${r.driver}</td><td class="${cls}">${r.call || "—"}</td>` +
      `<td>${r.gain > 0 ? "+" : ""}${r.gain}</td>` +
      `<td>${r.rejoin == null ? "—" : signed(r.rejoin, 1)}</td><td>${r.inference_ms}</td>`;
    tbody.appendChild(tr);
  });
}

function tickClock() { $("foot-clock").textContent = new Date().toLocaleTimeString(); }

/* ── Boot ── */
loadHealth();
loadTracks();
pollLive();
setInterval(pollLive, 1500);
setInterval(loadHealth, 10000);
setInterval(tickClock, 1000); tickClock();
