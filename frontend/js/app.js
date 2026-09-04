/**
 * UniArchives-Styled Personal Safety Monitor Dashboard Controller.
 * Manages WebSocket telemetry, audio alert synthesis, live UI updates,
 * replay controls, and demo scenario activation.
 */

// Web Audio API Alarm Synthesizer
class SoundEffects {
  constructor() {
    this.audioCtx = null;
    this.isMuted = false;
    this.lastAlarmTime = 0;
  }

  init() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) this.audioCtx = new AudioContext();
    }
  }

  playBeep(freq = 880, type = "sine", duration = 0.15) {
    if (this.isMuted) return;
    this.init();
    if (!this.audioCtx) return;

    try {
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);

      gain.gain.setValueAtTime(0.2, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + duration);

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc.start();
      osc.stop(this.audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("Audio play error:", e);
    }
  }

  playThreatAlarm() {
    const now = Date.now();
    if (now - this.lastAlarmTime > 2000) {
      this.lastAlarmTime = now;
      this.playBeep(920, "sawtooth", 0.25);
      setTimeout(() => this.playBeep(1240, "sawtooth", 0.35), 250);
    }
  }
}

const sfx = new SoundEffects();

// Charts
let edaChart, respChart, accChart, threatChart;

// App State
let socket = null;
let activeScenario = "LIVE_FREE";
let isPlaying = true;
let currentTheme = "dark";

// DOM References
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  initTheme();
  connectWebSocket();
  fetchSubjects();
  setupEventListeners();
});

function initCharts() {
  edaChart = new RealtimeWaveformChart("edaCanvas", 100, "#00f0ff", "EDA (uS)");
  respChart = new RealtimeWaveformChart("respCanvas", 100, "#ccff00", "Resp (std)");
  accChart = new RealtimeWaveformChart("accCanvas", 100, "#ffb700", "Acc Mag (g)");
  threatChart = new ThreatHistoryChart("threatCanvas", 120);
}

function initTheme() {
  document.documentElement.setAttribute("data-theme", currentTheme);
  const themeBtn = document.getElementById("themeToggleBtn");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      currentTheme = currentTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", currentTheme);
      themeBtn.innerHTML = currentTheme === "dark" ? "☀️ LIGHT" : "🌙 DARK";
    });
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  const statusBadge = document.getElementById("wsStatusBadge");
  if (statusBadge) {
    statusBadge.textContent = "CONNECTING...";
    statusBadge.className = "badge-uni badge-cyan";
  }

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log("[WS] Connected to telemetry backend.");
    if (statusBadge) {
      statusBadge.textContent = "● LIVE WS";
      statusBadge.className = "badge-uni badge-neon";
    }
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "TELEMETRY") {
        updateDashboard(data);
      }
    } catch (e) {
      console.error("[WS] Parse error:", e);
    }
  };

  socket.onclose = () => {
    console.warn("[WS] Connection lost. Reconnecting in 2s...");
    if (statusBadge) {
      statusBadge.textContent = "○ DISCONNECTED";
      statusBadge.className = "badge-uni badge-alert";
    }
    setTimeout(connectWebSocket, 2000);
  };
}

function updateDashboard(data) {
  // 1. Video Feed
  const feed = document.getElementById("webcamFeed");
  if (feed && data.camera_frame) {
    feed.src = data.camera_frame;
  }

  // 2. Telemetry Header Stats
  setText("streamFps", `${data.fps} FPS`);
  setText("packetCounter", data.packet_counter);
  setText("activeScenarioDisplay", data.demo_scenario || "LIVE_FREE");

  // 3. Raw Sensor Indicators
  const sp = data.sensor_packet || {};
  setText("valHR", sp.heart_rate ? `${Math.round(sp.heart_rate)} BPM` : "-- BPM");
  setText("valEDA", sp.eda !== undefined ? `${sp.eda.toFixed(2)} µS` : "--");
  setText("valResp", sp.respiration !== undefined ? `${sp.respiration.toFixed(2)}` : "--");
  setText("valTemp", sp.temperature !== undefined ? `${sp.temperature.toFixed(1)} °C` : "--");
  setText("valAcc", sp.acc_mag !== undefined ? `${sp.acc_mag.toFixed(2)} g` : "--");
  setText("groundTruthLabel", sp.ground_truth_label !== undefined ? `GT: ${data.replay?.ground_truth_name || sp.ground_truth_label}` : "GT: --");

  // Push chart values
  if (sp.eda !== undefined) edaChart.push(sp.eda);
  if (sp.respiration !== undefined) respChart.push(sp.respiration);
  if (sp.acc_mag !== undefined) accChart.push(sp.acc_mag);

  // 4. Sensor ML Breakdown
  const sm = data.sensor_ml || {};
  const probs = sm.probabilities || { baseline: 0, stress: 0, amusement: 0 };
  setText("sensorDistressScore", `${Math.round((sm.distress_score || 0) * 100)}%`);
  setBar("sensorDistressBar", (sm.distress_score || 0) * 100);

  setText("probBaseline", `${Math.round((probs.baseline || 0) * 100)}%`);
  setText("probStress", `${Math.round((probs.stress || 0) * 100)}%`);
  setText("probAmusement", `${Math.round((probs.amusement || 0) * 100)}%`);
  setBar("barBaseline", (probs.baseline || 0) * 100);
  setBar("barStress", (probs.stress || 0) * 100);
  setBar("barAmusement", (probs.amusement || 0) * 100);
  setText("sensorLatency", `${sm.latency_ms || 0} ms`);

  // 5. Vision ML Breakdown
  const vm = data.vision_ml || {};
  setText("visionDistressScore", `${Math.round((vm.distress_score || 0) * 100)}%`);
  setBar("visionDistressBar", (vm.distress_score || 0) * 100);
  setText("visionPosture", vm.posture || "UNKNOWN");
  setText("visionExpression", vm.facial_expression || "CALM / NEUTRAL");
  setText("visionMovement", vm.movement_level || "NORMAL");
  setText("visionConfidence", `${Math.round((vm.confidence || 0) * 100)}%`);

  // 6. Multimodal Threat Score & State Machine
  const fusion = data.fusion || {};
  const threatPct = fusion.threat_score_pct || 0;
  setText("threatScoreText", `${threatPct.toFixed(1)}%`);
  setBar("threatMeterFill", threatPct, fusion.verdict_color);

  threatChart.push(fusion.threat_score || 0);

  // State Banner & Styling
  const smState = data.state_machine || {};
  const currState = smState.current_state || fusion.verdict || "SAFE";
  const stateBadge = document.getElementById("stateBadge");
  const verdictBanner = document.getElementById("verdictBanner");

  if (stateBadge) {
    stateBadge.textContent = currState;
    if (currState.includes("THREAT") || currState.includes("ALERT")) {
      stateBadge.className = "btn-uni btn-alert text-sm px-4 py-2 font-display pulse-threat";
      sfx.playThreatAlarm();
    } else if (currState.includes("CAUTION") || currState.includes("SUSPICIOUS")) {
      stateBadge.className = "btn-uni bg-uni-dark text-uni-cyan border-uni-cyan text-sm px-4 py-2 font-display";
    } else {
      stateBadge.className = "btn-uni btn-neon text-sm px-4 py-2 font-display";
    }
  }

  setText("fusionConsensus", fusion.agreement_status || "CONGRUENT");
  setText("sensorWeightDisp", `Sensor (${fusion.sensor_weight * 100}%)`);
  setText("visionWeightDisp", `Vision (${fusion.vision_weight * 100}%)`);

  // 7. Replay Telemetry
  const rep = data.replay || {};
  setText("currentSubjectBadge", rep.subject_id || "S2");
  setText("replaySpeedBadge", `${rep.speed || 1.0}x`);
  setText("elapsedTime", `${rep.elapsed_seconds || 0}s / ${rep.total_seconds || 0}s`);
  const progressBar = document.getElementById("replayProgressBar");
  if (progressBar && rep.progress_fraction !== undefined) {
    progressBar.style.width = `${rep.progress_fraction * 100}%`;
  }

  // 8. Event Log Updates
  if (data.recent_logs && data.recent_logs.length > 0) {
    renderEventLog(data.recent_logs);
  }
}

function renderEventLog(logs) {
  const container = document.getElementById("eventLogContainer");
  if (!container) return;

  const html = logs.map(l => {
    let colorClass = "text-uni-muted";
    if (l.level === "CRITICAL" || l.level === "EMERGENCY") colorClass = "text-uni-alert font-bold";
    else if (l.level === "WARNING") colorClass = "text-uni-cyan";
    else if (l.category === "SYSTEM") colorClass = "text-uni-neon";

    return `
      <div class="flex items-start gap-2 py-1 border-b border-uni-border/40 text-xs font-mono">
        <span class="text-uni-muted shrink-0">[${l.timestamp_str}]</span>
        <span class="badge-uni shrink-0">${l.category}</span>
        <span class="${colorClass}">${l.message}</span>
      </div>
    `;
  }).join("");

  container.innerHTML = html;
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setBar(id, pct, color = null) {
  const el = document.getElementById(id);
  if (el) {
    el.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    if (color) el.style.backgroundColor = color;
  }
}

function setupEventListeners() {
  // Replay Controls
  document.getElementById("btnPlay")?.addEventListener("click", () => sendReplayControl({ action: "play" }));
  document.getElementById("btnPause")?.addEventListener("click", () => sendReplayControl({ action: "pause" }));
  document.getElementById("btnReset")?.addEventListener("click", () => sendReplayControl({ action: "reset" }));

  // Preset Segment Jumps
  document.getElementById("jumpBaseline")?.addEventListener("click", () => sendReplayControl({ action: "scenario", scenario: "BASELINE" }));
  document.getElementById("jumpStress")?.addEventListener("click", () => sendReplayControl({ action: "scenario", scenario: "STRESS" }));
  document.getElementById("jumpAmusement")?.addEventListener("click", () => sendReplayControl({ action: "scenario", scenario: "AMUSEMENT" }));

  // Speed buttons
  document.querySelectorAll("[data-speed]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const speed = parseFloat(e.currentTarget.getAttribute("data-speed"));
      sendReplayControl({ action: "speed", speed });
    });
  });

  // Subject Selector
  document.getElementById("subjectSelect")?.addEventListener("change", (e) => {
    sendReplayControl({ action: "subject", subject_id: e.target.value });
  });

  // Timeline scrubber
  const scrubber = document.getElementById("timelineScrubber");
  if (scrubber) {
    scrubber.addEventListener("click", (e) => {
      const rect = scrubber.getBoundingClientRect();
      const fraction = (e.clientX - rect.left) / rect.width;
      sendReplayControl({ action: "seek", seek_fraction: fraction });
    });
  }

  // Demo Scenarios
  document.querySelectorAll("[data-scenario]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const scenario = e.currentTarget.getAttribute("data-scenario");
      triggerScenario(scenario);
    });
  });

  // Alert Reset
  document.getElementById("btnAlertReset")?.addEventListener("click", () => {
    fetch("/api/alert/reset", { method: "POST" });
  });

  // Weight Slider
  const sensorSlider = document.getElementById("sensorWeightSlider");
  if (sensorSlider) {
    sensorSlider.addEventListener("input", (e) => {
      const sW = parseFloat(e.target.value);
      const vW = 1.0 - sW;
      document.getElementById("sensorWeightLabel").textContent = `${Math.round(sW * 100)}%`;
      document.getElementById("visionWeightLabel").textContent = `${Math.round(vW * 100)}%`;
      fetch("/api/fusion/weights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sensor_weight: sW, vision_weight: vW })
      });
    });
  }

  // Sound toggle
  const muteBtn = document.getElementById("muteBtn");
  if (muteBtn) {
    muteBtn.addEventListener("click", () => {
      sfx.isMuted = !sfx.isMuted;
      muteBtn.innerHTML = sfx.isMuted ? "🔇 MUTED" : "🔊 ALARM ON";
    });
  }
}

function sendReplayControl(payload) {
  fetch("/api/replay/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

function triggerScenario(scenario) {
  activeScenario = scenario;
  document.querySelectorAll("[data-scenario]").forEach(b => {
    b.classList.remove("border-uni-neon", "text-uni-neon", "bg-uni-neon/10");
    if (b.getAttribute("data-scenario") === scenario) {
      b.classList.add("border-uni-neon", "text-uni-neon", "bg-uni-neon/10");
    }
  });

  fetch("/api/demo/scenario", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario })
  });
}

function fetchSubjects() {
  fetch("/api/replay/subjects")
    .then(res => res.json())
    .then(data => {
      const select = document.getElementById("subjectSelect");
      if (select && data.subjects) {
        select.innerHTML = data.subjects.map(s => `<option value="${s}">${s}</option>`).join("");
      }
    })
    .catch(err => console.error("Error fetching subjects:", err));
}
