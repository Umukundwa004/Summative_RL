import { createScene } from "./scene.js";
import { Drone } from "./drone.js";

const $ = (id) => document.getElementById(id);
const STEP_SECONDS = 0.55;
const MAX_HITS = 3;

// North is y - 1. The grid x / y become world x / z.
const MOVE_DIRS = { 0: [0, -1], 1: [0, 1], 2: [1, 0], 3: [-1, 0] };

// ---------------------------------------------------------------------------
// Episode data (made by export_episode.py)
//
// episodes/index.json lists the available episodes. Each episode holds a
// layout and one frame per environment step with the action taken, the reward
// gained or lost, the running total and the drone / package / weather state.
// ---------------------------------------------------------------------------

const EVENT_TEXT = {
  start: "Mission start",
  move: "Flying",
  wait: "Waiting (hovering)",
  collected: "Blood collected",
  delivered: "Blood delivered",
  recharged: "Battery recharged",
  returned: "Returned to base",
  obstacle: "Hit an obstacle!",
  invalid: "Invalid action",
  crash: "Drone crashed",
  battery_dead: "Battery exhausted",
  unsafe_temp: "Blood too warm",
  deadline: "Deadline missed",
};

const FAILURES = new Set(["crash", "battery_dead", "unsafe_temp", "deadline"]);

async function getJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} not found`);
  return res.json();
}

async function loadEpisode(id) {
  const data = await getJson(`/episodes/${id}.json`);
  if (!data.layout || !data.frames || !data.frames.length) {
    throw new Error(`${id}.json is empty`);
  }
  return data;
}

function showError(message) {
  $("stage").textContent = "Cannot start";
  $("stage").style.color = "#f87171";
  $("message").textContent = message;
}

async function boot() {
  const index = await getJson("/episodes/index.json");
  if (!index.length) throw new Error("no episodes exported");

  const select = $("model");
  for (const entry of index) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = entry.label;
    select.appendChild(option);
  }

  // ?model=careless opens that episode directly.
  const wanted = new URLSearchParams(location.search).get("model");
  const id = index.some((e) => e.id === wanted) ? wanted : index[0].id;
  select.value = id;

  start(await loadEpisode(id));
}

// ---------------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------------

function start(first) {
  const layout = first.layout;
  let frames = first.frames;

  const { scene, camera, update: renderScene, setPath, setCharging, hitTower } =
    createScene($("scene"), layout);
  const drone = new Drone(scene, layout.bank);
  setPath(frames);

  let index = 0;
  let playing = true;
  let speed = 1;
  let accumulator = 0;
  let holdTime = 0;

  const fmt = (n) => (n > 0 ? `+${n.toFixed(1)}` : n.toFixed(1));
  const signClass = (n) => (n > 0.05 ? "pos" : n < -0.05 ? "neg" : "zero");

  function setBar(id, percent) {
    const bar = $(id);
    bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    bar.style.background = percent > 50 ? "#3ddc84" : percent > 25 ? "#f59e0b" : "#ef4444";
  }

  // Floating +/- number above the drone.
  function popup(text, color, size) {
    const p = drone.group.position.clone().project(camera);
    const el = document.createElement("div");
    el.className = "popup";
    el.textContent = text;
    el.style.color = color;
    el.style.fontSize = `${size}px`;
    el.style.left = `${(p.x * 0.5 + 0.5) * window.innerWidth}px`;
    el.style.top = `${(-p.y * 0.5 + 0.5) * window.innerHeight - 30}px`;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 1500);
  }

  function drawChart(upTo) {
    const canvas = $("chart");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const values = frames.map((f) => f.total_reward);
    const min = Math.min(0, ...values);
    const max = Math.max(1, ...values);
    const xAt = (i) => (i / Math.max(frames.length - 1, 1)) * (w - 4) + 2;
    const yAt = (v) => h - 4 - ((v - min) / (max - min)) * (h - 8);

    ctx.strokeStyle = "#2f4470";
    ctx.beginPath();
    ctx.moveTo(0, yAt(0));
    ctx.lineTo(w, yAt(0));
    ctx.stroke();

    ctx.lineWidth = 2;
    for (let i = 1; i <= upTo; i++) {
      ctx.strokeStyle = frames[i].reward >= 0 ? "#3ddc84" : "#f87171";
      ctx.beginPath();
      ctx.moveTo(xAt(i - 1), yAt(frames[i - 1].total_reward));
      ctx.lineTo(xAt(i), yAt(frames[i].total_reward));
      ctx.stroke();
    }
    ctx.lineWidth = 1;
  }

  function addLog(f) {
    const row = document.createElement("div");
    row.className = "log-row";
    row.innerHTML =
      `<span class="name">${f.step}. ${f.action_name}</span>` +
      `<span class="${signClass(f.reward)}">${fmt(f.reward)}</span>`;
    const list = $("log-list");
    list.appendChild(row);
    list.scrollTop = list.scrollHeight;
  }

  function stageText(f) {
    if (FAILURES.has(f.event)) return ["Mission failed", "#f87171"];
    if (f.complete) return ["Mission complete", "#3ddc84"];
    if (f.delivered) return ["Returning to base", "#e6edf7"];
    if (f.collected) return ["Carrying blood to hospital", "#e6edf7"];
    return ["Collect blood at the blood bank", "#e6edf7"];
  }

  // Big red flash, message and sound when the drone hits a tower.
  let audio = null;
  function crashSound() {
    try {
      audio = audio || new AudioContext();
      const t = audio.currentTime;
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(190, t);
      osc.frequency.exponentialRampToValueAtTime(40, t + 0.3);
      gain.gain.setValueAtTime(0.18, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
      osc.connect(gain).connect(audio.destination);
      osc.start(t);
      osc.stop(t + 0.35);
    } catch (err) {
      // Sound is optional.
    }
  }

  function impact(title, reward) {
    const el = $("impact");
    $("impact-text").textContent = title;
    $("impact-sub").textContent = `${fmt(reward)} reward`;
    el.classList.remove("show");
    void el.offsetWidth;
    el.classList.add("show");
    crashSound();
  }

  // Visual reaction to what just happened.
  function react(f, prev) {
    drone.setWaiting(f.event === "wait");
    setCharging(f.event === "recharged");

    if (f.event === "move") {
      drone.faceDirection(f.x - prev.x, f.y - prev.y);
    }

    // The drone flew into a tower (the third hit shows up as a crash).
    if (f.hits > prev.hits && MOVE_DIRS[f.action]) {
      const [dx, dz] = MOVE_DIRS[f.action];
      drone.faceDirection(dx, dz);
      drone.bump(dx, dz);
      drone.shake(0.6);
      drone.flash(0xef4444, 0.9);
      hitTower(f.x + dx, f.y + dz, dx, dz);
      impact(f.event === "crash" ? "CRASHED INTO A TOWER" : "TOWER HIT!", f.reward);
    }

    switch (f.event) {
      case "collected":
      case "delivered":
      case "returned":
        drone.flash(0x22c55e, 0.7);
        break;
      case "recharged":
        drone.flash(0x4ade80, 0.6);
        break;
      case "invalid":
        drone.shake(0.35);
        drone.flash(0xf97316, 0.5);
        break;
      case "crash":
      case "battery_dead":
      case "unsafe_temp":
      case "deadline":
        drone.shake(0.8);
        drone.flash(0xef4444, 1.2);
        break;
      default:
        break;
    }

    if (f.action !== null) {
      const big = Math.abs(f.reward) >= 10;
      popup(fmt(f.reward), f.reward >= 0 ? "#3ddc84" : "#f87171", big ? 30 : 17);
    }
  }

  function showFrame(i, animate) {
    const f = frames[i];
    const prev = frames[Math.max(i - 1, 0)];
    drone.setCell(f.x, f.y);
    drone.setCargoVisible(f.collected && !f.delivered);

    $("total").textContent = f.total_reward.toFixed(1);
    $("total").style.color = f.total_reward >= 0 ? "#3ddc84" : "#f87171";
    $("delta").textContent = f.action === null ? "" : fmt(f.reward);
    $("delta").style.color = f.reward >= 0 ? "#3ddc84" : "#f87171";
    $("action-chip").textContent =
      f.action === null ? "Start" : `Action ${f.action} · ${f.action_name}`;
    $("event-text").textContent = EVENT_TEXT[f.event] || f.event;

    $("battery-text").textContent = `${f.battery.toFixed(0)}%`;
    $("integrity-text").textContent = `${f.integrity.toFixed(0)}%`;
    $("temp-text").textContent = `${f.temperature.toFixed(1)} °C`;
    $("wind-text").textContent = f.wind.toFixed(1);
    $("rain-text").textContent = f.rain.toFixed(1);
    $("hits-text").textContent = `${f.hits} / ${MAX_HITS}`;
    $("step-text").textContent = `${f.step} / ${frames[frames.length - 1].step}`;
    setBar("battery-bar", f.battery);
    setBar("integrity-bar", f.integrity);

    const [text, color] = stageText(f);
    $("stage").textContent = text;
    $("stage").style.color = color;
    $("message").textContent =
      f.wind >= 6.5 ? "Storm: strong wind can push the drone off course." : "";

    drawChart(i);

    if (animate) {
      if (f.action !== null) addLog(f);
      react(f, prev);
    }
  }

  function restart() {
    index = 0;
    accumulator = 0;
    holdTime = 0;
    $("log-list").innerHTML = "";
    drone.setWaiting(false);
    setCharging(false);
    showFrame(0, false);
    drone.snapToTarget();
  }

  async function changeEpisode(id) {
    try {
      const episode = await loadEpisode(id);
      frames = episode.frames;
      setPath(frames);
      restart();
    } catch (err) {
      showError(err.message);
    }
  }

  $("play").addEventListener("click", () => {
    playing = !playing;
    $("play").textContent = playing ? "Pause" : "Play";
  });
  $("restart").addEventListener("click", restart);
  $("speed").addEventListener("change", (e) => { speed = Number(e.target.value); });
  $("model").addEventListener("change", (e) => changeEpisode(e.target.value));

  let last = performance.now();
  function loop(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;

    if (playing) {
      if (index < frames.length - 1) {
        accumulator += dt * speed;
        while (accumulator >= STEP_SECONDS && index < frames.length - 1) {
          accumulator -= STEP_SECONDS;
          index++;
          showFrame(index, true);
        }
      } else {
        // Stay on the last frame for a moment, then replay.
        holdTime += dt;
        if (holdTime > 4) restart();
      }
    }

    drone.update(dt, now / 1000);
    renderScene(dt, frames[index].rain);
    requestAnimationFrame(loop);
  }

  restart();

  // ?skip=N jumps N steps ahead (handy for screenshots and demos).
  const skip = Number(new URLSearchParams(location.search).get("skip")) || 0;
  for (let i = 0; i < skip && index < frames.length - 1; i++) {
    index++;
    showFrame(index, true);
  }

  requestAnimationFrame(loop);
}

boot().catch((err) => showError(`${err.message}. Run: python export_episode.py`));
