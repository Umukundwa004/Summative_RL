import { createScene } from "./scene.js";
import { Drone } from "./drone.js";

const $ = (id) => document.getElementById(id);
const STEP_SECONDS = 0.55;

// North is y - 1. The grid x / y become world x / z.
const MOVE_DIRS = { 0: [0, -1], 1: [0, 1], 2: [1, 0], 3: [-1, 0] };

// ---------------------------------------------------------------------------
// Episode data (made by export_episode.py)
//
// episodes/index.json lists the available episodes. Each episode holds a
// layout and one frame per environment step with the action taken, the reward
// gained or lost, the running total and the drone / package / weather state.
// ---------------------------------------------------------------------------

const ROUNDS = 1; // deliveries per mission (same as ROUNDS in custom_env.py)

const EVENT_TEXT = {
  start: "Mission start",
  move: "Flying",
  wait: "Waiting (hovering)",
  climbed: "Climbing",
  descended: "Descending",
  collected: "Blood collected",
  delivered: "Blood delivered",
  recharged: "Battery recharged",
  round_done: "Round finished, back at base",
  returned: "Mission complete",
  must_charge_first: "Must charge before the next round",
  not_enough_battery: "Not enough battery for a round",
  obstacle: "Hit a tower or building!",
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

// Shows a problem in the end-of-episode card (it is always on the page).
function showError(message) {
  $("summary").style.display = "block";
  $("summary").querySelector("h2").textContent = "Cannot start";
  $("summary-outcome").textContent = message;
}

async function boot() {
  const index = await getJson("/episodes/index.json");
  if (!index.length) throw new Error("no episodes exported");

  const select = $("episode");
  for (const entry of index) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = entry.label;
    select.appendChild(option);
  }

  // ?episode=ppo-seed-1 opens one model/seed replay. Keep ?model=dqn compatible.
  const params = new URLSearchParams(location.search);
  const wanted = params.get("episode") || params.get("model");
  const id = index.some((e) => e.id === wanted) ? wanted : index[0].id;
  select.value = id;

  start(await loadEpisode(id));
}

// Changing the model reloads the page with the new model in the address bar.
// Each model has its own episode, so a new scene is built from scratch.
function goTo(changes) {
  const params = new URLSearchParams(location.search);
  params.delete("model");
  for (const [key, value] of Object.entries(changes)) params.set(key, value);
  location.search = params.toString();
}

// ---------------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------------

function start(episode) {
  const layout = episode.layout;
  const frames = episode.frames;

  const { scene, camera, update: renderScene, setPath, setCharging, hitTower } =
    createScene($("scene"), layout);
  const drone = new Drone(scene, layout.bank);
  setPath(frames);

  let index = 0;
  let playing = true;
  let speed = 1;
  let accumulator = 0;
  let holdTime = 0;

  function setBar(id, percent) {
    const bar = $(id);
    bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    bar.style.background = percent > 50 ? "#3ddc84" : percent > 25 ? "#f59e0b" : "#ef4444";
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

  function impact(title) {
    const el = $("impact");
    $("impact-text").textContent = title;
    $("impact-sub").textContent = "";
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
      hitTower(f.x + dx, f.y + dz, dx, dz, f.altitude);
      impact(f.event === "crash" ? "CRASHED INTO A TOWER" : "TOWER HIT!");
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

  }

  function showFrame(i, animate) {
    const f = frames[i];
    const prev = frames[Math.max(i - 1, 0)];
    drone.setCell(f.x, f.y, f.altitude);
    drone.setCargoVisible(f.collected && !f.round_delivered);

    // Total reward so far (the only reward shown during the episode).
    $("total").textContent = f.total_reward.toFixed(1);
    $("total").style.color = f.total_reward >= 0 ? "#3ddc84" : "#f87171";

    // Battery percentage: the only status card on the dashboard.
    $("battery-text").textContent = `${f.battery.toFixed(0)}%`;
    setBar("battery-bar", f.battery);

    // At the end of the episode show the total reward of the whole episode.
    if (i === frames.length - 1) {
      showSummary(f);
    } else {
      hideSummary();
    }

    drawChart(i);

    if (animate) react(f, prev);
  }

  // The end-of-episode card: total reward and how the episode ended.
  function showSummary(f) {
    $("summary-total").textContent = f.total_reward.toFixed(1);
    $("summary-total").style.color = f.total_reward >= 0 ? "#3ddc84" : "#f87171";
    $("summary-outcome").textContent = EVENT_TEXT[f.event] || f.event;
    $("summary-rounds").textContent = `${f.rounds_done} / ${ROUNDS}`;
    $("summary-steps").textContent = String(f.step);
    $("summary").style.display = "block";
  }

  function hideSummary() {
    $("summary").style.display = "none";
  }

  function restart() {
    index = 0;
    accumulator = 0;
    holdTime = 0;
    hideSummary();
    drone.setWaiting(false);
    setCharging(false);
    showFrame(0, false);
    drone.snapToTarget();
  }

  $("play").addEventListener("click", () => {
    playing = !playing;
    $("play").textContent = playing ? "Pause" : "Play";
  });
  $("restart").addEventListener("click", restart);
  $("speed").addEventListener("change", (e) => { speed = Number(e.target.value); });
  $("episode").addEventListener("change", (e) => goTo({ episode: e.target.value }));

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
        // Stay on the last frame so the total reward can be read, then replay.
        holdTime += dt;
        if (holdTime > 10) restart();
      }
    }

    drone.update(dt, now / 1000);
    renderScene(dt, frames[index].rain, frames[index].storms || []);
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
