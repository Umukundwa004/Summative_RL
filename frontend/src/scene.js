import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// The map layout (grid size, bank, hospital, charger, obstacles) comes from
// the episode file, which export_episode.py reads from environment/custom_env.py.
export const world = { gridSize: 10 };
let LAYOUT = null;
let M = 0; // extra margin when the grid is larger than 8 x 8

export const CELL = 1;

// Altitude levels match environment/custom_env.py (0 = on the ground, 3 = highest).
// Level 0 is just above the ground; every level up is one step higher.
const GROUND_FLY = 0.6;
const LEVEL_STEP = 0.9;

// Height of the drone above the ground at an altitude level.
export function altitudeHeight(level = 0) {
  return GROUND_FLY + level * LEVEL_STEP;
}

// Top of a tower or building. A tower of height h blocks the drone at levels
// h and below, so the top sits between level h and level h + 1.
// Heights: 1 = low tower, 2 = medium, 3 = high, 4 = building.
export function towerTop(height) {
  return 0.8 + 0.9 * height;
}

const SKY = 0xcfe3f2;

// Fixed seed so the landscape looks the same on every page load.
function mulberry32(seed) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(7);
const range = (a, b) => a + (b - a) * rand();
const pick = (list) => list[Math.floor(rand() * list.length)];

// Grid cell (x, y) -> world position. Environment "north" (y - 1) is world -z.
export function cellToWorld(x, y, height = 0) {
  const offset = (world.gridSize - 1) / 2;
  return new THREE.Vector3((x - offset) * CELL, height, (y - offset) * CELL);
}

function box(w, h, d, color) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({ color, roughness: 0.7 })
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

// ---------------------------------------------------------------------------
// Mission area (the grid the agent plays on)
// ---------------------------------------------------------------------------

function buildGround(scene) {
  // Grass that surrounds everything.
  const grass = new THREE.Mesh(
    new THREE.PlaneGeometry(220, 220),
    new THREE.MeshStandardMaterial({ color: 0x6aa14f, roughness: 1 })
  );
  grass.rotation.x = -Math.PI / 2;
  grass.position.y = -0.1;
  grass.receiveShadow = true;
  scene.add(grass);

  for (let x = 0; x < world.gridSize; x++) {
    for (let y = 0; y < world.gridSize; y++) {
      const color = (x + y) % 2 === 0 ? 0xd5dbe2 : 0xc9d0d8;
      const tile = box(CELL * 0.98, 0.1, CELL * 0.98, color);
      tile.position.copy(cellToWorld(x, y, -0.05));
      scene.add(tile);
    }
  }
}

// Translucent zone with outline, like the no-fly zones in a drone map.
function addZone(scene, cell, color, height) {
  const g = new THREE.BoxGeometry(0.97, height, 0.97);
  const zone = new THREE.Mesh(
    g,
    new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.18, depthWrite: false,
    })
  );
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(g),
    new THREE.LineBasicMaterial({ color })
  );
  const p = cellToWorld(cell[0], cell[1], height / 2);
  zone.position.copy(p);
  edges.position.copy(p);
  scene.add(zone, edges);
}

function buildBank(scene) {
  const g = new THREE.Group();
  const base = box(0.8, 0.7, 0.8, 0x3b82f6);
  base.position.y = 0.35;
  const roof = box(0.9, 0.08, 0.9, 0x1e40af);
  roof.position.y = 0.74;
  g.add(base, roof);
  g.position.copy(cellToWorld(...LAYOUT.bank));
  scene.add(g);
}

// Draws one hospital at a cell. The drone may use several hospitals in one episode.
function buildHospital(scene, cell) {
  const g = new THREE.Group();
  const base = box(0.85, 1.1, 0.85, 0xf8fafc);
  base.position.y = 0.55;
  const crossV = box(0.12, 0.45, 0.02, 0xdc2626);
  const crossH = box(0.45, 0.12, 0.02, 0xdc2626);
  crossV.position.set(0, 0.65, 0.435);
  crossH.position.set(0, 0.65, 0.435);

  // Helipad on the roof.
  const roof = box(0.9, 0.04, 0.9, 0xcbd5e1);
  roof.position.y = 1.12;
  const pad = new THREE.Mesh(
    new THREE.CircleGeometry(0.3, 28),
    new THREE.MeshStandardMaterial({ color: 0x2563eb })
  );
  pad.rotation.x = -Math.PI / 2;
  pad.position.y = 1.145;
  const hMat = new THREE.MeshStandardMaterial({ color: 0xffffff });
  const left = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.01, 0.26), hMat);
  const right = left.clone();
  const mid = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.01, 0.05), hMat);
  left.position.set(-0.09, 1.155, 0);
  right.position.set(0.09, 1.155, 0);
  mid.position.set(0, 1.155, 0);

  g.add(base, crossV, crossH, roof, pad, left, right, mid);
  g.position.copy(cellToWorld(...cell));
  scene.add(g);

  addZone(scene, cell, 0xfacc15, 2.2);
}

// Landing pads: the shelters where the drone can land in a severe storm.
// The charging station is already a shelter, so it is skipped here.
function buildPads(scene) {
  for (const cell of LAYOUT.shelters || []) {
    if (cell[0] === LAYOUT.charger[0] && cell[1] === LAYOUT.charger[1]) continue;
    const pad = box(0.8, 0.06, 0.8, 0xa78bfa);
    pad.position.copy(cellToWorld(cell[0], cell[1], 0.03));
    scene.add(pad);
  }
}

// Obstacles: towers inside red no-fly zones. Kept so a hit can be animated.
const obstacleTowers = new Map();

function buildObstacles(scene) {
  obstacleTowers.clear();
  // Towers get a colour and width from their real height in the simulation.
  const colours = { 1: 0xbcc6d4, 2: 0x94a3b8, 3: 0x64748b, 4: 0x334155 };
  for (const entry of LAYOUT.obstacles) {
    // Each entry is [x, y, height]. Heights 1-3 are towers, 4 is a building.
    const [x, y, height = 1] = entry;
    const isBuilding = height >= 4;
    const top = towerTop(height);
    const width = isBuilding ? 0.85 : 0.5;
    const tower = box(width, top, width, colours[height] || colours[1]);
    tower.position.copy(cellToWorld(x, y, top / 2));
    scene.add(tower);
    addZone(scene, [x, y], isBuilding ? 0x7c3aed : 0xef4444, top + 0.2);
    obstacleTowers.set(`${x},${y}`, {
      mesh: tower, hit: 0, active: false, dirX: 0, dirZ: 0,
    });
  }
}

function buildCharger(scene) {
  const g = new THREE.Group();
  const pad = box(0.85, 0.06, 0.85, 0x16a34a);
  pad.position.y = 0.03;
  const post = box(0.12, 0.7, 0.12, 0x374151);
  post.position.y = 0.4;
  const head = box(0.4, 0.28, 0.2, 0x22c55e);
  head.position.y = 0.85;
  const bolt = new THREE.Mesh(
    new THREE.BoxGeometry(0.1, 0.2, 0.02),
    new THREE.MeshBasicMaterial({ color: 0xfde047 })
  );
  bolt.position.set(0, 0.85, 0.11);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.34, 0.42, 32),
    new THREE.MeshBasicMaterial({ color: 0x4ade80, transparent: true, opacity: 0.0, side: THREE.DoubleSide })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.08;
  g.add(pad, post, head, bolt, ring);
  g.position.copy(cellToWorld(...LAYOUT.charger));
  scene.add(g);
  return ring;
}

// ---------------------------------------------------------------------------
// Trees and vegetation (eucalyptus, broadleaf, pine, banana plants)
// ---------------------------------------------------------------------------

const GEO = {
  roundTrunk: new THREE.CylinderGeometry(0.05, 0.07, 0.5, 6),
  roundCanopy: new THREE.SphereGeometry(0.3, 8, 6),
  eucaTrunk: new THREE.CylinderGeometry(0.035, 0.055, 1.4, 6),
  eucaCanopy: new THREE.SphereGeometry(0.2, 8, 6),
  pineTrunk: new THREE.CylinderGeometry(0.04, 0.06, 0.4, 6),
  pineLow: new THREE.ConeGeometry(0.32, 0.65, 8),
  pineHigh: new THREE.ConeGeometry(0.23, 0.55, 8),
  bananaTrunk: new THREE.CylinderGeometry(0.03, 0.04, 0.35, 5),
  bananaLeaves: new THREE.SphereGeometry(0.28, 8, 4),
};

const BROWN = [0x6b4a2f, 0x7a5636];
const TREE_KINDS = {
  round: [
    ["roundTrunk", 0.25, [1, 1, 1], BROWN],
    ["roundCanopy", 0.7, [1, 1, 1], [0x2f8f46, 0x3aa356, 0x4aa84a, 0x287a3c]],
  ],
  eucalyptus: [
    ["eucaTrunk", 0.7, [1, 1, 1], [0x8d7a62, 0x9a8870]],
    ["eucaCanopy", 1.65, [1, 2.6, 1], [0x5f9b6b, 0x6aa877, 0x4f8c5d]],
  ],
  pine: [
    ["pineTrunk", 0.2, [1, 1, 1], BROWN],
    ["pineLow", 0.65, [1, 1, 1], [0x1f6b3a, 0x25773f, 0x1b5e34]],
    ["pineHigh", 1.05, [1, 1, 1], [0x1f6b3a, 0x25773f, 0x1b5e34]],
  ],
  banana: [
    ["bananaTrunk", 0.175, [1, 1, 1], [0x7a8f3a]],
    ["bananaLeaves", 0.42, [1, 0.3, 1], [0x8fd14f, 0x7dc243, 0x9bd95c]],
  ],
};

const treeItems = new Map();

function addTree(kind, x, z, scale = 1) {
  for (const [id, y, s, colors] of TREE_KINDS[kind]) {
    if (!treeItems.has(id)) treeItems.set(id, []);
    treeItems.get(id).push({
      x, z, y: y * scale,
      sx: s[0] * scale, sy: s[1] * scale, sz: s[2] * scale,
      yaw: rand() * Math.PI * 2,
      color: pick(colors),
    });
  }
}

function flushTrees(scene) {
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
  const matrix = new THREE.Matrix4();
  const quat = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const color = new THREE.Color();
  for (const [id, items] of treeItems) {
    const mesh = new THREE.InstancedMesh(GEO[id], material, items.length);
    items.forEach((t, i) => {
      quat.setFromAxisAngle(up, t.yaw);
      matrix.compose(
        new THREE.Vector3(t.x, t.y, t.z),
        quat,
        new THREE.Vector3(t.sx, t.sy, t.sz)
      );
      mesh.setMatrixAt(i, matrix);
      mesh.setColorAt(i, color.setHex(t.color));
    });
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
  }
  treeItems.clear();
}

// ---------------------------------------------------------------------------
// City, small town, towers, hills, forest and farmland
// ---------------------------------------------------------------------------

function flatStrip(scene, cx, cz, w, d, color, y = 0.012) {
  const strip = new THREE.Mesh(
    new THREE.BoxGeometry(w, 0.02, d),
    new THREE.MeshStandardMaterial({ color, roughness: 0.95 })
  );
  strip.position.set(cx, y, cz);
  strip.receiveShadow = true;
  scene.add(strip);
}

function buildCity(scene) {
  // Concrete slab and roads.
  const slab = box(12, 0.1, 36, 0xb8bec6);
  slab.position.set(-11.5 - M, -0.05, 0);
  scene.add(slab);

  const cols = [0, 1, 2, 3].map((i) => -7.5 - M - 2.6 * i);
  const rows = Array.from({ length: 14 }, (_, j) => -16.9 + 2.6 * j);

  for (const cx of cols) flatStrip(scene, cx + 1.3, 0, 0.5, 36, 0x55606e);
  for (const cz of rows) flatStrip(scene, -11.5 - M, cz + 1.3, 12, 0.5, 0x55606e);
  flatStrip(scene, -5 - M, 0, 0.6, 36, 0x55606e); // main road along the grid

  const palette = [0xe5e9ee, 0xd3dae2, 0xc2ccd6, 0xeef1f4, 0x9fb8cf];
  for (const cx of cols) {
    for (const cz of rows) {
      const x = cx + range(-0.15, 0.15);
      const z = cz + range(-0.15, 0.15);

      // A few lots become small parks.
      if (rand() < 0.1) {
        addTree("round", x - 0.3, z, range(1.2, 1.6));
        addTree("round", x + 0.4, z + 0.3, range(1, 1.4));
        continue;
      }

      const centre = Math.min(1, Math.hypot(cx + 11.4 + M, cz) / 18);
      let h = range(0.7, 2.2) + (1 - centre) * range(0, 3);
      if (rand() < 0.12) h = range(4.5, 8);

      const b = box(range(1.1, 1.7), h, range(1.1, 1.7), pick(palette));
      b.position.set(x, h / 2, z);
      scene.add(b);

      if (rand() < 0.4) {
        const unit = box(0.4, 0.2, 0.4, 0x9aa5b1);
        unit.position.set(x, h + 0.1, z);
        scene.add(unit);
      }
    }
  }
}

function buildTown(scene) {
  // Low houses with red tile roofs.
  flatStrip(scene, 0.5, 5.5 + M, 11, 0.5, 0x55606e);
  for (const rz of [8.2, 10.6]) flatStrip(scene, 0.5, rz + M, 11, 0.5, 0x55606e);

  const walls = [0xf3e9dc, 0xe9d8c0, 0xd9e3ea, 0xf6f1e7];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 3; j++) {
      const x = -3 + 2.6 * i + range(-0.1, 0.1);
      const z = 7 + M + 2.4 * j + range(-0.1, 0.1);
      const h = range(0.45, 0.8);
      const house = box(1.2, h, 1.0, pick(walls));
      house.position.set(x, h / 2, z);
      const roof = new THREE.Mesh(
        new THREE.ConeGeometry(0.95, 0.35, 4),
        new THREE.MeshStandardMaterial({ color: 0xb0583c, roughness: 0.8 })
      );
      roof.rotation.y = Math.PI / 4;
      roof.position.set(x, h + 0.17, z);
      roof.castShadow = true;
      scene.add(house, roof);
    }
  }
  for (let i = 0; i < 8; i++) {
    addTree(pick(["round", "eucalyptus"]), range(-4.5, 6), range(12.8 + M, 14.5 + M), range(1, 1.5));
  }
}

function buildTower(scene, x, z, height, blinkers) {
  const g = new THREE.Group();
  const bands = 6;
  const bandH = height / bands;
  for (let i = 0; i < bands; i++) {
    const r0 = 0.22 * (1 - i / bands) + 0.04;
    const r1 = 0.22 * (1 - (i + 1) / bands) + 0.04;
    const seg = new THREE.Mesh(
      new THREE.CylinderGeometry(r1, r0, bandH, 8),
      new THREE.MeshStandardMaterial({ color: i % 2 === 0 ? 0xdc2626 : 0xf8fafc, roughness: 0.6 })
    );
    seg.castShadow = true;
    seg.position.y = i * bandH + bandH / 2;
    g.add(seg);
  }
  const arm = box(0.55, 0.03, 0.03, 0x475569);
  arm.position.y = height * 0.85;
  g.add(arm);

  const light = new THREE.Mesh(
    new THREE.SphereGeometry(0.09, 10, 8),
    new THREE.MeshBasicMaterial({ color: 0xff2222 })
  );
  light.position.y = height + 0.09;
  g.add(light);
  blinkers.push(light.material);

  g.position.set(x, 0, z);
  scene.add(g);
}

function buildHill(scene, x, z, radius, height) {
  // Stacked terraces, like the farmed hills of Rwanda.
  const layers = 4;
  const colors = [0x5f9a4a, 0x7bb35a, 0x4f8a3e, 0x8cbf63];
  for (let i = 0; i < layers; i++) {
    const r = radius * (1 - i * 0.22);
    const h = height / layers;
    const terrace = new THREE.Mesh(
      new THREE.CylinderGeometry(r - 0.5, r, h, 24),
      new THREE.MeshStandardMaterial({ color: colors[i % colors.length], roughness: 1 })
    );
    terrace.position.set(x, i * h + h / 2 - 0.1, z);
    terrace.castShadow = true;
    terrace.receiveShadow = true;
    scene.add(terrace);
  }
}

function buildFarmland(scene) {
  // Banana plantation on terraced strips.
  for (let row = 0; row < 11; row++) {
    const z = 5.6 + M + row * 0.9;
    flatStrip(scene, 11 + M, z, 11, 0.8, row % 2 === 0 ? 0xa8854f : 0x86b85c, 0.015);
    for (let x = 6.4 + M; x < 16.5 + M; x += 0.95) {
      addTree("banana", x + range(-0.1, 0.1), z + range(-0.1, 0.1), range(0.9, 1.3));
    }
  }
}

function buildForest(scene) {
  // Red-earth road through the countryside.
  flatStrip(scene, 10.25, -6.5 - M, 31.5, 0.7, 0xb5654a);

  let placed = 0;
  let tries = 0;
  while (placed < 1000 && tries < 8000) {
    tries++;
    const x = range(-5.5 - M, 28);
    const z = range(-28, 5 + M);
    if (Math.abs(x) < 5.3 + M && Math.abs(z) < 5.3 + M) continue; // keep the mission area clear
    if (x > 5.5 + M && z > 4.3 + M) continue; // farmland
    if (Math.abs(z + 6.5 + M) < 0.6) continue; // red road
    if (Math.abs(x + 5 + M) < 0.5) continue; // main road

    const r = rand();
    const kind = r < 0.3 ? "pine" : r < 0.65 ? "eucalyptus" : "round";
    addTree(kind, x, z, range(1.1, 2.1));
    placed++;
  }
}

function buildLandscape(scene) {
  const blinkers = [];
  buildCity(scene);
  buildTown(scene);
  buildFarmland(scene);
  buildForest(scene);

  buildTower(scene, -8.8 - M, 2.6, 7, blinkers); // in the city
  buildTower(scene, -4.4 - M, -8 - M, 6, blinkers);
  buildTower(scene, 7.5 + M, -3.5, 6.5, blinkers);
  buildTower(scene, 6 + M, 5.8 + M, 5, blinkers);

  // Far hills.
  for (let i = 0; i < 10; i++) {
    const angle = (i / 10) * Math.PI * 2 + range(-0.2, 0.2);
    const dist = range(34, 52);
    buildHill(scene, Math.cos(angle) * dist, Math.sin(angle) * dist, range(6, 10), range(4, 7));
  }

  flushTrees(scene);
  return blinkers;
}

function buildRain(scene) {
  const count = 600;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - 0.5) * world.gridSize * 1.8;
    positions[i * 3 + 1] = Math.random() * 6;
    positions[i * 3 + 2] = (Math.random() - 0.5) * world.gridSize * 1.8;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0x4f7fc4, size: 0.06, transparent: true, opacity: 0.0,
  });
  const points = new THREE.Points(geometry, material);
  scene.add(points);
  return { points, positions, count };
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

export function createScene(canvas, layout) {
  LAYOUT = layout;
  world.gridSize = layout.grid_size;
  M = (layout.grid_size - 8) / 2;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(SKY);
  scene.fog = new THREE.Fog(SKY, 40, 110);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 300);
  camera.position.set(10, 11, 14);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.5, 0);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI / 2.05;
  controls.maxDistance = 70;

  scene.add(new THREE.HemisphereLight(0xdfeeff, 0x6b8f55, 0.7));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(12, 22, 9);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -30;
  sun.shadow.camera.right = 30;
  sun.shadow.camera.top = 30;
  sun.shadow.camera.bottom = -30;
  sun.shadow.camera.far = 80;
  scene.add(sun);

  buildGround(scene);
  buildBank(scene);
  // Every hospital the drone may visit (older files have only one).
  for (const cell of LAYOUT.hospitals || [LAYOUT.hospital]) {
    buildHospital(scene, cell);
  }
  buildObstacles(scene);
  buildPads(scene);
  const chargerRing = buildCharger(scene);
  const blinkers = buildLandscape(scene);
  const rain = buildRain(scene);

  // A blue disc on the ground marks each storm area (rain is local, not map-wide).
  const stormDiscs = [];
  for (let i = 0; i < 3; i++) {
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(1, 40),
      new THREE.MeshBasicMaterial({ color: 0x3b6fc9, transparent: true, opacity: 0.0, depthWrite: false }),
    );
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = 0.16;
    scene.add(disc);
    stormDiscs.push(disc);
  }

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // Dotted blue line that shows the route the drone follows.
  let routeLine = null;
  function setPath(frames) {
    if (routeLine) {
      scene.remove(routeLine);
      routeLine.geometry.dispose();
    }
    const points = [];
    let lastKey = "";
    for (const f of frames) {
      const key = `${f.x},${f.y},${f.altitude}`;
      if (key === lastKey) continue;
      lastKey = key;
      points.push(cellToWorld(f.x, f.y, altitudeHeight(f.altitude)));
    }
    routeLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineDashedMaterial({ color: 0x2b7fff, dashSize: 0.14, gapSize: 0.1 })
    );
    routeLine.computeLineDistances();
    scene.add(routeLine);
  }

  // Impact effects: sparks, tower wobble and camera shake ---------------
  const HIT_TIME = 1.2;
  const sparks = [];
  const sparkGeometry = new THREE.SphereGeometry(0.05, 6, 6);
  for (let i = 0; i < 48; i++) {
    const mesh = new THREE.Mesh(
      sparkGeometry,
      new THREE.MeshBasicMaterial({ color: i % 2 ? 0xfbbf24 : 0xef4444, transparent: true })
    );
    mesh.visible = false;
    scene.add(mesh);
    sparks.push({ mesh, vel: new THREE.Vector3(), life: 0 });
  }
  let cameraShake = 0;


  // (dirX, dirZ) is the direction the drone was flying when it hit the tower.
  function hitTower(cellX, cellY, dirX, dirZ, altitude = 0) {
    const tower = obstacleTowers.get(`${cellX},${cellY}`);
    if (tower) {
      tower.hit = HIT_TIME;
      tower.active = true;
      tower.dirX = dirX;
      tower.dirZ = dirZ;
    }

    const at = cellToWorld(cellX, cellY, altitudeHeight(altitude));
    at.x -= dirX * 0.3;
    at.z -= dirZ * 0.3;
    for (const s of sparks) {
      s.mesh.position.copy(at);
      s.vel.set(
        (Math.random() - 0.5) * 3 - dirX * 1.2,
        Math.random() * 3 + 0.5,
        (Math.random() - 0.5) * 3 - dirZ * 1.2
      );
      s.life = 0.9;
      s.mesh.visible = true;
    }
    cameraShake = 0.6;
  }

  function updateEffects(dt, time) {
    for (const t of obstacleTowers.values()) {
      if (!t.active) continue;
      t.hit -= dt;
      if (t.hit <= 0) {
        t.active = false;
        t.mesh.rotation.set(0, 0, 0);
        t.mesh.material.emissive.setRGB(0, 0, 0);
        continue;
      }
      const k = t.hit / HIT_TIME;
      const wobble = Math.sin(time * 38) * 0.14 * k;
      t.mesh.rotation.x = t.dirZ * wobble;
      t.mesh.rotation.z = -t.dirX * wobble;
      t.mesh.material.emissive.setRGB(k * 0.9, 0, 0);
    }

    for (const s of sparks) {
      if (s.life <= 0) continue;
      s.life -= dt;
      s.vel.y -= 9 * dt;
      s.mesh.position.addScaledVector(s.vel, dt);
      s.mesh.material.opacity = Math.max(s.life / 0.9, 0);
      if (s.life <= 0) s.mesh.visible = false;
    }
  }

  let clock = 0;
  let charging = false;
  const setCharging = (on) => { charging = on; };
  const shakeOffset = new THREE.Vector3();

  // storms: list of {x, y, radius, intensity} from the environment. Rain only
  // falls inside these areas, so the rest of the map stays dry.
  function update(dt, rainIntensity, storms = []) {
    clock += dt;

    // Tower warning lights blink.
    const on = clock % 1.4 < 0.7;
    for (const m of blinkers) m.color.setHex(on ? 0xff2222 : 0x4a1010);

    chargerRing.material.opacity = charging ? 0.5 + 0.5 * Math.sin(clock * 8) : 0;
    chargerRing.scale.setScalar(charging ? 1 + 0.25 * Math.sin(clock * 8) : 1);

    stormDiscs.forEach((disc, i) => {
      const storm = storms[i];
      disc.visible = Boolean(storm);
      if (!storm) return;
      const centre = cellToWorld(storm.x, storm.y);
      disc.position.x = centre.x;
      disc.position.z = centre.z;
      disc.scale.setScalar(storm.radius * CELL);
      disc.material.opacity = 0.28 * storm.intensity;
    });

    rain.points.material.opacity = storms.length ? 0.8 : 0;
    const pos = rain.positions;
    for (let i = 0; i < rain.count; i++) {
      pos[i * 3 + 1] -= dt * 8;
      if (pos[i * 3 + 1] < 0) {
        pos[i * 3 + 1] = 6;
        // A fallen drop starts again somewhere inside one of the storms.
        const storm = storms.length ? storms[i % storms.length] : null;
        if (storm) {
          const centre = cellToWorld(storm.x, storm.y);
          const angle = Math.random() * Math.PI * 2;
          const reach = Math.sqrt(Math.random()) * storm.radius * CELL;
          pos[i * 3] = centre.x + Math.cos(angle) * reach;
          pos[i * 3 + 2] = centre.z + Math.sin(angle) * reach;
        }
      }
    }
    rain.points.geometry.attributes.position.needsUpdate = true;

    updateEffects(dt, clock);
    controls.update();

    // Shake the camera for a moment, then put it back.
    shakeOffset.set(0, 0, 0);
    if (cameraShake > 0) {
      cameraShake -= dt;
      const a = cameraShake * 0.35;
      shakeOffset.set((Math.random() - 0.5) * a, (Math.random() - 0.5) * a, (Math.random() - 0.5) * a);
      camera.position.add(shakeOffset);
    }
    renderer.render(scene, camera);
    camera.position.sub(shakeOffset);
  }

  return { scene, camera, update, setPath, setCharging, hitTower };
}
