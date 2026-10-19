/**
 * Biosim Sandbox — Genetic Neural Network Evolution
 * Inspired by David R. Miller's "biosim4" video ("I programmed some creatures. They Evolved.")
 */

// --- Grid & Simulation Constants ---
const GRID_SIZE = 128;

const SENSORS = [
  { id: 0, code: 'LOC_X', label: 'East/West Pos (X)' },
  { id: 1, code: 'LOC_Y', label: 'North/South Pos (Y)' },
  { id: 2, code: 'BND_DIST', label: 'Nearest Border Dist' },
  { id: 3, code: 'AGE', label: 'Epoch Age (0→1)' },
  { id: 4, code: 'OSC', label: 'Oscillator Wave' },
  { id: 5, code: 'RND', label: 'Random Noise' },
  { id: 6, code: 'LAST_VX', label: 'Last Move X' },
  { id: 7, code: 'LAST_VY', label: 'Last Move Y' },
  { id: 8, code: 'FWD_BLK', label: 'Forward Blocked' },
  { id: 9, code: 'BIAS', label: 'Constant Bias (1.0)' }
];

const ACTIONS = [
  { id: 0, code: 'MV_E', label: 'Move East (+X)' },
  { id: 1, code: 'MV_W', label: 'Move West (-X)' },
  { id: 2, code: 'MV_S', label: 'Move South (+Y)' },
  { id: 3, code: 'MV_N', label: 'Move North (-Y)' },
  { id: 4, code: 'MV_FWD', label: 'Move Forward' },
  { id: 5, code: 'MV_RND', label: 'Move Random' },
  { id: 6, code: 'MV_X', label: 'Drive X (±)' },
  { id: 7, code: 'MV_Y', label: 'Drive Y (±)' }
];

// --- Mutable Configuration ---
const config = {
  popSize: 400,
  numGenes: 10,
  numHidden: 3,
  stepsPerGen: 250,
  mutationRate: 0.02, // probability per gene of mutating
  safeZoneType: 'right_strip',
  barrierPreset: 'center_wall',
  speedMode: 1, // 1, 5, 20, or 'max'
  toolMode: 'inspect' // 'inspect', 'wall', 'erase'
};

// --- Simulation State ---
let generation = 1;
let currentStep = 0;
let paused = false;
let survivalHistory = [];
let lastSurvivalRate = null;

// Grid arrays:
// wallsGrid: 1 if solid wall, 0 if empty
// occGrid: index of creature + 1 occupying cell, or 0 if empty
const wallsGrid = new Uint8Array(GRID_SIZE * GRID_SIZE);
const occGrid = new Int32Array(GRID_SIZE * GRID_SIZE);

let creatures = [];
let selectedCreature = null;

// --- DOM Elements ---
const arenaCanvas = document.getElementById('arena-canvas');
const arenaCtx = arenaCanvas.getContext('2d');
const brainCanvas = document.getElementById('brain-canvas');
const brainCtx = brainCanvas.getContext('2d');
const chartCanvas = document.getElementById('chart-canvas');
const chartCtx = chartCanvas.getContext('2d');

const statGenEl = document.getElementById('stat-gen');
const statStepEl = document.getElementById('stat-step');
const statSurvivalEl = document.getElementById('stat-survival');
const statInsideEl = document.getElementById('stat-inside');

const genomePillsEl = document.getElementById('genome-pills');
const dotColorSwatchEl = document.getElementById('dot-color-swatch');
const inspectorStatusEl = document.getElementById('inspector-status');

// --- Helper Math & Gene Functions ---
function clamp(v, min, max) {
  return v < min ? min : v > max ? max : v;
}

function createRandomGene(numHidden) {
  return {
    sourceType: Math.random() < 0.72 ? 0 : 1, // 0 = Sensor, 1 = Hidden neuron
    sourceId: Math.floor(Math.random() * 100),
    sinkType: Math.random() < 0.72 ? 1 : 0,   // 0 = Hidden neuron, 1 = Action neuron
    sinkId: Math.floor(Math.random() * 100),
    weight: (Math.random() * 8) - 4           // range [-4.0, +4.0]
  };
}

function cloneGene(g) {
  return {
    sourceType: g.sourceType,
    sourceId: g.sourceId,
    sinkType: g.sinkType,
    sinkId: g.sinkId,
    weight: g.weight
  };
}

function geneToHex(g, numHidden) {
  const srcIdx = g.sourceType === 0 ? (g.sourceId % SENSORS.length) : (g.sourceId % numHidden);
  const snkIdx = g.sinkType === 0 ? (g.sinkId % numHidden) : (g.sinkId % ACTIONS.length);
  const wInt = Math.round(clamp((g.weight + 4) / 8, 0, 1) * 65535);
  const byte0 = ((g.sourceType & 1) << 7) | (srcIdx & 0x7f);
  const byte1 = ((g.sinkType & 1) << 7) | (snkIdx & 0x7f);
  const hex = (
    byte0.toString(16).padStart(2, '0') +
    byte1.toString(16).padStart(2, '0') +
    wInt.toString(16).padStart(4, '0')
  ).toUpperCase();
  return hex;
}

/**
 * Compute an RGB color fingerprint from a creature's genome so genetically
 * related creatures share visually similar colors.
 */
function computeGenomeColor(genome, numHidden) {
  let rHash = 0;
  let gHash = 0;
  let bHash = 0;

  for (let i = 0; i < genome.length; i++) {
    const g = genome[i];
    const sIdx = g.sourceType === 0 ? (g.sourceId % SENSORS.length) : (g.sourceId % numHidden) + 16;
    const tIdx = g.sinkType === 1 ? (g.sinkId % ACTIONS.length) : (g.sinkId % numHidden) + 16;
    const wSign = g.weight >= 0 ? 1 : -1;
    const wMag = Math.abs(g.weight);

    rHash += (sIdx * 37 + tIdx * 13) * wSign * (0.5 + wMag * 0.25);
    gHash += (tIdx * 53 + sIdx * 19) * wSign * (0.5 + wMag * 0.25);
    bHash += ((sIdx ^ tIdx) * 41 + 17) * g.weight;
  }

  // Map hashes into bright, readable HSL/RGB colors
  const h = ((Math.round(Math.abs(rHash * 11 + gHash * 7)) % 360) + 360) % 360;
  const s = 72 + (Math.round(Math.abs(bHash)) % 22);
  const l = 55 + (Math.round(Math.abs(gHash)) % 15);
  return `hsl(${h}, ${s}%, ${l}%)`;
}

// --- Creature Class ---
class Creature {
  constructor(x, y, genome, numHidden) {
    this.x = x;
    this.y = y;
    this.vx = (Math.random() < 0.5 ? -1 : 1);
    this.vy = (Math.random() < 0.5 ? -1 : 1);
    this.genome = genome;
    this.numHidden = numHidden;
    this.hiddenState = new Float32Array(numHidden);
    this.nextHidden = new Float32Array(numHidden);
    this.lastSensorVals = new Float32Array(SENSORS.length);
    this.lastActionVals = new Float32Array(ACTIONS.length);
    this.color = computeGenomeColor(genome, numHidden);
  }

  step(stepIdx, totalSteps) {
    const s = this.lastSensorVals;
    const invGrid = 1 / (GRID_SIZE - 1);

    // 1. Compute Sensory Inputs [0..1] or [-1..1]
    s[0] = this.x * invGrid; // LOC_X
    s[1] = this.y * invGrid; // LOC_Y

    const distX = Math.min(this.x, GRID_SIZE - 1 - this.x);
    const distY = Math.min(this.y, GRID_SIZE - 1 - this.y);
    s[2] = (Math.min(distX, distY) * 2) * invGrid; // BND_DIST (0 at wall, 1 at center)
    s[3] = stepIdx / totalSteps; // AGE
    s[4] = Math.sin(stepIdx * 0.18); // OSC
    s[5] = Math.random() * 2 - 1; // RND
    s[6] = this.vx; // LAST_VX
    s[7] = this.vy; // LAST_VY

    // Check if cell ahead in current heading is blocked
    const fx = this.x + this.vx;
    const fy = this.y + this.vy;
    const fwdBlocked = !isCellFree(fx, fy);
    s[8] = fwdBlocked ? 1.0 : 0.0; // FWD_BLK
    s[9] = 1.0; // BIAS

    // 2. Accumulate synapses into Hidden Neurons first
    const nHid = this.numHidden;
    const nextH = this.nextHidden;
    nextH.fill(0);

    const actionAccum = this.lastActionVals;
    actionAccum.fill(0);

    const genes = this.genome;
    for (let i = 0; i < genes.length; i++) {
      const g = genes[i];
      if (g.sinkType === 0) {
        const srcVal = g.sourceType === 0
          ? s[g.sourceId % SENSORS.length]
          : this.hiddenState[g.sourceId % nHid];
        nextH[g.sinkId % nHid] += srcVal * g.weight;
      }
    }

    for (let h = 0; h < nHid; h++) {
      this.hiddenState[h] = Math.tanh(nextH[h]);
    }

    // 3. Accumulate synapses into Action Neurons
    for (let i = 0; i < genes.length; i++) {
      const g = genes[i];
      if (g.sinkType === 1) {
        const srcVal = g.sourceType === 0
          ? s[g.sourceId % SENSORS.length]
          : this.hiddenState[g.sourceId % nHid];
        actionAccum[g.sinkId % ACTIONS.length] += srcVal * g.weight;
      }
    }

    for (let a = 0; a < ACTIONS.length; a++) {
      actionAccum[a] = Math.tanh(actionAccum[a]);
    }

    // 4. Resolve Desired Movement (dx, dy)
    let driveX = actionAccum[0] - actionAccum[1] + actionAccum[6]; // East - West + DriveX
    let driveY = actionAccum[2] - actionAccum[3] + actionAccum[7]; // South - North + DriveY

    if (actionAccum[4] > 0) {
      driveX += this.vx * actionAccum[4];
      driveY += this.vy * actionAccum[4];
    }

    if (actionAccum[5] > 0.15 && Math.random() < actionAccum[5]) {
      driveX += (Math.random() * 2 - 1);
      driveY += (Math.random() * 2 - 1);
    }

    let dx = 0;
    let dy = 0;

    // Probabilistic activation like biosim4: magnitude of tanh output is probability of step
    const probX = Math.min(1, Math.abs(driveX));
    const probY = Math.min(1, Math.abs(driveY));

    if (Math.random() < probX) {
      dx = driveX >= 0 ? 1 : -1;
    }
    if (Math.random() < probY) {
      dy = driveY >= 0 ? 1 : -1;
    }

    if (dx !== 0 || dy !== 0) {
      const curIdx = this.y * GRID_SIZE + this.x;
      const creatureId = occGrid[curIdx];
      let moved = false;

      // Try diagonal/combined move first
      const nx = this.x + dx;
      const ny = this.y + dy;
      if (isCellFree(nx, ny)) {
        occGrid[curIdx] = 0;
        this.x = nx;
        this.y = ny;
        occGrid[ny * GRID_SIZE + nx] = creatureId;
        moved = true;
      } else if (dx !== 0 && isCellFree(this.x + dx, this.y)) {
        // Slide horizontally along obstacle
        occGrid[curIdx] = 0;
        this.x += dx;
        occGrid[this.y * GRID_SIZE + this.x] = creatureId;
        moved = true;
      } else if (dy !== 0 && isCellFree(this.x, this.y + dy)) {
        // Slide vertically along obstacle
        occGrid[curIdx] = 0;
        this.y += dy;
        occGrid[this.y * GRID_SIZE + this.x] = creatureId;
        moved = true;
      }

      if (moved) {
        if (dx !== 0) this.vx = dx;
        if (dy !== 0) this.vy = dy;
      }
    }
  }
}

// --- Environment & Barriers ---
function isCellFree(x, y) {
  if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE) return false;
  const idx = y * GRID_SIZE + x;
  return wallsGrid[idx] === 0 && occGrid[idx] === 0;
}

function isInsideSafeZone(x, y) {
  const half = GRID_SIZE / 2;
  switch (config.safeZoneType) {
    case 'right_strip':
      return x >= Math.floor(GRID_SIZE * 0.8);
    case 'right_half':
      return x >= half;
    case 'left_right':
      return x < Math.floor(GRID_SIZE * 0.16) || x >= Math.floor(GRID_SIZE * 0.84);
    case 'center_circle': {
      const dx = x - half;
      const dy = y - half;
      const r = GRID_SIZE * 0.24;
      return (dx * dx + dy * dy) <= r * r;
    }
    case 'corners': {
      const c = Math.floor(GRID_SIZE * 0.24);
      const inLeft = x < c;
      const inRight = x >= GRID_SIZE - c;
      const inTop = y < c;
      const inBottom = y >= GRID_SIZE - c;
      return (inLeft || inRight) && (inTop || inBottom);
    }
    case 'top_left':
      return x < Math.floor(GRID_SIZE * 0.32) && y < Math.floor(GRID_SIZE * 0.32);
    default:
      return x >= half;
  }
}

function buildBarrierPreset(preset) {
  wallsGrid.fill(0);
  const mid = Math.floor(GRID_SIZE / 2);

  const setWallRect = (x0, y0, x1, y1) => {
    const minX = clamp(Math.min(x0, x1), 0, GRID_SIZE - 1);
    const maxX = clamp(Math.max(x0, x1), 0, GRID_SIZE - 1);
    const minY = clamp(Math.min(y0, y1), 0, GRID_SIZE - 1);
    const maxY = clamp(Math.max(y0, y1), 0, GRID_SIZE - 1);
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        wallsGrid[y * GRID_SIZE + x] = 1;
      }
    }
  };

  if (preset === 'center_wall') {
    setWallRect(mid - 1, Math.floor(GRID_SIZE * 0.22), mid + 1, Math.floor(GRID_SIZE * 0.78));
  } else if (preset === 'funnel') {
    setWallRect(mid - 1, 0, mid + 1, Math.floor(GRID_SIZE * 0.38));
    setWallRect(mid - 1, Math.floor(GRID_SIZE * 0.62), mid + 1, GRID_SIZE - 1);
  } else if (preset === 'staggered') {
    const x1 = Math.floor(GRID_SIZE * 0.35);
    const x2 = Math.floor(GRID_SIZE * 0.65);
    setWallRect(x1 - 1, Math.floor(GRID_SIZE * 0.18), x1 + 1, Math.floor(GRID_SIZE * 0.82));
    setWallRect(x2 - 1, 0, x2 + 1, Math.floor(GRID_SIZE * 0.38));
    setWallRect(x2 - 1, Math.floor(GRID_SIZE * 0.62), x2 + 1, GRID_SIZE - 1);
  } else if (preset === 'box_trap') {
    const cx = Math.floor(GRID_SIZE * 0.58);
    const top = Math.floor(GRID_SIZE * 0.25);
    const bot = Math.floor(GRID_SIZE * 0.75);
    const left = Math.floor(GRID_SIZE * 0.32);
    setWallRect(cx - 1, top, cx + 1, bot);
    setWallRect(left, top, cx + 1, top + 2);
    setWallRect(left, bot - 2, cx + 1, bot);
  }
}

// --- Population Spawning & Reproduction ---
function findRandomEmptyCell() {
  for (let tries = 0; tries < 500; tries++) {
    const x = Math.floor(Math.random() * GRID_SIZE);
    const y = Math.floor(Math.random() * GRID_SIZE);
    const idx = y * GRID_SIZE + x;
    if (wallsGrid[idx] === 0 && occGrid[idx] === 0) {
      return { x, y };
    }
  }
  // Fallback linear scan
  for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
    if (wallsGrid[i] === 0 && occGrid[i] === 0) {
      return { x: i % GRID_SIZE, y: Math.floor(i / GRID_SIZE) };
    }
  }
  return { x: 0, y: 0 };
}

function initSimulation() {
  generation = 1;
  currentStep = 0;
  survivalHistory = [];
  lastSurvivalRate = null;
  occGrid.fill(0);
  creatures = [];

  for (let i = 0; i < config.popSize; i++) {
    const pos = findRandomEmptyCell();
    const genome = [];
    for (let g = 0; g < config.numGenes; g++) {
      genome.push(createRandomGene(config.numHidden));
    }
    const c = new Creature(pos.x, pos.y, genome, config.numHidden);
    creatures.push(c);
    occGrid[pos.y * GRID_SIZE + pos.x] = i + 1;
  }

  selectedCreature = creatures[0] || null;
  updateInspectorUI();
  updateStatsUI();
  drawChart();
}

function breedChildGenome(parentA, parentB) {
  const targetLen = config.numGenes;
  const childGenome = [];

  for (let i = 0; i < targetLen; i++) {
    const srcParent = Math.random() < 0.5 ? parentA : parentB;
    const baseGene = srcParent.genome[i % srcParent.genome.length]
      ? cloneGene(srcParent.genome[i % srcParent.genome.length])
      : createRandomGene(config.numHidden);

    // Apply mutations
    if (Math.random() < config.mutationRate) {
      const mutType = Math.random();
      if (mutType < 0.55) {
        // Perturb synaptic weight slightly
        baseGene.weight = clamp(baseGene.weight + (Math.random() * 2 - 1) * 1.2, -4.0, 4.0);
      } else if (mutType < 0.75) {
        // Flip source neuron
        baseGene.sourceType = Math.random() < 0.72 ? 0 : 1;
        baseGene.sourceId = Math.floor(Math.random() * 100);
      } else if (mutType < 0.95) {
        // Flip sink neuron
        baseGene.sinkType = Math.random() < 0.72 ? 1 : 0;
        baseGene.sinkId = Math.floor(Math.random() * 100);
      } else {
        // Brand new random synapse
        Object.assign(baseGene, createRandomGene(config.numHidden));
      }
    }

    childGenome.push(baseGene);
  }

  return childGenome;
}

function advanceGeneration() {
  // 1. Identify survivors inside the green safe zone
  const survivors = [];
  for (let i = 0; i < creatures.length; i++) {
    const c = creatures[i];
    if (isInsideSafeZone(c.x, c.y)) {
      survivors.push(c);
    }
  }

  const rate = creatures.length > 0 ? (survivors.length / creatures.length) * 100 : 0;
  lastSurvivalRate = rate;
  survivalHistory.push(rate);
  if (survivalHistory.length > 150) {
    survivalHistory.shift();
  }

  // Remember if selected creature survived so we can follow its lineage
  const prevSelectedGenome = selectedCreature ? selectedCreature.genome : null;

  // 2. Spawn next generation
  occGrid.fill(0);
  const nextCreatures = [];

  for (let i = 0; i < config.popSize; i++) {
    const pos = findRandomEmptyCell();
    let genome;

    if (survivors.length === 0) {
      // Total wipeout fallback: spawn fresh random genomes
      genome = [];
      for (let g = 0; g < config.numGenes; g++) {
        genome.push(createRandomGene(config.numHidden));
      }
    } else if (i === 0 && survivors.length > 0) {
      // Keep 1 unmutated elite survivor so we can also track lineage
      const elite = survivors[Math.floor(Math.random() * survivors.length)];
      genome = elite.genome.slice(0, config.numGenes).map(cloneGene);
      while (genome.length < config.numGenes) {
        genome.push(createRandomGene(config.numHidden));
      }
    } else {
      const pA = survivors[Math.floor(Math.random() * survivors.length)];
      const pB = survivors[Math.floor(Math.random() * survivors.length)];
      genome = breedChildGenome(pA, pB);
    }

    const child = new Creature(pos.x, pos.y, genome, config.numHidden);
    nextCreatures.push(child);
    occGrid[pos.y * GRID_SIZE + pos.x] = i + 1;
  }

  creatures = nextCreatures;
  generation++;
  currentStep = 0;

  if (!selectedCreature || !creatures.includes(selectedCreature)) {
    selectedCreature = creatures[0] || null;
  }

  updateInspectorUI();
  drawChart();
}

function stepSingleTick() {
  if (currentStep >= config.stepsPerGen) {
    advanceGeneration();
    return;
  }

  for (let i = 0; i < creatures.length; i++) {
    creatures[i].step(currentStep, config.stepsPerGen);
  }
  currentStep++;
}

// --- Rendering Functions ---
function drawArena() {
  const w = arenaCanvas.width;
  const h = arenaCanvas.height;
  const cellScale = w / GRID_SIZE;

  arenaCtx.fillStyle = '#0d1322';
  arenaCtx.fillRect(0, 0, w, h);

  // 1. Draw subtle grid lines
  arenaCtx.strokeStyle = 'rgba(42, 57, 92, 0.22)';
  arenaCtx.lineWidth = 1;
  const gridStep = 16 * cellScale;
  arenaCtx.beginPath();
  for (let p = gridStep; p < w; p += gridStep) {
    arenaCtx.moveTo(p, 0);
    arenaCtx.lineTo(p, h);
    arenaCtx.moveTo(0, p);
    arenaCtx.lineTo(w, p);
  }
  arenaCtx.stroke();

  // 2. Draw Safe Survival Zone
  arenaCtx.fillStyle = 'rgba(34, 197, 94, 0.16)';
  arenaCtx.strokeStyle = 'rgba(34, 197, 94, 0.55)';
  arenaCtx.lineWidth = 1.5;

  const half = w / 2;
  if (config.safeZoneType === 'right_strip') {
    const x0 = Math.floor(GRID_SIZE * 0.8) * cellScale;
    arenaCtx.fillRect(x0, 0, w - x0, h);
    arenaCtx.strokeRect(x0, 0, w - x0, h);
  } else if (config.safeZoneType === 'right_half') {
    arenaCtx.fillRect(half, 0, half, h);
    arenaCtx.strokeRect(half, 0, half, h);
  } else if (config.safeZoneType === 'left_right') {
    const wEdge = Math.floor(GRID_SIZE * 0.16) * cellScale;
    const eEdge = Math.floor(GRID_SIZE * 0.84) * cellScale;
    arenaCtx.fillRect(0, 0, wEdge, h);
    arenaCtx.strokeRect(0, 0, wEdge, h);
    arenaCtx.fillRect(eEdge, 0, w - eEdge, h);
    arenaCtx.strokeRect(eEdge, 0, w - eEdge, h);
  } else if (config.safeZoneType === 'center_circle') {
    const r = GRID_SIZE * 0.24 * cellScale;
    arenaCtx.beginPath();
    arenaCtx.arc(half, half, r, 0, Math.PI * 2);
    arenaCtx.fill();
    arenaCtx.stroke();
  } else if (config.safeZoneType === 'corners') {
    const c = Math.floor(GRID_SIZE * 0.24) * cellScale;
    const coords = [
      [0, 0],
      [w - c, 0],
      [0, h - c],
      [w - c, h - c]
    ];
    coords.forEach(([cx, cy]) => {
      arenaCtx.fillRect(cx, cy, c, c);
      arenaCtx.strokeRect(cx, cy, c, c);
    });
  } else if (config.safeZoneType === 'top_left') {
    const c = Math.floor(GRID_SIZE * 0.32) * cellScale;
    arenaCtx.fillRect(0, 0, c, c);
    arenaCtx.strokeRect(0, 0, c, c);
  }

  // 3. Draw Walls
  arenaCtx.fillStyle = '#475569';
  for (let y = 0; y < GRID_SIZE; y++) {
    const rowOffset = y * GRID_SIZE;
    for (let x = 0; x < GRID_SIZE; x++) {
      if (wallsGrid[rowOffset + x] === 1) {
        arenaCtx.fillRect(x * cellScale, y * cellScale, cellScale + 0.4, cellScale + 0.4);
      }
    }
  }

  // 4. Draw Creatures
  const radius = Math.max(2.2, cellScale * 0.46);
  for (let i = 0; i < creatures.length; i++) {
    const c = creatures[i];
    const cx = (c.x + 0.5) * cellScale;
    const cy = (c.y + 0.5) * cellScale;

    arenaCtx.beginPath();
    arenaCtx.arc(cx, cy, radius, 0, Math.PI * 2);
    arenaCtx.fillStyle = c.color;
    arenaCtx.fill();
  }

  // 5. Highlight Selected Creature
  if (selectedCreature) {
    const cx = (selectedCreature.x + 0.5) * cellScale;
    const cy = (selectedCreature.y + 0.5) * cellScale;

    arenaCtx.beginPath();
    arenaCtx.arc(cx, cy, radius + 4.5, 0, Math.PI * 2);
    arenaCtx.strokeStyle = '#f59e0b';
    arenaCtx.lineWidth = 2.2;
    arenaCtx.stroke();

    arenaCtx.beginPath();
    arenaCtx.arc(cx, cy, radius + 1.5, 0, Math.PI * 2);
    arenaCtx.strokeStyle = '#ffffff';
    arenaCtx.lineWidth = 1.2;
    arenaCtx.stroke();
  }
}

function drawBrainInspector() {
  const w = brainCanvas.width;
  const h = brainCanvas.height;
  brainCtx.clearRect(0, 0, w, h);

  if (!selectedCreature) return;

  const c = selectedCreature;
  const nHid = c.numHidden;

  // Collect active nodes that participate in at least one gene
  const activeSensors = new Set();
  const activeHidden = new Set();
  const activeActions = new Set();

  c.genome.forEach(g => {
    if (g.sourceType === 0) activeSensors.add(g.sourceId % SENSORS.length);
    else activeHidden.add(g.sourceId % nHid);

    if (g.sinkType === 0) activeHidden.add(g.sinkId % nHid);
    else activeActions.add(g.sinkId % ACTIONS.length);
  });

  const sList = Array.from(activeSensors).sort((a, b) => a - b);
  const hList = Array.from(activeHidden).sort((a, b) => a - b);
  const aList = Array.from(activeActions).sort((a, b) => a - b);

  const colSensorX = 72;
  const colHiddenX = w / 2;
  const colActionX = w - 72;

  const sensorPos = new Map();
  const hiddenPos = new Map();
  const actionPos = new Map();

  const layoutColumn = (list, x, map) => {
    const count = list.length;
    if (count === 0) return;
    const topPad = 24;
    const usableH = h - topPad * 2;
    list.forEach((id, idx) => {
      const y = count === 1 ? h / 2 : topPad + (idx / (count - 1)) * usableH;
      map.set(id, { x, y });
    });
  };

  layoutColumn(sList, colSensorX, sensorPos);
  layoutColumn(hList, colHiddenX, hiddenPos);
  layoutColumn(aList, colActionX, actionPos);

  // Draw Synaptic Connections
  c.genome.forEach(g => {
    const srcId = g.sourceType === 0 ? (g.sourceId % SENSORS.length) : (g.sourceId % nHid);
    const snkId = g.sinkType === 0 ? (g.sinkId % nHid) : (g.sinkId % ACTIONS.length);

    const p1 = g.sourceType === 0 ? sensorPos.get(srcId) : hiddenPos.get(srcId);
    const p2 = g.sinkType === 0 ? hiddenPos.get(snkId) : actionPos.get(snkId);
    if (!p1 || !p2) return;

    const isPositive = g.weight >= 0;
    const lineW = Math.max(1, Math.abs(g.weight) * 0.95);
    brainCtx.strokeStyle = isPositive
      ? 'rgba(34, 197, 94, 0.75)'
      : 'rgba(239, 68, 68, 0.75)';
    brainCtx.lineWidth = lineW;

    brainCtx.beginPath();
    if (g.sourceType === 1 && g.sinkType === 0 && srcId === snkId) {
      // Self-loop on hidden neuron
      brainCtx.arc(p1.x, p1.y - 12, 10, 0, Math.PI * 2);
    } else {
      const midX = (p1.x + p2.x) / 2;
      brainCtx.moveTo(p1.x, p1.y);
      brainCtx.bezierCurveTo(midX, p1.y, midX, p2.y, p2.x, p2.y);
    }
    brainCtx.stroke();
  });

  // Helper to draw a neuron node
  const drawNode = (x, y, code, activation, align) => {
    const r = 11;
    const actClamp = clamp(activation || 0, -1, 1);

    brainCtx.beginPath();
    brainCtx.arc(x, y, r, 0, Math.PI * 2);
    if (actClamp >= 0) {
      const alpha = 0.2 + actClamp * 0.65;
      brainCtx.fillStyle = `rgba(56, 189, 248, ${alpha})`;
    } else {
      const alpha = 0.2 + Math.abs(actClamp) * 0.65;
      brainCtx.fillStyle = `rgba(244, 63, 94, ${alpha})`;
    }
    brainCtx.fill();
    brainCtx.strokeStyle = '#94a3b8';
    brainCtx.lineWidth = 1.5;
    brainCtx.stroke();

    // Label
    brainCtx.fillStyle = '#e2e8f0';
    brainCtx.font = '600 10px "Segoe UI", sans-serif';
    brainCtx.textBaseline = 'middle';
    if (align === 'left') {
      brainCtx.textAlign = 'right';
      brainCtx.fillText(code, x - 15, y);
    } else if (align === 'right') {
      brainCtx.textAlign = 'left';
      brainCtx.fillText(code, x + 15, y);
    } else {
      brainCtx.textAlign = 'center';
      brainCtx.fillText(code, x, y);
    }
  };

  sList.forEach(id => {
    const p = sensorPos.get(id);
    drawNode(p.x, p.y, SENSORS[id].code, c.lastSensorVals[id], 'left');
  });

  hList.forEach(id => {
    const p = hiddenPos.get(id);
    drawNode(p.x, p.y, `H${id}`, c.hiddenState[id], 'center');
  });

  aList.forEach(id => {
    const p = actionPos.get(id);
    drawNode(p.x, p.y, ACTIONS[id].code, c.lastActionVals[id], 'right');
  });
}

function updateInspectorUI() {
  genomePillsEl.innerHTML = '';
  if (!selectedCreature) {
    inspectorStatusEl.textContent = 'Click any dot in the arena to inspect its neural wiring.';
    return;
  }

  const c = selectedCreature;
  const inside = isInsideSafeZone(c.x, c.y);
  inspectorStatusEl.textContent = `Selected Dot @ (${c.x}, ${c.y}) — ${inside ? 'Inside Safe Zone' : 'Outside Safe Zone'}`;
  dotColorSwatchEl.style.background = c.color;

  c.genome.forEach(g => {
    const pill = document.createElement('span');
    pill.className = `gene-pill ${g.weight >= 0 ? 'pos' : 'neg'}`;
    const srcName = g.sourceType === 0
      ? SENSORS[g.sourceId % SENSORS.length].code
      : `H${g.sourceId % c.numHidden}`;
    const snkName = g.sinkType === 0
      ? `H${g.sinkId % c.numHidden}`
      : ACTIONS[g.sinkId % ACTIONS.length].code;
    const hex = geneToHex(g, c.numHidden);
    pill.textContent = hex;
    pill.title = `${srcName} → ${snkName} (weight: ${g.weight >= 0 ? '+' : ''}${g.weight.toFixed(2)})`;
    genomePillsEl.appendChild(pill);
  });
}

function drawChart() {
  const w = chartCanvas.width;
  const h = chartCanvas.height;
  chartCtx.clearRect(0, 0, w, h);

  const padL = 32;
  const padR = 10;
  const padT = 12;
  const padB = 20;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Grid lines (0%, 50%, 100%)
  chartCtx.strokeStyle = 'rgba(42, 57, 92, 0.5)';
  chartCtx.fillStyle = '#8b9bb4';
  chartCtx.font = '10px "Segoe UI", sans-serif';
  chartCtx.textAlign = 'right';
  chartCtx.textBaseline = 'middle';

  [0, 50, 100].forEach(pct => {
    const y = padT + plotH - (pct / 100) * plotH;
    chartCtx.beginPath();
    chartCtx.moveTo(padL, y);
    chartCtx.lineTo(w - padR, y);
    chartCtx.stroke();
    chartCtx.fillText(`${pct}%`, padL - 5, y);
  });

  if (survivalHistory.length === 0) {
    chartCtx.fillStyle = '#64748b';
    chartCtx.textAlign = 'center';
    chartCtx.fillText('Complete Generation 1 to plot survival curve', padL + plotW / 2, padT + plotH / 2);
    return;
  }

  // Plot area & line
  chartCtx.beginPath();
  survivalHistory.forEach((rate, idx) => {
    const x = padL + (survivalHistory.length === 1 ? plotW / 2 : (idx / (survivalHistory.length - 1)) * plotW);
    const y = padT + plotH - (clamp(rate, 0, 100) / 100) * plotH;
    if (idx === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  });
  chartCtx.strokeStyle = '#22c55e';
  chartCtx.lineWidth = 2;
  chartCtx.stroke();

  // Fill gradient under curve
  if (survivalHistory.length > 1) {
    chartCtx.lineTo(padL + plotW, padT + plotH);
    chartCtx.lineTo(padL, padT + plotH);
    chartCtx.closePath();
    chartCtx.fillStyle = 'rgba(34, 197, 94, 0.15)';
    chartCtx.fill();
  }
}

function updateStatsUI() {
  statGenEl.textContent = generation;
  statStepEl.textContent = `${currentStep} / ${config.stepsPerGen}`;
  statSurvivalEl.textContent = lastSurvivalRate !== null ? `${lastSurvivalRate.toFixed(1)}%` : '—';

  let insideCount = 0;
  for (let i = 0; i < creatures.length; i++) {
    if (isInsideSafeZone(creatures[i].x, creatures[i].y)) insideCount++;
  }
  const insidePct = creatures.length > 0 ? Math.round((insideCount / creatures.length) * 100) : 0;
  statInsideEl.textContent = `${insidePct}%`;

  if (selectedCreature) {
    const inside = isInsideSafeZone(selectedCreature.x, selectedCreature.y);
    inspectorStatusEl.textContent = `Selected Dot @ (${selectedCreature.x}, ${selectedCreature.y}) — ${inside ? 'Inside Safe Zone' : 'Outside Safe Zone'}`;
  }
}

// --- Main Animation Loop ---
function animate() {
  if (!paused) {
    if (config.speedMode === 'max') {
      // Run an entire generation per animation frame for rapid evolution!
      const remaining = config.stepsPerGen - currentStep;
      for (let s = 0; s < remaining; s++) {
        stepSingleTick();
      }
      advanceGeneration();
    } else {
      const ticks = Number(config.speedMode) || 1;
      for (let s = 0; s < ticks; s++) {
        stepSingleTick();
      }
    }
  }

  drawArena();
  drawBrainInspector();
  updateStatsUI();

  requestAnimationFrame(animate);
}

// --- Event Listeners & Interactive Controls ---
document.getElementById('btn-play-pause').addEventListener('click', (e) => {
  paused = !paused;
  e.target.textContent = paused ? 'Resume' : 'Pause';
});

document.getElementById('btn-step-gen').addEventListener('click', () => {
  while (currentStep < config.stepsPerGen) {
    stepSingleTick();
  }
  advanceGeneration();
  drawArena();
  drawBrainInspector();
  updateStatsUI();
});

document.getElementById('btn-reset').addEventListener('click', () => {
  initSimulation();
});

document.getElementById('speed-control').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  document.querySelectorAll('#speed-control button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const sp = btn.dataset.speed;
  config.speedMode = sp === 'max' ? 'max' : parseInt(sp, 10);
});

document.getElementById('tool-control').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  document.querySelectorAll('#tool-control button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  config.toolMode = btn.dataset.tool;
});

document.getElementById('select-zone').addEventListener('change', (e) => {
  config.safeZoneType = e.target.value;
});

document.getElementById('select-barrier').addEventListener('change', (e) => {
  config.barrierPreset = e.target.value;
  buildBarrierPreset(config.barrierPreset);
});

document.getElementById('btn-clear-walls').addEventListener('click', () => {
  wallsGrid.fill(0);
  document.getElementById('select-barrier').value = 'none';
});

document.getElementById('btn-pick-random').addEventListener('click', () => {
  if (creatures.length > 0) {
    selectedCreature = creatures[Math.floor(Math.random() * creatures.length)];
    updateInspectorUI();
  }
});

// Sliders
function bindSlider(sliderId, valId, key, formatter = v => v) {
  const slider = document.getElementById(sliderId);
  const valSpan = document.getElementById(valId);
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    config[key] = key === 'mutationRate' ? v / 100 : v;
    valSpan.textContent = formatter(v);
  });
}

bindSlider('slider-pop', 'val-pop', 'popSize');
bindSlider('slider-genes', 'val-genes', 'numGenes');
bindSlider('slider-hidden', 'val-hidden', 'numHidden');
bindSlider('slider-steps', 'val-steps', 'stepsPerGen');
bindSlider('slider-mutation', 'val-mutation', 'mutationRate', v => `${v.toFixed(1)}%`);

// Canvas Mouse Interaction (Inspect Dot or Draw/Erase Wall)
let isPointerDown = false;

function getGridCoordsFromEvent(e) {
  const rect = arenaCanvas.getBoundingClientRect();
  const normX = (e.clientX - rect.left) / rect.width;
  const normY = (e.clientY - rect.top) / rect.height;
  const gx = clamp(Math.floor(normX * GRID_SIZE), 0, GRID_SIZE - 1);
  const gy = clamp(Math.floor(normY * GRID_SIZE), 0, GRID_SIZE - 1);
  return { gx, gy };
}

function applyCanvasTool(e) {
  const { gx, gy } = getGridCoordsFromEvent(e);

  if (config.toolMode === 'inspect') {
    // Find nearest dot within a small radius
    let bestDist = Infinity;
    let bestDot = null;
    for (let i = 0; i < creatures.length; i++) {
      const c = creatures[i];
      const dx = c.x - gx;
      const dy = c.y - gy;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestDist && d2 <= 64) {
        bestDist = d2;
        bestDot = c;
      }
    }
    if (bestDot) {
      selectedCreature = bestDot;
      updateInspectorUI();
      drawBrainInspector();
    }
  } else if (config.toolMode === 'wall' || config.toolMode === 'erase') {
    const val = config.toolMode === 'wall' ? 1 : 0;
    const brushRadius = 1;
    for (let dy = -brushRadius; dy <= brushRadius; dy++) {
      for (let dx = -brushRadius; dx <= brushRadius; dx++) {
        const nx = gx + dx;
        const ny = gy + dy;
        if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
          const idx = ny * GRID_SIZE + nx;
          if (val === 0 || occGrid[idx] === 0) {
            wallsGrid[idx] = val;
          }
        }
      }
    }
  }
}

arenaCanvas.addEventListener('mousedown', (e) => {
  isPointerDown = true;
  applyCanvasTool(e);
});

window.addEventListener('mousemove', (e) => {
  if (isPointerDown && config.toolMode !== 'inspect') {
    applyCanvasTool(e);
  }
});

window.addEventListener('mouseup', () => {
  isPointerDown = false;
});

// --- Boot Simulation ---
buildBarrierPreset(config.barrierPreset);
initSimulation();
requestAnimationFrame(animate);

