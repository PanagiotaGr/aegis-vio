const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');

const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const blurSlider = document.getElementById('blurSlider');
const lightSlider = document.getElementById('lightSlider');
const imuSlider = document.getElementById('imuSlider');
const modeText = document.getElementById('modeText');
const modeHint = document.getElementById('modeHint');
const modeCard = document.getElementById('modeCard');
const riskValue = document.getElementById('riskValue');
const qualityValue = document.getElementById('qualityValue');
const speedValue = document.getElementById('speedValue');
const errorValue = document.getElementById('errorValue');
const covValue = document.getElementById('covValue');
const riskBar = document.getElementById('riskBar');
const qualityBar = document.getElementById('qualityBar');

const modeColors = {
  NORMAL: '#4ade80',
  CAUTIOUS: '#facc15',
  RECOVERY: '#fb923c',
  HALT: '#ef4444',
};

const modeHints = {
  NORMAL: 'Full speed. Localization is trusted.',
  CAUTIOUS: 'Reduced speed. Safety margin is wider.',
  RECOVERY: 'Slow recovery. The robot searches for safer perception.',
  HALT: 'Robot stopped. Risk is too high.',
};

const obstacles = [
  { x: 260, y: 135, w: 90, h: 120 },
  { x: 515, y: 360, w: 120, h: 85 },
  { x: 730, y: 165, w: 95, h: 150 },
];
const goal = { x: 900, y: 505 };

let running = false;
let frameId = null;
let t = 0;
let covariance = 0.04;
let risk = 0;
let quality = 1;
let speedScale = 1;
let robot = { x: 80, y: 500 };
let estimate = { x: 80, y: 500 };
let truthTrail = [];
let estimateTrail = [];
let lastError = 0;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function modeFromRisk(value) {
  if (value >= 0.9) return ['HALT', 0.0, 2.5];
  if (value >= 0.65) return ['RECOVERY', 0.25, 2.0];
  if (value >= 0.35) return ['CAUTIOUS', 0.6, 1.5];
  return ['NORMAL', 1.0, 1.0];
}

function referencePoint(step) {
  const phase = step / 620;
  const x = 80 + phase * 820;
  const y = 500 - phase * 345 + Math.sin(step * 0.026) * 85 + Math.sin(step * 0.008) * 30;
  return { x, y: clamp(y, 80, 530) };
}

function reset() {
  running = false;
  if (frameId) cancelAnimationFrame(frameId);
  frameId = null;
  t = 0;
  covariance = 0.04;
  risk = 0;
  quality = 1;
  speedScale = 1;
  robot = { x: 80, y: 500 };
  estimate = { x: 80, y: 500 };
  truthTrail = [robot];
  estimateTrail = [estimate];
  lastError = 0;
  updateUi('NORMAL');
  draw();
}

function updateUi(mode) {
  modeText.textContent = mode;
  modeHint.textContent = modeHints[mode];
  modeCard.style.borderColor = modeColors[mode];
  modeCard.style.background = `linear-gradient(135deg, ${modeColors[mode]}33, rgba(105, 230, 255, 0.08))`;
  riskValue.textContent = risk.toFixed(2);
  qualityValue.textContent = quality.toFixed(2);
  speedValue.textContent = `${speedScale.toFixed(2)}x`;
  errorValue.textContent = `${(lastError / 18).toFixed(1)} m`;
  covValue.textContent = covariance.toFixed(2);
  riskBar.style.width = `${risk * 100}%`;
  riskBar.style.background = modeColors[mode];
  qualityBar.style.width = `${quality * 100}%`;
  qualityBar.style.background = quality > 0.6 ? '#4ade80' : quality > 0.3 ? '#facc15' : '#ef4444';
}

function step() {
  const blur = Number(blurSlider.value);
  const lowLight = Number(lightSlider.value);
  const imuNoise = Number(imuSlider.value);
  const [, nextSpeedScale] = modeFromRisk(risk);
  speedScale = nextSpeedScale;
  t += 2.2 * speedScale;

  robot = referencePoint(t);
  const obstaclePenalty = obstacles.some(o => robot.x > o.x - 55 && robot.x < o.x + o.w + 55 && robot.y > o.y - 55 && robot.y < o.y + o.h + 55) ? 0.12 : 0;
  const lightingWave = 0.5 + 0.5 * Math.sin(t * 0.028);
  quality = clamp(1 - 0.38 * blur - 0.44 * lowLight - 0.14 * lightingWave - obstaclePenalty, 0.05, 1);

  covariance += 0.0035 + imuNoise * 0.026 + (1 - quality) * 0.045;
  if (quality > 0.74) covariance *= 0.962;
  covariance = clamp(covariance, 0.02, 4.4);

  risk = clamp((covariance / Math.max(quality, 0.05)) / 4.3, 0, 1);
  const [newMode] = modeFromRisk(risk);

  const driftScale = 3 + covariance * 4.2;
  estimate = {
    x: robot.x + Math.sin(t * 0.065) * driftScale + imuNoise * 16,
    y: robot.y + Math.cos(t * 0.052) * driftScale - lowLight * 18,
  };
  lastError = Math.hypot(estimate.x - robot.x, estimate.y - robot.y);

  truthTrail.push({ ...robot });
  estimateTrail.push({ ...estimate });
  if (truthTrail.length > 460) truthTrail.shift();
  if (estimateTrail.length > 460) estimateTrail.shift();

  updateUi(newMode);
  draw();

  if (robot.x > goal.x - 12 || t > 640) {
    t = 0;
    covariance *= 0.55;
    truthTrail = [];
    estimateTrail = [];
  }
}

function drawGrid() {
  const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, '#07101d');
  gradient.addColorStop(1, '#0b1a2c');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = 'rgba(255,255,255,0.055)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= canvas.width; x += 45) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= canvas.height; y += 45) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
}

function drawObstacles() {
  for (const obstacle of obstacles) {
    ctx.fillStyle = 'rgba(239, 68, 68, 0.18)';
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.65)';
    ctx.lineWidth = 2;
    roundRect(obstacle.x, obstacle.y, obstacle.w, obstacle.h, 14);
    ctx.fill();
    ctx.stroke();
  }
}

function drawGoal() {
  ctx.beginPath();
  ctx.arc(goal.x, goal.y, 20, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(74, 222, 128, 0.20)';
  ctx.fill();
  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.fillStyle = '#4ade80';
  ctx.font = '700 13px system-ui, sans-serif';
  ctx.fillText('GOAL', goal.x - 18, goal.y - 28);
}

function drawTrail(points, color, width) {
  if (points.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.stroke();
}

function drawLidar(cx, cy, mode) {
  ctx.strokeStyle = `${modeColors[mode]}55`;
  ctx.lineWidth = 1;
  for (let i = 0; i < 18; i++) {
    const a = (i / 18) * Math.PI * 2;
    const length = 48 + quality * 58;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a) * length, cy + Math.sin(a) * length);
    ctx.stroke();
  }
}

function drawRobot() {
  const [mode] = modeFromRisk(risk);
  const radius = 16 + risk * 42;

  ctx.beginPath();
  ctx.arc(estimate.x, estimate.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = `${modeColors[mode]}22`;
  ctx.fill();
  ctx.strokeStyle = `${modeColors[mode]}aa`;
  ctx.lineWidth = 2;
  ctx.stroke();

  drawLidar(robot.x, robot.y, mode);

  ctx.save();
  ctx.translate(robot.x, robot.y);
  ctx.rotate(Math.sin(t * 0.035) * 0.35);
  ctx.fillStyle = '#69e6ff';
  ctx.strokeStyle = '#dffbff';
  ctx.lineWidth = 2;
  roundRect(-17, -13, 34, 26, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#06101d';
  ctx.fillRect(2, -5, 10, 10);
  ctx.restore();

  ctx.beginPath();
  ctx.arc(estimate.x, estimate.y, 8, 0, Math.PI * 2);
  ctx.fillStyle = modeColors[mode];
  ctx.fill();
}

function drawLegend() {
  ctx.fillStyle = 'rgba(238,245,255,0.92)';
  ctx.font = '14px system-ui, sans-serif';
  ctx.fillText('cyan robot: ground truth pose', 22, 32);
  ctx.fillText('white path: estimated trajectory', 22, 54);
  ctx.fillText('colored bubble: uncertainty / safety margin', 22, 76);
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  drawGoal();
  drawObstacles();
  drawTrail(truthTrail, '#69e6ff', 3);
  drawTrail(estimateTrail, '#f8fafc', 2);
  drawRobot();
  drawLegend();
}

function loop() {
  if (!running) return;
  step();
  frameId = requestAnimationFrame(loop);
}

startBtn.addEventListener('click', () => {
  if (!running) {
    running = true;
    loop();
  }
});

pauseBtn.addEventListener('click', () => {
  running = false;
});

resetBtn.addEventListener('click', reset);

reset();
