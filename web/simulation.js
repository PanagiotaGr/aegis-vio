const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');

const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const blurSlider = document.getElementById('blurSlider');
const lightSlider = document.getElementById('lightSlider');
const imuSlider = document.getElementById('imuSlider');
const modeText = document.getElementById('modeText');
const modeCard = document.getElementById('modeCard');
const riskValue = document.getElementById('riskValue');
const qualityValue = document.getElementById('qualityValue');
const riskBar = document.getElementById('riskBar');
const qualityBar = document.getElementById('qualityBar');

const modeColors = {
  NORMAL: '#4ade80',
  CAUTIOUS: '#facc15',
  RECOVERY: '#fb923c',
  HALT: '#ef4444',
};

let running = false;
let frameId = null;
let t = 0;
let covariance = 0.04;
let risk = 0;
let quality = 1;
let robot = { x: 90, y: 280 };
let estimate = { x: 90, y: 280 };
let truthTrail = [];
let estimateTrail = [];

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
  const x = 90 + step * 1.35;
  const y = 280 + Math.sin(step * 0.035) * 96 + Math.sin(step * 0.012) * 30;
  return { x, y };
}

function reset() {
  running = false;
  if (frameId) cancelAnimationFrame(frameId);
  frameId = null;
  t = 0;
  covariance = 0.04;
  risk = 0;
  quality = 1;
  robot = { x: 90, y: 280 };
  estimate = { x: 90, y: 280 };
  truthTrail = [robot];
  estimateTrail = [estimate];
  updateUi('NORMAL');
  draw();
}

function updateUi(mode) {
  modeText.textContent = mode;
  modeCard.style.borderColor = modeColors[mode];
  modeCard.style.background = `linear-gradient(135deg, ${modeColors[mode]}33, rgba(105, 230, 255, 0.08))`;
  riskValue.textContent = risk.toFixed(2);
  qualityValue.textContent = quality.toFixed(2);
  riskBar.style.width = `${risk * 100}%`;
  riskBar.style.background = modeColors[mode];
  qualityBar.style.width = `${quality * 100}%`;
  qualityBar.style.background = quality > 0.6 ? '#4ade80' : quality > 0.3 ? '#facc15' : '#ef4444';
}

function step() {
  const blur = Number(blurSlider.value);
  const lowLight = Number(lightSlider.value);
  const imuNoise = Number(imuSlider.value);

  const [mode, speedScale] = modeFromRisk(risk);
  const speed = 1.6 * speedScale;
  t += speed;

  robot = referencePoint(t);

  const disturbanceWindow = 0.5 + 0.5 * Math.sin(t * 0.03);
  quality = clamp(1 - 0.42 * blur - 0.48 * lowLight - 0.20 * disturbanceWindow, 0.05, 1);

  covariance += 0.004 + imuNoise * 0.025 + (1 - quality) * 0.05;
  if (quality > 0.72) covariance *= 0.965;
  covariance = clamp(covariance, 0.02, 4.2);

  risk = clamp((covariance / Math.max(quality, 0.05)) / 4.2, 0, 1);
  const [newMode] = modeFromRisk(risk);

  const driftScale = 2.5 + covariance * 3.5;
  estimate = {
    x: robot.x + Math.sin(t * 0.07) * driftScale + imuNoise * 14,
    y: robot.y + Math.cos(t * 0.05) * driftScale - lowLight * 16,
  };

  truthTrail.push({ ...robot });
  estimateTrail.push({ ...estimate });
  if (truthTrail.length > 420) truthTrail.shift();
  if (estimateTrail.length > 420) estimateTrail.shift();

  updateUi(newMode);
  draw();

  if (robot.x > canvas.width - 70) {
    t = 0;
    truthTrail = [];
    estimateTrail = [];
  }
}

function drawGrid() {
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

function drawTrail(points, color, width) {
  if (points.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.stroke();
}

function drawRobot() {
  const [mode] = modeFromRisk(risk);
  const radius = 12 + risk * 26;

  ctx.beginPath();
  ctx.arc(estimate.x, estimate.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = `${modeColors[mode]}22`;
  ctx.fill();
  ctx.strokeStyle = `${modeColors[mode]}aa`;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(robot.x, robot.y, 10, 0, Math.PI * 2);
  ctx.fillStyle = '#69e6ff';
  ctx.fill();

  ctx.beginPath();
  ctx.arc(estimate.x, estimate.y, 8, 0, Math.PI * 2);
  ctx.fillStyle = modeColors[mode];
  ctx.fill();
}

function drawLegend() {
  ctx.fillStyle = 'rgba(238,245,255,0.9)';
  ctx.font = '14px system-ui, sans-serif';
  ctx.fillText('cyan: ground truth', 22, 32);
  ctx.fillText('mode color: estimated pose + uncertainty radius', 22, 54);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
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
