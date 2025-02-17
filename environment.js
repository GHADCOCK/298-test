// Canvas setup and context
const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");

// Add control state
let isPaused = false;
let speedMultiplier = 1;

// Add control elements
const speedSlider = document.getElementById("speedSlider");
const speedValue = document.getElementById("speedValue");
const pauseButton = document.getElementById("pauseButton");

// Add event listeners
speedSlider.addEventListener("input", (e) => {
  speedMultiplier = parseFloat(e.target.value);
  speedValue.textContent = `${speedMultiplier}x`;
});

pauseButton.addEventListener("click", () => {
  isPaused = !isPaused;
  pauseButton.textContent = isPaused ? "Resume" : "Pause";
  if (!isPaused) {
    lastMoveTime = performance.now() - MOVE_INTERVAL; // Immediate move on resume
    requestAnimationFrame(animate);
  }
});

/*
 * Vehicle State Management
 * Each vehicle tracks its:
 * - Current position [row, col]
 * - Target position for next move
 * - Intermediate position for wrap-around
 * - Movement progress (0 to 1)
 */
// Vehicle position state
let carPosition = [0, 0];
let carTargetPosition = [0, 0];
let carIntermediatePosition = null;
let carMoveProgress = 1;

// Add bus state (starting at opposite corner)
let busPosition = [2, 2];
let busTargetPosition = [2, 2];
let busIntermediatePosition = null;
let busMoveProgress = 1;

/*
 * Animation timing constants
 * MOVE_DURATION: Time for one movement animation (ms)
 * MOVE_INTERVAL: Time between movements (ms)
 */
// Timing constants
const MOVE_DURATION = 400;
let lastMoveTime = 0;
const MOVE_INTERVAL = 1000;

/**
 * Draws a single grid square
 * @param {number} x - X coordinate in pixels
 * @param {number} y - Y coordinate in pixels
 * @param {boolean} isLight - Whether to use light or dark color
 */
function drawSquare(x, y, isLight) {
  ctx.fillStyle = isLight ? "white" : "gray";
  ctx.strokeStyle = "black";
  ctx.lineWidth = 8;
  ctx.fillRect(x, y, 200, 200);
  ctx.strokeRect(x, y, 200, 200);
}

/**
 * Creates the 3x3 checkerboard grid pattern
 */
function createGrid() {
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const isLight = (row + col) % 2 === 0;
      drawSquare(col * 200, row * 200, isLight);
    }
  }
}

/**
 * Draws the red car at specified coordinates
 * @param {number} x - X coordinate in pixels
 * @param {number} y - Y coordinate in pixels
 */
function drawCar(x, y) {
  // Car body (rectangle)
  ctx.fillStyle = "red";
  ctx.fillRect(x + 40, y + 60, 120, 50);

  // Car roof
  ctx.fillRect(x + 80, y + 30, 60, 30);

  // Wheels
  ctx.beginPath();
  ctx.fillStyle = "black";
  // Left wheel
  ctx.arc(x + 70, y + 110, 20, 0, Math.PI * 2);
  // Right wheel
  ctx.arc(x + 130, y + 110, 20, 0, Math.PI * 2);
  ctx.fill();
}

/**
 * Draws the yellow bus at specified coordinates
 * @param {number} x - X coordinate in pixels
 * @param {number} y - Y coordinate in pixels
 */
function drawBus(x, y) {
  // Bus body (longer than car)
  ctx.fillStyle = "yellow"; // Changed from blue to yellow
  ctx.fillRect(x + 20, y + 60, 160, 50);

  // Bus roof
  ctx.fillRect(x + 20, y + 30, 160, 30);

  // Windows (multiple)
  ctx.fillStyle = "lightyellow"; // Changed to match yellow theme
  for (let i = 0; i < 3; i++) {
    ctx.fillRect(x + 35 + i * 50, y + 35, 30, 20);
  }

  // Wheels
  ctx.beginPath();
  ctx.fillStyle = "black";
  // Front wheels
  ctx.arc(x + 50, y + 110, 15, 0, Math.PI * 2);
  ctx.arc(x + 90, y + 110, 15, 0, Math.PI * 2);
  // Rear wheels
  ctx.arc(x + 110, y + 110, 15, 0, Math.PI * 2);
  ctx.arc(x + 150, y + 110, 15, 0, Math.PI * 2);
  ctx.fill();
}

/**
 * Converts grid position to pixel coordinates
 * @param {Array} gridPos - [row, col] position in grid
 * @returns {Array} [x, y] pixel coordinates
 */
function getPixelPosition(gridPos) {
  return [gridPos[1] * 200, gridPos[0] * 200];
}

/**
 * Linear interpolation between two values
 * @param {number} start - Starting value
 * @param {number} end - Ending value
 * @param {number} t - Progress (0 to 1)
 */
function lerp(start, end, t) {
  return start + (end - start) * t;
}

/**
 * Determines if movement requires wrap-around
 * @param {number} start - Starting position
 * @param {number} end - Ending position
 * @returns {boolean} True if wrap-around is needed
 */
function needsWrapAround(start, end) {
  return Math.abs(start - end) > 1;
}

/**
 * Calculates intermediate position for wrap-around movement
 * @param {Array} current - Current [row, col] position
 * @param {Array} target - Target [row, col] position
 * @returns {Array|null} Intermediate position or null if no wrap needed
 */
function getWrapAroundPath(current, target) {
  // Calculate differences
  const rowDiff = target[0] - current[0];
  const colDiff = target[1] - current[1];

  // If wrapping is needed, create intermediate position
  if (needsWrapAround(current[0], target[0])) {
    // Wrapping vertically
    const intermediateRow = rowDiff > 0 ? -0.5 : 3.5;
    return [current[1], intermediateRow];
  } else if (needsWrapAround(current[1], target[1])) {
    // Wrapping horizontally
    const intermediateCol = colDiff > 0 ? -0.5 : 3.5;
    return [intermediateCol, current[0]];
  }
  return null;
}

/**
 * Generates random movement for a vehicle
 * @param {Array} currentPos - Current position
 * @param {Array} targetPos - Current target position
 * @param {Array|null} intermediatePos - Current intermediate position
 * @param {number} moveProgress - Current movement progress
 * @returns {Object|null} New movement data or null if movement in progress
 */
function moveVehicleRandomly(
  currentPos,
  targetPos,
  intermediatePos,
  moveProgress
) {
  if (moveProgress < 1) return null;

  const possibleMoves = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];

  const move = possibleMoves[Math.floor(Math.random() * possibleMoves.length)];

  const newTargetPos = [
    (currentPos[0] + move[0] + 3) % 3,
    (currentPos[1] + move[1] + 3) % 3,
  ];

  return {
    targetPos: newTargetPos,
    intermediatePos: getWrapAroundPath(currentPos, newTargetPos),
  };
}

/**
 * Main animation loop
 * Handles:
 * 1. Grid drawing
 * 2. Vehicle movement timing
 * 3. Position updates
 * 4. Vehicle drawing
 * @param {number} currentTime - Current timestamp
 */
function animate(currentTime) {
  if (isPaused) return;

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  createGrid();

  if (currentTime - lastMoveTime >= MOVE_INTERVAL / speedMultiplier) {
    const carMove = moveVehicleRandomly(
      carPosition,
      carTargetPosition,
      carIntermediatePosition,
      carMoveProgress
    );
    const busMove = moveVehicleRandomly(
      busPosition,
      busTargetPosition,
      busIntermediatePosition,
      busMoveProgress
    );

    if (carMove) {
      carTargetPosition = carMove.targetPos;
      carIntermediatePosition = carMove.intermediatePos;
      carMoveProgress = 0;
    }

    if (busMove) {
      busTargetPosition = busMove.targetPos;
      busIntermediatePosition = busMove.intermediatePos;
      busMoveProgress = 0;
    }

    lastMoveTime = currentTime;
  }

  if (carMoveProgress < 1) {
    carMoveProgress = Math.min(
      1,
      carMoveProgress + (16 / MOVE_DURATION) * speedMultiplier
    );
  }
  if (busMoveProgress < 1) {
    busMoveProgress = Math.min(
      1,
      busMoveProgress + (16 / MOVE_DURATION) * speedMultiplier
    );
  }

  let carPos = calculatePosition(
    carPosition,
    carTargetPosition,
    carIntermediatePosition,
    carMoveProgress
  );
  drawCar(carPos.x * 200, carPos.y * 200);

  let busPos = calculatePosition(
    busPosition,
    busTargetPosition,
    busIntermediatePosition,
    busMoveProgress
  );
  drawBus(busPos.x * 200, busPos.y * 200);

  if (carMoveProgress === 1) {
    carPosition = [...carTargetPosition];
    carIntermediatePosition = null;
  }
  if (busMoveProgress === 1) {
    busPosition = [...busTargetPosition];
    busIntermediatePosition = null;
  }

  requestAnimationFrame(animate);
}

/**
 * Calculates current position during animation
 * Handles three phases of movement:
 * 1. Moving to edge (0-40% of animation)
 * 2. Teleporting (40-60% of animation)
 * 3. Moving from edge to target (60-100% of animation)
 */
function calculatePosition(currentPos, targetPos, intermediatePos, progress) {
  if (!intermediatePos) {
    return {
      x: lerp(currentPos[1], targetPos[1], progress),
      y: lerp(currentPos[0], targetPos[0], progress),
    };
  }

  if (progress < 0.4) {
    const t = progress * 2.5;
    return {
      x: lerp(currentPos[1], intermediatePos[0], t),
      y: lerp(currentPos[0], intermediatePos[1], t),
    };
  } else if (progress < 0.6) {
    if (intermediatePos[0] < 0 || intermediatePos[0] > 2) {
      return {
        x: intermediatePos[0] < 0 ? 3 : -1,
        y: targetPos[0],
      };
    } else {
      return {
        x: targetPos[1],
        y: intermediatePos[1] < 0 ? 3 : -1,
      };
    }
  } else {
    const t = (progress - 0.6) * 2.5;
    if (intermediatePos[0] < 0 || intermediatePos[0] > 2) {
      return {
        x: lerp(intermediatePos[0] < 0 ? 3 : -1, targetPos[1], t),
        y: targetPos[0],
      };
    } else {
      return {
        x: targetPos[1],
        y: lerp(intermediatePos[1] < 0 ? 3 : -1, targetPos[0], t),
      };
    }
  }
}

/**
 * Initial canvas setup and animation start
 */
function setupCanvas() {
  canvas.width = 600;
  canvas.height = 600;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  createGrid();

  // Draw both vehicles in initial positions
  const [carPixelX, carPixelY] = getPixelPosition(carPosition);
  const [busPixelX, busPixelY] = getPixelPosition(busPosition);
  drawCar(carPixelX, carPixelY);
  drawBus(busPixelX, busPixelY);

  requestAnimationFrame(animate);
}

// Start the animation
setupCanvas();
