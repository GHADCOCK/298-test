// Canvas setup and context
const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");

// Add control state
let _isPaused = false;
// Define a getter and setter for isPaused, so it does something whenever isPaused gets updated automatically
Object.defineProperty(window, "isPaused", {
  get: function () {
    return _isPaused;
  },
  set: function (value) {
    _isPaused = value;
    // Update button text automatically whenever isPaused changes
    pauseButton.textContent = _isPaused ? "Resume" : "Pause";

    // Additional logic when value changes
    if (!_isPaused) {
      lastMoveTime = performance.now() - move_interval; // Immediate move on resume
      requestAnimationFrame(animate);
    }
  },
});

let speedMultiplier = 1;

// Position probability data from text file
let positionProbabilities = {};

// Add control elements
const speedSlider = document.getElementById("speedSlider");
const speedValue = document.getElementById("speedValue");
const pauseButton = document.getElementById("pauseButton");

// Query elements
const carRowInput = document.getElementById("carRowInput");
const carColInput = document.getElementById("carColInput");
const busRowInput = document.getElementById("busRowInput");
const busColInput = document.getElementById("busColInput");
const queryButton = document.getElementById("queryButton");
const queryResult = document.getElementById("queryResult");

// Define direction labels and corresponding vectors
const directionLabels = ["Up", "Down", "Left", "Right"];
const directionVectors = [
  [0, -1], // Up
  [0, 1], // Down
  [-1, 0], // Left
  [1, 0], // Right
];
let NUM_ROWS = 1;
let NUM_COLS = 1;
let SCALE_FACTOR = 1;

// Load the probability data from text file
async function loadProbabilityData() {
  try {
    const response = await fetch("sharedNetworkProbabilitiesStd.txt"); //nash_equilibrium.txt
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const text = await response.text();
    const lines = text.split("\n");
    // Get the number of rows and columns from the nash file
    let maxRow = 0;
    let maxCol = 0;
    lines.forEach((line) => {
      const trimmedLine = line.trim();
      if (trimmedLine && !trimmedLine.startsWith("#")) {
        const [position] = trimmedLine.split(":");
        if (position) {
          const [row, col] = position.split(",").map((n) => parseInt(n));
          if (isNaN(row) || isNaN(col)) {
            // alert("Invalid row or column: " + row + " " + col);
            return;
          }
          maxRow = Math.max(maxRow, row);
          maxCol = Math.max(maxCol, col);
        }
      }
    });
    // alert(maxRow + " " + maxCol);
    NUM_ROWS = maxRow + 1;
    NUM_COLS = maxCol + 1;
    // Parse the text file content
    // New format: X_1,Y_1,X_2,Y_2:U1,D1,L1,R1,U2,D2,L2,R2
    lines.forEach((line) => {
      const trimmedLine = line.trim();
      if (trimmedLine && !trimmedLine.startsWith("#")) {
        // Skip empty lines and comments
        const [position, probabilities] = trimmedLine.split(":");
        if (position && probabilities) {
          // Store as an object with car and bus probabilities
          const probArray = probabilities
            .split(",")
            .map((p) => parseFloat(p.trim()));

          if (probArray.length === 8) {
            // Ensure correct format
            positionProbabilities[position.trim()] = {
              bus: probArray.slice(0, 4), // U1, D1, L1, R1
              car: probArray.slice(4), // U2, D2, L2, R2
            };
          }
        }
      }
    });
    // setupCanvas();

    console.log("Probability data loaded:", positionProbabilities);
    return true;
  } catch (error) {
    console.error("Error loading probability data:", error);
    queryResult.textContent = "Error loading probability data";
  }
}

// Query position function
function queryPosition(pause = true) {
  const carRow = parseInt(carRowInput.value);
  const carCol = parseInt(carColInput.value);
  const busRow = parseInt(busRowInput.value);
  const busCol = parseInt(busColInput.value);
  if (pause) {
    isPaused = true;
  }
  // Validate input
  if (
    isNaN(carRow) ||
    isNaN(carCol) ||
    isNaN(busRow) ||
    isNaN(busCol) ||
    carRow < 0 ||
    carRow > NUM_ROWS - 1 ||
    carCol < 0 ||
    carCol > NUM_COLS - 1 ||
    busRow < 0 ||
    busRow > NUM_ROWS - 1 ||
    busCol < 0 ||
    busCol > NUM_COLS - 1
  ) {
    queryResult.textContent = `Invalid positions. All row and column values must be between 0 and ${
      NUM_ROWS - 1
    }.`;
    return;
  }

  // Update vehicle positions to teleport them to queried locations
  carPosition = [carRow, carCol];
  carTargetPosition = [carRow, carCol];
  carMoveProgress = 1; // Ensure movement is complete
  carIntermediatePosition = null;

  busPosition = [busRow, busCol];
  busTargetPosition = [busRow, busCol];
  busMoveProgress = 1; // Ensure movement is complete
  busIntermediatePosition = null;

  // Clear the canvas completely before redrawing
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Redraw the scene with updated positions
  createGrid();
  drawCar(carCol * 200 * SCALE_FACTOR, carRow * 200 * SCALE_FACTOR);
  drawBus(busCol * 200 * SCALE_FACTOR, busRow * 200 * SCALE_FACTOR);

  // Get probability for this state
  const key = `${busRow},${busCol},${carRow},${carCol}`; // Position Order: Bus, Car
  const data = positionProbabilities[key];

  if (data) {
    // Create a formatted result display
    let resultHtml = `<strong>Position State:</strong> Car at (${carRow},${carCol}), Bus at (${busRow},${busCol})<br><br>`;

    // Car probabilities
    resultHtml += "<strong>Car Movement Probabilities:</strong><br>";
    data.car.forEach((prob, index) => {
      resultHtml += `${directionLabels[index]}: ${(prob * 100).toFixed(
        1
      )}%<br>`;
    });

    resultHtml += "<br><strong>Bus Movement Probabilities:</strong><br>";
    data.bus.forEach((prob, index) => {
      resultHtml += `${directionLabels[index]}: ${(prob * 100).toFixed(
        1
      )}%<br>`;
    });

    queryResult.innerHTML = resultHtml;

    // Highlight and draw probability arrows for both positions
    // highlightPositions(carRow, carCol, busRow, busCol);
    drawProbabilityArrows(carRow, carCol, data.car, "car");
    drawProbabilityArrows(busRow, busCol, data.bus, "bus");
  } else {
    queryResult.textContent = `No data found for the specified position state`;
  }
}

// Highlight positions of both vehicles
function highlightPositions(carRow, carCol, busRow, busCol) {
  // Draw a highlighted overlay on the squares
  ctx.fillStyle = "rgba(255, 0, 0, 0.3)"; // Semi-transparent red for car
  ctx.fillRect(
    carCol * 200 * SCALE_FACTOR,
    carRow * 200 * SCALE_FACTOR,
    200 * SCALE_FACTOR,
    200 * SCALE_FACTOR
  );

  ctx.fillStyle = "rgba(255, 255, 0, 0.3)"; // Semi-transparent yellow for bus
  ctx.fillRect(
    busCol * 200 * SCALE_FACTOR,
    busRow * 200 * SCALE_FACTOR,
    200 * SCALE_FACTOR,
    200 * SCALE_FACTOR
  );
}

/**
 * Draws probability arrows starting from a cell
 * @param {number} row - Grid row
 * @param {number} col - Grid column
 * @param {Array} probabilities - Array of movement probabilities [U,D,L,R]
 * @param {string} vehicleType - "car" or "bus"
 */
function drawProbabilityArrows(row, col, probabilities, vehicleType) {
  const cellCenterX = col * 200 * SCALE_FACTOR + 100 * SCALE_FACTOR;
  const cellCenterY = row * 200 * SCALE_FACTOR + 100 * SCALE_FACTOR;
  const maxArrowLength = 80 * SCALE_FACTOR; // Maximum length in pixels

  // Set arrow color based on vehicle type
  const arrowColor =
    vehicleType === "car" ? "rgba(51, 255, 0, 0.8)" : "rgba(218, 165, 32, 0.8)";
  const arrowWidth = vehicleType === "car" ? 6 : 4; // Car arrows slightly thicker
  let maxProbs = 0;
  probabilities.forEach((probability, index) => {
    maxProbs = Math.max(maxProbs, probability);
  });

  // Draw an arrow for each direction
  probabilities.forEach((probability, index) => {
    // Calculate arrow length based on probability
    // const arrowLength = probability * maxArrowLength;
    const arrowLength = (probability / maxProbs) * maxArrowLength;

    // Get direction vector
    const dirVector = directionVectors[index];

    // Calculate end point
    const endX = cellCenterX + dirVector[0] * arrowLength;
    const endY = cellCenterY + dirVector[1] * arrowLength;

    // Draw arrow
    drawArrow(cellCenterX, cellCenterY, endX, endY, arrowColor, arrowWidth);
    // Add probability text
    if (probability > 0.01) {
      // Only draw text if probability is significant
      // Position text near the arrow but not on top of it
      const textOffsetFactor = 1.1; // Adjust this for text placement
      const isLeftRight = dirVector[0] === -1 || dirVector[0] === 1;
      const textX = cellCenterX + dirVector[0] * arrowLength * textOffsetFactor;
      let textY = cellCenterY + dirVector[1] * arrowLength * textOffsetFactor;
      if (isLeftRight) {
        textY -= 10; // Shift text down for left/right arrows
      }

      // Format probability as percentage
      const probText = `${(probability * 100).toFixed(0)}%`;

      // Set text style
      ctx.font = "bold 16px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      // Add white stroke for better visibility
      ctx.strokeStyle = "white";
      ctx.lineWidth = 3;
      ctx.strokeText(probText, textX, textY);

      // Draw the text in black
      ctx.fillStyle = "black";
      ctx.fillText(probText, textX, textY);
    }
  });
}

/**
 * Draws an arrow from (x1,y1) to (x2,y2)
 * @param {number} x1 - Start x coordinate
 * @param {number} y1 - Start y coordinate
 * @param {number} x2 - End x coordinate
 * @param {number} y2 - End y coordinate
 * @param {string} color - Arrow color
 * @param {number} lineWidth - Arrow line width
 */
function drawArrow(x1, y1, x2, y2, color, lineWidth = 2) {
  // If arrow is too short, don't draw it
  if (Math.abs(x2 - x1) < 5 && Math.abs(y2 - y1) < 5) return;

  const headSize = 10; // Arrow head size
  const angle = Math.atan2(y2 - y1, x2 - x1);

  // Calculate where the rectangle should end (before the arrow head)
  const shortenedLength =
    Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2)) - headSize;
  const endX = x1 + Math.cos(angle) * shortenedLength;
  const endY = y1 + Math.sin(angle) * shortenedLength;

  // Save context
  ctx.save();

  // Set arrow style
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;

  // Draw the line (rectangle portion)
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  // Draw the arrow head
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - headSize * Math.cos(angle - Math.PI / 6),
    y2 - headSize * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    x2 - headSize * Math.cos(angle + Math.PI / 6),
    y2 - headSize * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();

  // Restore context
  ctx.restore();
}

// Add event listeners
speedSlider.addEventListener("input", (e) => {
  speedMultiplier = parseFloat(e.target.value);
  speedValue.textContent = `${speedMultiplier}x`;
});

pauseButton.addEventListener("click", () => {
  isPaused = !isPaused;
  // pauseButton.textContent = isPaused ? "Resume" : "Pause";
  if (!isPaused) {
    lastMoveTime = performance.now() - move_interval; // Immediate move on resume
    requestAnimationFrame(animate);
  }
});

// Add query button event listener
queryButton.addEventListener("click", queryPosition);

// Add "Use Current Positions" functionality
const useCurrPosButton = document.createElement("button");
useCurrPosButton.textContent = "Use Current Positions";
useCurrPosButton.style.marginLeft = "10px";
useCurrPosButton.addEventListener("click", () => {
  carRowInput.value = carPosition[0];
  carColInput.value = carPosition[1];
  busRowInput.value = busPosition[0];
  busColInput.value = busPosition[1];
  isPaused = true;
  queryPosition();
});
queryButton.parentNode.insertBefore(useCurrPosButton, queryButton.nextSibling);

// Add "Clear Arrows" functionality
const clearArrowsButton = document.createElement("button");
clearArrowsButton.textContent = "Clear Arrows";
clearArrowsButton.style.marginLeft = "10px";
clearArrowsButton.style.background = "#ff6347";
clearArrowsButton.addEventListener("click", () => {
  createGrid();

  // Redraw vehicles at their current positions
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
});
useCurrPosButton.parentNode.insertBefore(
  clearArrowsButton,
  useCurrPosButton.nextSibling
);

// Add reward matrix data
let tileRewards = {};
let minReward = 0;
let maxReward = 0;

// Add a checkbox to toggle reward visualization
const rewardToggleLabel = document.createElement("label");
rewardToggleLabel.style.marginLeft = "15px";
rewardToggleLabel.style.fontWeight = "bold";
const rewardToggle = document.createElement("input");
rewardToggle.type = "checkbox";
rewardToggle.id = "rewardToggle";
rewardToggle.checked = false;
rewardToggleLabel.appendChild(rewardToggle);
rewardToggleLabel.appendChild(document.createTextNode(" Show Rewards"));
pauseButton.parentNode.insertBefore(rewardToggleLabel, pauseButton.nextSibling);

rewardToggle.addEventListener("change", function () {
  createGrid(); // Redraw the grid with or without rewards

  // Redraw vehicles at their current positions
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
});

// Load the reward data from text file
async function loadRewardData() {
  try {
    const response = await fetch("TileRewardMatrix.txt");
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const text = await response.text();
    const lines = text.split("\n");

    let allRewards = [];

    // Parse the text file content
    lines.forEach((line) => {
      const trimmedLine = line.trim();
      if (trimmedLine && !trimmedLine.startsWith("#")) {
        // Skip empty lines and comments
        const [position, rewardStr] = trimmedLine.split(":");
        if (position && rewardStr) {
          const reward = parseFloat(rewardStr.trim());
          tileRewards[position.trim()] = reward;
          allRewards.push(reward);
        }
      }
    });

    // Calculate min and max reward for color scaling
    minReward = Math.min(...allRewards);
    maxReward = Math.max(...allRewards);

    console.log("Reward data loaded:", tileRewards);
    console.log(`Reward range: ${minReward} to ${maxReward}`);
  } catch (error) {
    console.error("Error loading reward data:", error);
  }
}

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

let diagramStep = document.getElementById("diagramStep");

/*
 * Animation timing constants
 * MOVE_DURATION: Time for one movement animation (ms)
 * MOVE_INTERVAL: Time between movements (ms)
 */
// Timing constants
const MOVE_DURATION = 400;
let lastMoveTime = 0;
const MOVE_INTERVAL_LABEL = document.getElementById("carWaitStep"); //1000;
let move_interval = MOVE_INTERVAL_LABEL.value * 1000;

MOVE_INTERVAL_LABEL.addEventListener("input", (e) => {
  move_interval = e.target.value * 1000;
});

/**
 * Draws a single grid square with optional reward-based coloring
 * @param {number} x - X coordinate in pixels
 * @param {number} y - Y coordinate in pixels
 * @param {boolean} isLight - Whether to use light or dark color
 * @param {number} row - Grid row
 * @param {number} col - Grid column
 */
function drawSquare(x, y, isLight, row, col) {
  // Default colors
  let fillColor = isLight ? "white" : "gray";

  // If reward visualization is enabled
  if (rewardToggle && rewardToggle.checked) {
    const key = `${row},${col}`;
    if (tileRewards[key] !== undefined) {
      // Normalize reward color based on min and max values
      if (tileRewards[key] < 0) {
        // Negative reward: white to red
        const intensity = Math.abs(tileRewards[key]) / Math.abs(minReward);
        fillColor = `rgb(255, ${255 - Math.round(intensity * 255)}, ${
          255 - Math.round(intensity * 255)
        })`;
      } else if (tileRewards[key] > 0) {
        // Positive reward: white to blue
        const intensity = tileRewards[key] / maxReward;
        fillColor = `rgb(${255 - Math.round(intensity * 255)}, ${
          255 - Math.round(intensity * 255)
        }, 255)`;
      } else {
        // Zero reward: white
        fillColor = "rgb(255, 255, 255)";
      }
    }
  }

  ctx.fillStyle = fillColor;
  ctx.strokeStyle = "black";
  ctx.lineWidth = 8;
  ctx.fillRect(x, y, 200 * SCALE_FACTOR, 200 * SCALE_FACTOR);
  ctx.strokeRect(x, y, 200 * SCALE_FACTOR, 200 * SCALE_FACTOR);

  // If reward visualization is enabled, add text showing the reward value
  if (rewardToggle && rewardToggle.checked) {
    const key = `${row},${col}`;
    if (tileRewards[key] !== undefined) {
      ctx.font = "bold 24px Arial";
      ctx.fillStyle = "black";
      ctx.textAlign = "center";
      ctx.fillText(
        `${tileRewards[key].toFixed(1)}`,
        x + 160 * SCALE_FACTOR,
        y + 180 * SCALE_FACTOR
      );
    }
  }
}

/**
 * Creates the 3x3 checkerboard grid pattern
 */
function createGrid() {
  for (let row = 0; row < NUM_ROWS; row++) {
    for (let col = 0; col < NUM_COLS; col++) {
      const isLight = (row + col) % 2 === 0;
      drawSquare(
        col * 200 * SCALE_FACTOR,
        row * 200 * SCALE_FACTOR,
        isLight,
        row,
        col
      );
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
  ctx.fillStyle = "green";
  ctx.fillRect(
    x + 40 * SCALE_FACTOR,
    y + 60 * SCALE_FACTOR,
    120 * SCALE_FACTOR,
    50 * SCALE_FACTOR
  );

  // Car roof
  ctx.fillRect(
    x + 80 * SCALE_FACTOR,
    y + 30 * SCALE_FACTOR,
    60 * SCALE_FACTOR,
    30 * SCALE_FACTOR
  );

  // Wheels
  ctx.beginPath();
  ctx.fillStyle = "black";
  // Left wheel
  ctx.arc(
    x + 70 * SCALE_FACTOR,
    y + 110 * SCALE_FACTOR,
    20 * SCALE_FACTOR,
    0,
    Math.PI * 2
  );
  // Right wheel
  ctx.arc(
    x + 130 * SCALE_FACTOR,
    y + 110 * SCALE_FACTOR,
    20 * SCALE_FACTOR,
    0,
    Math.PI * 2
  );
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
  ctx.fillRect(
    x + 20 * SCALE_FACTOR,
    y + 60 * SCALE_FACTOR,
    160 * SCALE_FACTOR,
    50 * SCALE_FACTOR
  );

  // Bus roof
  ctx.fillRect(
    x + 20 * SCALE_FACTOR,
    y + 30 * SCALE_FACTOR,
    160 * SCALE_FACTOR,
    30 * SCALE_FACTOR
  );

  // Windows (multiple)
  ctx.fillStyle = "lightyellow"; // Changed to match yellow theme
  for (let i = 0; i < 3; i++) {
    ctx.fillRect(
      x + 35 * SCALE_FACTOR + i * 50 * SCALE_FACTOR,
      y + 35 * SCALE_FACTOR,
      30 * SCALE_FACTOR,
      20 * SCALE_FACTOR
    );
  }

  // Wheels
  ctx.beginPath();
  ctx.fillStyle = "black";
  // Front wheels
  ctx.arc(
    x + 50 * SCALE_FACTOR,
    y + 110 * SCALE_FACTOR,
    15 * SCALE_FACTOR,
    0,
    Math.PI * 2
  );
  ctx.arc(
    x + 90 * SCALE_FACTOR,
    y + 110 * SCALE_FACTOR,
    15 * SCALE_FACTOR,
    0,
    Math.PI * 2
  );
  // Rear wheels
  ctx.arc(
    x + 110 * SCALE_FACTOR,
    y + 110 * SCALE_FACTOR,
    15 * SCALE_FACTOR,
    0,
    Math.PI * 2
  );
  ctx.arc(
    x + 150 * SCALE_FACTOR,
    y + 110 * SCALE_FACTOR,
    15 * SCALE_FACTOR,
    0,
    Math.PI * 2
  );
  ctx.fill();
}

/**
 * Converts grid position to pixel coordinates
 * @param {Array} gridPos - [row, col] position in grid
 * @returns {Array} [x, y] pixel coordinates
 */
function getPixelPosition(gridPos) {
  return [gridPos[1] * 200 * SCALE_FACTOR, gridPos[0] * 200 * SCALE_FACTOR];
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

function chooseBestMove(probabilities) {
  const move = Math.random();
  let cumulative = 0;

  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (move < cumulative) {
      return i;
    }
  }

  // Default to last move if probabilities don't sum to 1
  return probabilities.length - 1;
}

/**
 * Generates random movement for a vehicle
 * @param {Array} currentPos - Current position
 * @param {Array} targetPos - Current target position
 * @param {Array|null} intermediatePos - Current intermediate position
 * @param {number} moveProgress - Current movement progress
 * @param {Object} vehicleType - Vehicle object with probabilities
 * @returns {Object|null} New movement data or null if movement in progress
 */
function moveVehicleRandomly(
  currentPos,
  targetPos,
  intermediatePos,
  moveProgress,
  vehicleType
) {
  if (moveProgress < 1) return null;

  const possibleMoves = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];

  const moveInd = chooseBestMove(vehicleType); //possibleMoves[Math.floor(Math.random() * possibleMoves.length)];
  const move = possibleMoves[moveInd];

  const newTargetPos = [
    (currentPos[0] + move[0] + NUM_ROWS) % NUM_ROWS,
    (currentPos[1] + move[1] + NUM_COLS) % NUM_COLS,
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
  // animate function
  if (isPaused) return;

  // Store the current drawings to check for arrows
  const hasArrows = ctx.getImageData(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  createGrid();

  if (currentTime - lastMoveTime >= move_interval / speedMultiplier) {
    // Get current state key
    const key = `${busPosition[0]},${busPosition[1]},${carPosition[0]},${carPosition[1]}`; // Position Order: Bus, Car
    // Get probabilities for this state
    const data = positionProbabilities[key];
    if (data) {
      const carMove = moveVehicleRandomly(
        carPosition,
        carTargetPosition,
        carIntermediatePosition,
        carMoveProgress,
        data.car // Use the car probabilities for this state
      );
      const busMove = moveVehicleRandomly(
        busPosition,
        busTargetPosition,
        busIntermediatePosition,
        busMoveProgress,
        data.bus // Use the bus probabilities for this state
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
    } else {
      console.warn(`No probability data found for state: ${key}`);
    }

    lastMoveTime = currentTime;
  }

  // Update progress
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

  if (diagramStep.checked && carMoveProgress === 1 && busMoveProgress === 1) {
    carRowInput.value = carPosition[0];
    carColInput.value = carPosition[1];
    busRowInput.value = busPosition[0];
    busColInput.value = busPosition[1];
    // _isPaused = true;
    queryPosition(false);
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
      x: lerp(
        currentPos[1] * SCALE_FACTOR,
        targetPos[1] * SCALE_FACTOR,
        progress
      ),
      y: lerp(
        currentPos[0] * SCALE_FACTOR,
        targetPos[0] * SCALE_FACTOR,
        progress
      ),
    };
  }

  if (progress < 0.4) {
    const t = progress * 2.5;
    return {
      x: lerp(
        currentPos[1] * SCALE_FACTOR,
        intermediatePos[0] * SCALE_FACTOR,
        t
      ),
      y: lerp(
        currentPos[0] * SCALE_FACTOR,
        intermediatePos[1] * SCALE_FACTOR,
        t
      ),
    };
  } else if (progress < 0.6) {
    if (intermediatePos[0] < 0 || intermediatePos[0] > 2) {
      return {
        x: intermediatePos[0] < 0 ? 3 * SCALE_FACTOR : -1 * SCALE_FACTOR,
        y: targetPos[0] * SCALE_FACTOR,
      };
    } else {
      return {
        x: targetPos[1] * SCALE_FACTOR,
        y: intermediatePos[1] < 0 ? 3 * SCALE_FACTOR : -1 * SCALE_FACTOR,
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
async function setupCanvas() {
  const probabilitiesLoaded = await loadProbabilityData();
  await loadRewardData();
  canvas.width = 200 * NUM_COLS * SCALE_FACTOR;
  canvas.height = 200 * NUM_ROWS * SCALE_FACTOR;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Load both datasets
  // loadProbabilityData();

  createGrid();

  // Draw both vehicles in initial positions
  const [carPixelX, carPixelY] = getPixelPosition(carPosition);
  const [busPixelX, busPixelY] = getPixelPosition(busPosition);
  drawCar(carPixelX, carPixelY);
  drawBus(busPixelX, busPixelY);
  setupDraggableModal();
  setupSettingsModal();
  requestAnimationFrame(animate);
}

function setupSettingsModal() {
  const settingsModal = document.getElementById("settingsModal");
  const settingsHeader = settingsModal.querySelector(".modal-header");
  const settingsContent = settingsModal.querySelector(".modal-content");
  const settingsMinimizeBtn = settingsModal.querySelector(".minimize-btn");
  const darkModeToggle = document.getElementById("darkModeToggle");

  // Position settings modal
  settingsModal.style.transform = "translate(240px, 20px)";

  function toggleContent(intent = "none") {
    const isCurrentlyMinimized = settingsContent.style.display === "none";

    // If no specific intent, just toggle
    if (intent === "none") {
      if (isCurrentlyMinimized) {
        maximize();
      } else {
        minimize();
      }
    }
    // If specific intent, only act if needed
    else if (intent === "minimize" && !isCurrentlyMinimized) {
      minimize();
    } else if (intent === "maximize" && isCurrentlyMinimized) {
      maximize();
    }
  }

  function minimize() {
    settingsModal.dataset.previousHeight = settingsModal.offsetHeight + "px";
    settingsContent.style.display = "none";
    settingsMinimizeBtn.textContent = "+";
    settingsModal.style.height = settingsHeader.offsetHeight + "px";
  }

  function maximize() {
    settingsContent.style.display = "block";
    settingsMinimizeBtn.textContent = "−";
    if (settingsModal.dataset.previousHeight) {
      settingsModal.style.height = settingsModal.dataset.previousHeight;
      settingsModal.dataset.previousHeight = "";
    } else {
      settingsModal.style.height = "auto";
    }
  }

  // Setup minimize functionality
  settingsMinimizeBtn.addEventListener("click", () => {
    toggleContent();
  });

  // Setup dark mode toggle
  darkModeToggle.addEventListener("change", () => {
    document.body.classList.toggle("dark-mode");
    // Save preference
    localStorage.setItem("darkMode", darkModeToggle.checked);
  });

  // Load saved preference
  const savedDarkMode = localStorage.getItem("darkMode") === "true";
  darkModeToggle.checked = savedDarkMode;
  if (savedDarkMode) {
    document.body.classList.add("dark-mode");
  }

  // Make settings modal draggable
  let isDragging = false;
  let currentX;
  let currentY;
  let initialX;
  let initialY;
  let xOffset = 620;
  let yOffset = 20;

  settingsModal.style.transform = `translate(${xOffset}px, ${yOffset}px)`;
  function dragStart(e) {
    if (e.type === "touchstart") {
      initialX = e.touches[0].clientX - xOffset;
      initialY = e.touches[0].clientY - yOffset;
    } else {
      initialX = e.clientX - xOffset;
      initialY = e.clientY - yOffset;
    }

    if (e.target === settingsHeader) {
      isDragging = true;
    }
  }

  function dragEnd() {
    isDragging = false;
  }

  function drag(e) {
    if (isDragging) {
      e.preventDefault();

      if (e.type === "touchmove") {
        currentX = e.touches[0].clientX - initialX;
        currentY = e.touches[0].clientY - initialY;
      } else {
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;
      }

      xOffset = currentX;
      yOffset = currentY;

      settingsModal.style.transform = `translate(${currentX}px, ${currentY}px)`;
    }
  }

  settingsHeader.addEventListener("touchstart", dragStart, false);
  settingsHeader.addEventListener("touchend", dragEnd, false);
  settingsHeader.addEventListener("touchmove", drag, false);
  settingsHeader.addEventListener("mousedown", dragStart, false);
  document.addEventListener("mouseup", dragEnd, false);
  document.addEventListener("mousemove", drag, false);
}

// Toggle the modal
function toggleContent(intent = "none") {
  const isCurrentlyMinimized = content.style.display === "none";

  // If no specific intent, just toggle
  if (intent === "none") {
    if (isCurrentlyMinimized) {
      maximize();
    } else {
      minimize();
    }
  }
  // If specific intent, only act if needed
  else if (intent === "minimize" && !isCurrentlyMinimized) {
    minimize();
  } else if (intent === "maximize" && isCurrentlyMinimized) {
    maximize();
  }
}

function minimize() {
  modal.dataset.previousHeight = modal.offsetHeight + "px";
  content.style.display = "none";
  minimizeBtn.textContent = "+";
  modal.style.height = header.offsetHeight + "px";
}

function maximize() {
  content.style.display = "block";
  minimizeBtn.textContent = "−";
  if (modal.dataset.previousHeight) {
    modal.style.height = modal.dataset.previousHeight;
    modal.dataset.previousHeight = "";
  } else {
    modal.style.height = "auto";
  }
}

// Create a modal for controls
function setupDraggableModal() {
  // Controls Modal
  const modal = document.getElementById("controlsModal");
  const header = modal.querySelector(".modal-header");
  const content = modal.querySelector(".modal-content");
  const minimizeBtn = modal.querySelector(".minimize-btn");
  let isDragging = false;
  let currentX;
  let currentY;
  let initialX;
  let initialY;
  let xOffset = 620;
  let yOffset = 220;

  // Set initial position
  modal.style.transform = `translate(${xOffset}px, ${yOffset}px)`;

  function toggleContent(intent = "none") {
    const isCurrentlyMinimized = content.style.display === "none";

    // If no specific intent, just toggle
    if (intent === "none") {
      if (isCurrentlyMinimized) {
        maximize();
      } else {
        minimize();
      }
    }
    // If specific intent, only act if needed
    else if (intent === "minimize" && !isCurrentlyMinimized) {
      minimize();
    } else if (intent === "maximize" && isCurrentlyMinimized) {
      maximize();
    }
  }

  function minimize() {
    modal.dataset.previousHeight = modal.offsetHeight + "px";
    content.style.display = "none";
    minimizeBtn.textContent = "+";
    modal.style.height = header.offsetHeight + "px";
  }

  function maximize() {
    content.style.display = "block";
    minimizeBtn.textContent = "−";
    if (modal.dataset.previousHeight) {
      modal.style.height = modal.dataset.previousHeight;
      modal.dataset.previousHeight = "";
    } else {
      modal.style.height = "auto";
    }
  }

  // Handle minimize/maximize
  minimizeBtn.addEventListener("click", () => {
    toggleContent();
  });

  function dragStart(e) {
    if (e.type === "touchstart") {
      initialX = e.touches[0].clientX - xOffset;
      initialY = e.touches[0].clientY - yOffset;
    } else {
      initialX = e.clientX - xOffset;
      initialY = e.clientY - yOffset;
    }

    if (e.target === header) {
      isDragging = true;
    }
  }

  function dragEnd() {
    isDragging = false;
    // SCALE_FACTOR = 1;
  }

  function drag(e) {
    if (isDragging) {
      e.preventDefault();

      if (e.type === "touchmove") {
        currentX = e.touches[0].clientX - initialX;
        currentY = e.touches[0].clientY - initialY;
      } else {
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;
      }

      xOffset = currentX;
      yOffset = currentY;

      modal.style.transform = `translate(${currentX}px, ${currentY}px)`;
    }
  }

  header.addEventListener("touchstart", dragStart, false);
  header.addEventListener("touchend", dragEnd, false);
  header.addEventListener("touchmove", drag, false);
  header.addEventListener("mousedown", dragStart, false);
  document.addEventListener("mouseup", dragEnd, false);
  document.addEventListener("mousemove", drag, false);
  // toggleContent("minimize");
}

const scaleFactorInput = document.getElementById("scaleFactor");
scaleFactorInput.addEventListener("change", (e) => {
  SCALE_FACTOR = parseFloat(e.target.value);
  setupCanvas(); // Redraw everything with new scale
});
// Start the animation
setupCanvas();
