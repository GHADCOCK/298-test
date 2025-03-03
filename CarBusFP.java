import java.io.File;
import java.util.ArrayList;
import processing.core.PApplet;
import processing.core.PImage;
import java.util.Random;
import java.util.Arrays;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class CarBusFP extends PApplet {
    private ArrayList<Square> squares;
    Player car;
    Player bus;
    public static double[][] rewards = {
        {0, 5, 0},
        {5, 10, 5},
        {0, 5, 0}
    };
    public static double crashReward = 20;
    public static double discountFactor = 0.9;
    public static double learningRate = 0.1;
    int decisionInterval = 15;
    int decisionTimer = 0;
    int iterationCount = 0; 
    double convergenceThreshold = 0.01; // Convergence threshold

    Random rand = new Random();

    public static void main(String[] args) {
        PApplet.main("CarBusFP");
    }

    @Override
    public void settings() {
        size(900, 900);
    }

    @Override
    public void setup() {
        this.imageMode(PApplet.CENTER);
        this.rectMode(PApplet.CENTER);
        this.focused = true;
        this.textAlign(PApplet.CENTER);
        this.textSize(30);
        frameRate(60);

        squares = new ArrayList<>();
        background(255);
        Square.setProcessing(this);

        // Create squares
        squares.add(new Square("blue", 0, 0, "images" + File.separator + "bluesquare.png"));
        squares.add(new Square("white", 0, 0, "images" + File.separator + "whitesquare.png"));
        squares.add(new Square("blue", 0, 0, "images" + File.separator + "bluesquare.png"));
        squares.add(new Square("white", 0, 0, "images" + File.separator + "whitesquare.png"));
        squares.add(new Square("red", 0, 0, "images" + File.separator + "redsquare.jpg"));
        squares.add(new Square("white", 0, 0, "images" + File.separator + "whitesquare.png"));
        squares.add(new Square("blue", 0, 0, "images" + File.separator + "bluesquare.png"));
        squares.add(new Square("white", 0, 0, "images" + File.separator + "whitesquare.png"));
        squares.add(new Square("blue", 0, 0, "images" + File.separator + "bluesquare.png"));

        // Resize square images
        for (Square square : squares) {
            square.image.resize(300, 300);
        }

        Player.setProcessing(this);

        // Initialize car and bus with random positions
        int carRow = rand.nextInt(3);
        int carCol = rand.nextInt(3);
        int busRow = rand.nextInt(3);
        int busCol = rand.nextInt(3);

        while (carRow == busRow && carCol == busCol) {
            busRow = rand.nextInt(3);
            busCol = rand.nextInt(3);
        }

        int carX = 150 + carCol * 300;
        int carY = 150 + carRow * 300;
        int busX = 150 + busCol * 300;
        int busY = 150 + busRow * 300;

        car = new Player(1, "images" + File.separator + "car.jpg", carX, carY);
        bus = new Player(2, "images" + File.separator + "bus.jpg", busX, busY);
        car.setOtherPlayer(bus);
        bus.setOtherPlayer(car);

        // Resize car and bus images
        car.image.resize(100, 100);
        bus.image.resize(150, 100);
    }

    @Override
    public void draw() {
        background(255);

        // Draw squares on the grid
        for (int i = 0; i < 9; i++) {
            int x = 150 + (i % 3) * 300;
            int y = 150 + (i / 3) * 300;
            image(squares.get(i).image, x, y);
        }

        // Draw car and bus at their current positions
        image(car.image, car.x, car.y);
        image(bus.image, bus.x, bus.y);

        // Update decision timer
        decisionTimer++;
        if (decisionTimer >= decisionInterval) {
            decisionTimer = 0;
            makeDecision();
        }

        // Move car and bus continuously
        car.move();
        bus.move();

        // Display rewards for car and bus
        fill(0);
        text("Car Reward: " + car.reward, 150, 850);
        text("Bus Reward: " + bus.reward, 450, 850);
    }

    void makeDecision() {
      // Increment iteration counter
      iterationCount++;
      System.out.println("Iteration: " + iterationCount);

      // Get current states for both players
      int carState = car.getState();
      int busState = bus.getState();
      
      // Choose actions using Fictitious Play principles
      int carAction = car.getBestAction(carState);
      int busAction = bus.getBestAction(busState);

      // Debug: Print chosen actions
      System.out.println("Car action: " + carAction + ", Bus action: " + busAction);

      // Set target positions based on actions
      car.setTarget(carAction, bus);
      bus.setTarget(busAction, car);

      // Store old Q-tables for convergence check
      int[][] carQTablePrev = deepCopyQTable(car.qTable);
      int[][] busQTablePrev = deepCopyQTable(bus.qTable);

      // Update rewards based on current state
      updateRewards();

      // Get next states
      int carNextState = car.getState();
      int busNextState = bus.getState();

      // Update Q-values
      car.updateQValue(carState, carAction, car.reward, carNextState);
      bus.updateQValue(busState, busAction, bus.reward, busNextState);

      // Update beliefs about opponent's strategy
      car.updateBelief(carState, busAction);
      bus.updateBelief(busState, carAction);

      // Check for convergence
      double carQDiff = calculateQValueDifference(car.qTable, carQTablePrev);
      double busQDiff = calculateQValueDifference(bus.qTable, busQTablePrev);

      System.out.println("Car Q-value difference: " + carQDiff);
      System.out.println("Bus Q-value difference: " + busQDiff);

      if (carQDiff < convergenceThreshold && busQDiff < convergenceThreshold) {
          System.out.println("Q-learning converged after " + iterationCount + " iterations.");
          saveNashEquilibrium();
          if (iterationCount > 100) {
              noLoop(); // Stop the simulation
          }
      }
  }
    
    double calculateQValueDifference(int[][] qTable, int[][] qTablePrev) {
      double totalDiff = 0;
      int numStates = qTable.length;
      int numActions = qTable[0].length;

      for (int s = 0; s < numStates; s++) {
          for (int a = 0; a < numActions; a++) {
              totalDiff += Math.abs(qTable[s][a] - qTablePrev[s][a]);
          }
      }

      // Normalize by dividing by the total number of state-action pairs
      return totalDiff / (numStates * numActions);
  }

  int[][] deepCopyQTable(int[][] qTable) {
      int[][] copy = new int[qTable.length][qTable[0].length];
      for (int s = 0; s < qTable.length; s++) {
          for (int a = 0; a < qTable[s].length; a++) {
              copy[s][a] = qTable[s][a];
          }
      }
      return copy;
  }

    void printQTable(int[][] qTable) {
        for (int s = 0; s < qTable.length; s++) {
            for (int a = 0; a < qTable[s].length; a++) {
                System.out.print(qTable[s][a] + " ");
            }
            System.out.println();
        }
    }

    void updateRewards() {
        // Get current state (row and column for car and bus)
        int carRow = (car.y - 150) / 300;
        int carCol = (car.x - 150) / 300;
        int busRow = (bus.y - 150) / 300;
        int busCol = (bus.x - 150) / 300;

        // Reset rewards before applying new rewards
        car.reward = 0;
        bus.reward = 0;

        // Update rewards based on square color
        car.reward += rewards[carRow][carCol];
        bus.reward += rewards[busRow][busCol];

        // Check for collision with some tolerance
        int collisionTolerance = 10;
        if (Math.abs(car.x - bus.x) < collisionTolerance && Math.abs(car.y - bus.y) < collisionTolerance) {
            car.reward += -crashReward;
            bus.reward += crashReward;
            System.out.println("Collision detected!");
        }

        // Debug: Print rewards
        System.out.println("Car reward: " + car.reward + ", Bus reward: " + bus.reward);
    }

    
    void saveNashEquilibrium() {
      try {
          PrintWriter writer = new PrintWriter(new FileWriter("nash_equilibrium.txt"));
          
          // Write the format header
          writer.println("# Format: X_1,Y_1,X_2,Y_2:U1,D1,L1,R1,U2,D2,L2,R2");
          writer.println("# This file contains movement probabilities for both vehicles based on their positions");
          writer.println("# X_1,Y_1: Car position, X_2,Y_2: Bus position");
          writer.println("# U1,D1,L1,R1: Car Up,Down,Left,Right probabilities");
          writer.println("# U2,D2,L2,R2: Bus Up,Down,Left,Right probabilities");
          
          for (int state = 0; state < 81; state++) {
              // Extract positions from state - using the correct order
              int carRow = (state / 27) % 3;
              int carCol = (state / 9) % 3;
              int busRow = (state / 3) % 3;
              int busCol = state % 3;
              
              // Get probabilities for car and bus
              double[] carStrategy = car.getOptimalStrategy(state);
              double[] busStrategy = bus.getOptimalStrategy(state);
              
              // Format the line: X_1,Y_1,X_2,Y_2:U1,D1,L1,R1,U2,D2,L2,R2
              writer.printf("%d,%d,%d,%d:%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                            carRow, carCol, busRow, busCol,
                            carStrategy[0], carStrategy[1], carStrategy[2], carStrategy[3],
                            busStrategy[0], busStrategy[1], busStrategy[2], busStrategy[3]);
          }
          
          writer.close();
          System.out.println("Nash equilibrium probabilities saved to nash_equilibrium.txt");
      } catch (IOException e) {
          System.err.println("Error writing to file: " + e.getMessage());
      }
  }
    
    
}

class Square {
    String color;
    int x;
    int y;
    PImage image; // Image for the square
    static PApplet processing;

    Square(String color, int x, int y, String imagePath) {
        this.color = color;
        this.x = x;
        this.y = y;
        this.image = processing.loadImage(imagePath);
    }

    static void setProcessing(PApplet p) {
        processing = p;
    }
}

class Player {
  int id;
  PImage image;
  int x, y;
  int targetX, targetY;
  double reward;
  int[][] qTable; // Q-table for this player (state, action)
  int moveSpeed = 20;
  static PApplet processing;
  Player otherPlayer;
  Random rand = new Random();

  // Fictitious Play variables - track opponent beliefs by state
  int[][] opponentActionCounts; // [state][action]
  int[] totalStateVisits;       // [state]
  
  // For collision handling
  int previousAction = -1;
  Player(int id, String imagePath, int x, int y) {
    this.id = id;
    this.image = processing.loadImage(imagePath);
    this.x = x;
    this.y = y;
    this.targetX = x;
    this.targetY = y;
    this.reward = 0;
    
    // Q-table for 81 states (3x3x3x3 grid positions) and 4 actions
    this.qTable = new int[81][4];
    
    // Initialize opponent modeling structures - track beliefs for each state
    this.opponentActionCounts = new int[81][4];
    this.totalStateVisits = new int[81];
    
    // Initialize Q-values with small random values
    initializeQTable();
    
    this.otherPlayer = null;
}

  void initializeQTable() {
    for (int s = 0; s < 81; s++) {
        // Convert state index to car and bus positions
        int carRow = (s / 27) % 3;
        int carCol = (s / 9) % 3;
        int busRow = (s / 3) % 3;
        int busCol = s % 3;

        for (int a = 0; a < 4; a++) {
            // Calculate new position after action
            int newRow = carRow;
            int newCol = carCol;
            
            if (id == 1) { // Car
                if (a == 0) newRow = (carRow - 1 + 3) % 3;      // Up
                else if (a == 1) newRow = (carRow + 1) % 3;     // Down
                else if (a == 2) newCol = (carCol - 1 + 3) % 3; // Left
                else if (a == 3) newCol = (carCol + 1) % 3;     // Right
                
                // Initialize Q-value based on rewards
                double reward = CarBusFP.rewards[newRow][newCol];
                
                // Penalize potential collisions
                if (newRow == busRow && newCol == busCol) {
                    reward -= CarBusFP.crashReward;
                }
                
                qTable[s][a] = (int)(reward * 100);
            }
            else { // Bus (id == 2)
                if (a == 0) newRow = (busRow - 1 + 3) % 3;      // Up
                else if (a == 1) newRow = (busRow + 1) % 3;     // Down
                else if (a == 2) newCol = (busCol - 1 + 3) % 3; // Left
                else if (a == 3) newCol = (busCol + 1) % 3;     // Right
                
                // Initialize Q-value based on rewards
                double reward = CarBusFP.rewards[newRow][newCol];
                
                // Reward potential collisions for bus
                if (carRow == newRow && carCol == newCol) {
                    reward += CarBusFP.crashReward;
                }
                
                qTable[s][a] = (int)(reward * 100);
            }
        }
    }
}

  static void setProcessing(PApplet p) {
      processing = p;
  }

  void setOtherPlayer(Player otherPlayer) {
      this.otherPlayer = otherPlayer;
  }


 

  int getReverseAction(int action) {
      // Return the reverse of the given action
      switch (action) {
          case 0: return 1; // Up -> Down
          case 1: return 0; // Down -> Up
          case 2: return 3; // Left -> Right
          case 3: return 2; // Right -> Left
          default: return -1; // Invalid action
      }
  }

  double calculatePayoff(int state, int myAction, double[] opponentStrategy) {
      double expectedPayoff = 0;

      // Calculate expected payoff based on opponent's strategy
      for (int oppAction = 0; oppAction < 4; oppAction++) {
          double probability = opponentStrategy[oppAction];
          double payoff = getPayoffForActions(myAction, oppAction);
          expectedPayoff += probability * payoff;
      }

      return expectedPayoff;
  }

  double getPayoffForActions(int myAction, int oppAction) {
      // Calculate new positions for car and bus based on actions
      int carRow = (y - 150) / 300;
      int carCol = (x - 150) / 300;
      int busRow = (otherPlayer.y - 150) / 300;
      int busCol = (otherPlayer.x - 150) / 300;

      // Update car's position based on action
      if (myAction == 0) carRow = (carRow - 1 + 3) % 3; // Up
      else if (myAction == 1) carRow = (carRow + 1) % 3; // Down
      else if (myAction == 2) carCol = (carCol - 1 + 3) % 3; // Left
      else if (myAction == 3) carCol = (carCol + 1) % 3; // Right

      // Update bus's position based on action
      if (oppAction == 0) busRow = (busRow - 1 + 3) % 3; // Up
      else if (oppAction == 1) busRow = (busRow + 1) % 3; // Down
      else if (oppAction == 2) busCol = (busCol - 1 + 3) % 3; // Left
      else if (oppAction == 3) busCol = (busCol + 1) % 3; // Right

      // Calculate reward based on new positions
      double payoff = CarBusFP.rewards[carRow][carCol];

      // Check for collision
      if (carRow == busRow && carCol == busCol) {
          payoff += -CarBusFP.crashReward; // Negative reward for collision
      }

      return payoff;
  }


  // Update beliefs about opponent's strategy for a specific state
  void updateBelief(int state, int opponentAction) {
      opponentActionCounts[state][opponentAction]++;
      totalStateVisits[state]++;
  }

  // Update Q-value using observed reward and next state
  void updateQValue(int state, int action, double reward, int nextState) {
      // Get best Q-value for next state
      double bestNextQ = getBestValueForState(nextState);
      
      // Q-learning update rule
      qTable[state][action] += (int)(CarBusFP.learningRate * 
                                 (reward + CarBusFP.discountFactor * bestNextQ - qTable[state][action]));
      
      // Debug
      System.out.println("Player " + id + " updated Q[" + state + "][" + action + "] = " + qTable[state][action]);
  }

  void setTarget(int action, Player otherPlayer) {
      // Store the current action as the previous action
      previousAction = action;

      // Set the target position based on the action
      if (action == 0) { // Up
          targetY = (y - 300 + 900) % 900;
      } else if (action == 1) { // Down
          targetY = (y + 300) % 900;
      } else if (action == 2) { // Left
          targetX = (x - 300 + 900) % 900;
      } else if (action == 3) { // Right
          targetX = (x + 300) % 900;
      }

      // Handle collision avoidance
      if (Math.abs(x - otherPlayer.x) < 10 && Math.abs(y - otherPlayer.y) < 10) {
          if (action == 0) {
              targetY = (y + 300) % 900;
          } else if (action == 1) {
              targetY = (y - 300 + 900) % 900;
          } else if (action == 2) {
              targetX = (x + 300) % 900;
          } else if (action == 3) {
              targetX = (x - 300 + 900) % 900;
          }
      }
  }

  void move() {
      int dx = targetX - x;
      int dy = targetY - y;

      if (dx > 450) dx -= 900;
      else if (dx < -450) dx += 900;

      if (dy > 450) dy -= 900;
      else if (dy < -450) dy += 900;

      if (dx > 0) x += moveSpeed;
      else if (dx < 0) x -= moveSpeed;
      if (dy > 0) y += moveSpeed;
      else if (dy < 0) y -= moveSpeed;

      if (x < 0) x += 900;
      else if (x >= 900) x -= 900;
      if (y < 0) y += 900;
      else if (y >= 900) y -= 900;
  }
//Calculate immediate reward for a joint action
 double getReward(int state, int myAction, int oppAction) {
     // Extract positions from state
     int carRow = (state / 27) % 3;
     int carCol = (state / 9) % 3;
     int busRow = (state / 3) % 3;
     int busCol = state % 3;
     
     // Calculate new positions
     int newCarRow = carRow;
     int newCarCol = carCol;
     int newBusRow = busRow;
     int newBusCol = busCol;
     
     if (id == 1) { // Car perspective
         // Update car position based on my action
         if (myAction == 0) newCarRow = (carRow - 1 + 3) % 3;
         else if (myAction == 1) newCarRow = (carRow + 1) % 3;
         else if (myAction == 2) newCarCol = (carCol - 1 + 3) % 3;
         else if (myAction == 3) newCarCol = (carCol + 1) % 3;
         
         // Update bus position based on opponent action
         if (oppAction == 0) newBusRow = (busRow - 1 + 3) % 3;
         else if (oppAction == 1) newBusRow = (busRow + 1) % 3;
         else if (oppAction == 2) newBusCol = (busCol - 1 + 3) % 3;
         else if (oppAction == 3) newBusCol = (busCol + 1) % 3;
         
         // Calculate reward for car
         double reward = CarBusFP.rewards[newCarRow][newCarCol];
         
         // Check for collision
         if (newCarRow == newBusRow && newCarCol == newBusCol) {
             reward -= CarBusFP.crashReward;
         }
         
         return reward;
     }
     else { // Bus perspective
         // Update bus position based on my action
         if (myAction == 0) newBusRow = (busRow - 1 + 3) % 3;
         else if (myAction == 1) newBusRow = (busRow + 1) % 3;
         else if (myAction == 2) newBusCol = (busCol - 1 + 3) % 3;
         else if (myAction == 3) newBusCol = (busCol + 1) % 3;
         
         // Update car position based on opponent action
         if (oppAction == 0) newCarRow = (carRow - 1 + 3) % 3;
         else if (oppAction == 1) newCarRow = (carRow + 1) % 3;
         else if (oppAction == 2) newCarCol = (carCol - 1 + 3) % 3;
         else if (oppAction == 3) newCarCol = (carCol + 1) % 3;
         
         // Calculate reward for bus
         double reward = CarBusFP.rewards[newBusRow][newBusCol];
         
         // Check for collision
         if (newCarRow == newBusRow && newCarCol == newBusCol) {
             reward += CarBusFP.crashReward; // Positive for bus
         }
         
         return reward;
     }
 }
 
  //Get best action considering opponent's strategy (Nash-like decision)
  int getBestAction(int state) {
      // Get opponent's estimated strategy
      double[] opponentStrategy = getOpponentStrategy(state);
      
      // Choose action with highest expected value
      int bestAction = 0;
      double bestValue = Double.NEGATIVE_INFINITY;
      
      for (int myAction = 0; myAction < 4; myAction++) {
          double expectedValue = 0;
          
          // Calculate expected value against opponent's mixed strategy
          for (int oppAction = 0; oppAction < 4; oppAction++) {
              // Skip actions with zero probability
              if (opponentStrategy[oppAction] > 0) {
                  // Calculate next state and reward
                  int nextState = getNextState(state, myAction, oppAction);
                  double immediateReward = getReward(state, myAction, oppAction);
                  
                  // Add to expected value
                  expectedValue += opponentStrategy[oppAction] * 
                                  (immediateReward + CarBusFP.discountFactor * getBestValueForState(nextState));
              }
          }
          
          if (expectedValue > bestValue) {
              bestValue = expectedValue;
              bestAction = myAction;
          }
      }
      
      return bestAction;
  }
  
  //Calculate the next state given current state and joint actions
   int getNextState(int state, int myAction, int oppAction) {
       // Extract positions from state
       int carRow = (state / 27) % 3;
       int carCol = (state / 9) % 3;
       int busRow = (state / 3) % 3;
       int busCol = state % 3;
       
       // Update based on actions
       if (id == 1) { // Car perspective
           // Update car position
           if (myAction == 0) carRow = (carRow - 1 + 3) % 3;
           else if (myAction == 1) carRow = (carRow + 1) % 3;
           else if (myAction == 2) carCol = (carCol - 1 + 3) % 3;
           else if (myAction == 3) carCol = (carCol + 1) % 3;
           
           // Update bus position
           if (oppAction == 0) busRow = (busRow - 1 + 3) % 3;
           else if (oppAction == 1) busRow = (busRow + 1) % 3;
           else if (oppAction == 2) busCol = (busCol - 1 + 3) % 3;
           else if (oppAction == 3) busCol = (busCol + 1) % 3;
       } 
       else { // Bus perspective
           // Update bus position
           if (myAction == 0) busRow = (busRow - 1 + 3) % 3;
           else if (myAction == 1) busRow = (busRow + 1) % 3;
           else if (myAction == 2) busCol = (busCol - 1 + 3) % 3;
           else if (myAction == 3) busCol = (busCol + 1) % 3;
           
           // Update car position
           if (oppAction == 0) carRow = (carRow - 1 + 3) % 3;
           else if (oppAction == 1) carRow = (carRow + 1) % 3;
           else if (oppAction == 2) carCol = (carCol - 1 + 3) % 3;
           else if (oppAction == 3) carCol = (carCol + 1) % 3;
       }
       
       // Construct new state index
       return carRow * 27 + carCol * 9 + busRow * 3 + busCol;
   }
   
// Calculate the best value for a state (used for bootstrapping)
  double getBestValueForState(int state) {
      double[] opponentStrategy = getOpponentStrategy(state);
      double bestValue = Double.NEGATIVE_INFINITY;
      
      for (int myAction = 0; myAction < 4; myAction++) {
          double expectedValue = 0;
          
          for (int oppAction = 0; oppAction < 4; oppAction++) {
              if (opponentStrategy[oppAction] > 0) {
                  expectedValue += opponentStrategy[oppAction] * qTable[state][myAction];
              }
          }
          
          if (expectedValue > bestValue) {
              bestValue = expectedValue;
          }
      }
      
      return bestValue;
  }
  
  // Get opponent's estimated strategy for a state (returns probability distribution)
  double[] getOpponentStrategy(int state) {
      double[] strategy = new double[4];
      
      // If we've never seen this state, assume uniform distribution
      if (totalStateVisits[state] == 0) {
          Arrays.fill(strategy, 0.25);
          return strategy;
      }
      
      // Calculate probabilities based on observed frequencies
      for (int a = 0; a < 4; a++) {
          strategy[a] = (double) opponentActionCounts[state][a] / totalStateVisits[state];
          
          // Add small epsilon for exploration
          strategy[a] = 0.95 * strategy[a] + 0.05 * 0.25;
      }
      
      return strategy;
  }
  // Get the current state index
  int getState() {
      // For car player
      if (id == 1) {
          int carRow = (y - 150) / 300;
          int carCol = (x - 150) / 300;
          int busRow = (otherPlayer.y - 150) / 300;
          int busCol = (otherPlayer.x - 150) / 300;
          return carRow * 27 + carCol * 9 + busRow * 3 + busCol;
      }
      // For bus player
      else {
          int busRow = (y - 150) / 300;
          int busCol = (x - 150) / 300;
          int carRow = (otherPlayer.y - 150) / 300;
          int carCol = (otherPlayer.x - 150) / 300;
          return carRow * 27 + carCol * 9 + busRow * 3 + busCol;
      }
  }


//Add this method to your Player class
double[] getOptimalStrategy(int state) {
   double[] strategy = new double[4];
   double[] values = new double[4];
   double sum = 0;
   
   // Get Q-values for this state
   for (int a = 0; a < 4; a++) {
       values[a] = qTable[state][a];
       // Make all values positive by finding min value
       double minValue = Double.MAX_VALUE;
       for (int i = 0; i < 4; i++) {
           if (qTable[state][i] < minValue) {
               minValue = qTable[state][i];
           }
       }
       
       // Shift all values to be positive and add 1 to avoid division by zero
       for (int i = 0; i < 4; i++) {
           values[i] = qTable[state][i] - minValue + 1;
       }
   }
   
   // Convert to probabilities using softmax
   for (int a = 0; a < 4; a++) {
       sum += values[a];
   }
   
   // Normalize to get probabilities
   for (int a = 0; a < 4; a++) {
       strategy[a] = values[a] / sum;
   }
   
   return strategy;
}
}



