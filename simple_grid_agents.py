import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class SimpleValueNetwork(nn.Module):
    """Simple neural network to predict value of a position"""
    def __init__(self, input_dim=2, hidden_dim=16):
        super(SimpleValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # Input layer to first hidden layer
            nn.ReLU(),                            # Activation function
            nn.Linear(hidden_dim, hidden_dim),    # First hidden layer to second hidden layer
            nn.ReLU(),                            # Activation function
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class SimpleGridWorld:
    def __init__(self):
        # Size of the square grid (3x3)
        self.grid_size = 3
        # Available actions (movement in four directions plus stay)
        self.actions = ['up', 'down', 'left', 'right', 'stay']
        # Discount factor for future rewards
        self.discount = 0.8
        
        # Grid defining position-based rewards
        self.reward_grid = np.array([
            [5.0,  0.0, -0.5],  # Top row    [TL, TC, TR]
            [0.0,  2.0, -0.5],  # Middle row [ML, MM, MR]
            [5.0,  0.0, -0.5]   # Bottom row [BL, BC, BR]
        ])
        
        # All possible positions on the grid
        self.positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

    def get_next_position(self, current_pos, action):
        """Calculate next position with wrap-around movement (toroidal grid)"""
        if action == 'stay':
            return current_pos  # Stay in the same position
            
        x, y = current_pos
        if action == 'up':
            return ((x - 1) % self.grid_size, y)
        elif action == 'down':
            return ((x + 1) % self.grid_size, y)
        elif action == 'left':
            return (x, (y - 1) % self.grid_size)
        else:  # right
            return (x, (y + 1) % self.grid_size)

    def get_reward(self, position):
        """Get the reward for a position"""
        return self.reward_grid[position[0]][position[1]]

    def position_to_tensor(self, position):
        """Convert a position to a tensor for neural network input"""
        return torch.tensor([position[0] / (self.grid_size-1), position[1] / (self.grid_size-1)], 
                           dtype=torch.float32)

    def generate_training_data(self, num_samples=500):
        """Generate training data for value function approximation"""
        X, y = [], []
        
        # Visit each position and compute the target value
        for pos in self.positions:
            # Get immediate reward for this position
            immediate_reward = self.get_reward(pos)
            
            # Find max future value by looking at all possible next positions
            future_values = []
            for action in self.actions:
                next_pos = self.get_next_position(pos, action)
                next_reward = self.get_reward(next_pos)
                future_values.append(next_reward)
            
            max_future_value = max(future_values)
            
            # Target is immediate reward + discounted future value
            target = immediate_reward + self.discount * max_future_value
            
            # Add to training data multiple times for positions with higher rewards
            repeat_factor = int(5 * (immediate_reward + 1.5)) if immediate_reward > 0 else 1
            for _ in range(repeat_factor):
                X.append(self.position_to_tensor(pos))
                y.append(target)
        
        # Add some random jitter for regularization
        for _ in range(num_samples):
            pos = random.choice(self.positions)
            X.append(self.position_to_tensor(pos) + torch.randn(2) * 0.05)
            
            immediate_reward = self.get_reward(pos)
            future_values = [self.get_reward(self.get_next_position(pos, a)) for a in self.actions]
            max_future_value = max(future_values)
            target = immediate_reward + self.discount * max_future_value
            
            y.append(target + random.uniform(-0.2, 0.2))
            
        return torch.stack(X), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

def train_value_network(world, agent_name, epochs=100):
    """Train a neural network to predict the value of a position"""
    print(f"\nTraining {agent_name}'s value network...")
    
    # Generate training data
    X, y = world.generate_training_data()
    
    # Create and train the model
    model = SimpleValueNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def find_best_path(world, model, start_pos, num_steps=5):
    """Find the best path from a starting position using the trained model"""
    path = [start_pos]
    current_pos = start_pos
    actions_taken = []
    
    best_reward = np.max(world.reward_grid)
    for _ in range(num_steps):
        if world.get_reward(current_pos) == best_reward:
                    break

        best_action = None
        best_value = float('-inf')
        
        # Try each action and pick the one with highest predicted value
        for action in world.actions:
            next_pos = world.get_next_position(current_pos, action)
            pos_tensor = world.position_to_tensor(next_pos)
            
            with torch.no_grad():
                predicted_value = model(pos_tensor).item()
            
            if predicted_value > best_value:
                best_value = predicted_value
                best_action = action
        
        # Move to the best next position
        current_pos = world.get_next_position(current_pos, best_action)
        path.append(current_pos)
        actions_taken.append(best_action)
    
    return path, actions_taken

def main():
    # Create world
    world = SimpleGridWorld()
    
    print("Reward grid:")
    print(world.reward_grid)
    
    # Train value networks for car and bus (they're identical since they don't interact)
    car_model = train_value_network(world, "car", epochs=100)
    
    # We could use the same model for both agents since they're independent and identical
    # But for clarity, let's train a separate model for the bus
    bus_model = train_value_network(world, "bus", epochs=100)
    
    # Find best paths from different starting positions
    for start_pos in [(0, 0), (1, 1), (2, 2)]:
        print(f"\nStarting from position {start_pos}:")
        
        car_path, car_actions = find_best_path(world, car_model, start_pos)
        print(f"Car's path: {car_path}")
        print(f"Car's actions: {car_actions}")
        print(f"Car's rewards: {[world.get_reward(pos) for pos in car_path]}")
        
        bus_path, bus_actions = find_best_path(world, bus_model, start_pos)
        print(f"Bus's path: {bus_path}")
        print(f"Bus's actions: {bus_actions}")
        print(f"Bus's rewards: {[world.get_reward(pos) for pos in bus_path]}")

if __name__ == "__main__":
    main() 