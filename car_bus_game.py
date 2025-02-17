# Implementation of a two-player zero-sum game between a car and a bus on a 3x3 grid
# The car tries to maximize rewards while avoiding collision, the bus tries to catch the car
import numpy as np
from itertools import product
from collections import defaultdict

class CarBusGame:
    def __init__(self):
        # Size of the square grid (3x3)
        self.grid_size = 3
        # Available actions for both players (movement in four directions)
        self.actions = ['up', 'down', 'left', 'right']
        # Rewards/penalties for collision events
        self.collision_reward = {'bus': 10.0, 'car': -10.0}  # Bus wins on collision, car loses
        # Discount factor for future rewards (0.8 means future rewards are worth 80% of immediate rewards)
        self.discount = 0.8
        
        # Grid defining position-based rewards:
        # - High rewards (5.0) in top-left and bottom-left corners
        # - Medium reward (2.0) in center
        # - Small penalties (-0.5) in rightmost column
        # - No rewards (0.0) in remaining positions
        self.reward_grid = np.array([
            [5.0,  0.0, -0.5],  # Top row    [TL, TC, TR]
            [0.0,  2.0, -0.5],  # Middle row [ML, MM, MR]
            [5.0,  0.0, -0.5]   # Bottom row [BL, BC, BR]
        ])

    def get_next_position(self, current_pos, action):
        # Calculate next position with wrap-around movement (toroidal grid)
        x, y = current_pos
        if action == 'up':
            return ((x - 1) % self.grid_size, y)  # Wrap to bottom if moving up from top
        elif action == 'down':
            return ((x + 1) % self.grid_size, y)  # Wrap to top if moving down from bottom
        elif action == 'left':
            return (x, (y - 1) % self.grid_size)  # Wrap to right if moving left from leftmost
        else:  # right
            return (x, (y + 1) % self.grid_size)  # Wrap to left if moving right from rightmost

    def get_reward(self, car_pos, bus_pos):
        # Calculate rewards for both players based on their positions and potential collision
        
        # Get position-based rewards from the reward grid
        car_tile_reward = self.reward_grid[car_pos[0]][car_pos[1]]
        bus_tile_reward = self.reward_grid[bus_pos[0]][bus_pos[1]]
        
        # If collision occurs, add collision rewards to position-based rewards
        if car_pos == bus_pos:
            return (self.collision_reward['car'] + car_tile_reward,  # Car gets negative collision reward plus tile reward
                   self.collision_reward['bus'] + bus_tile_reward)   # Bus gets positive collision reward plus tile reward
        
        # If no collision, players only get their position-based rewards
        return (car_tile_reward, bus_tile_reward)

def compute_value_function(game, car_strategy, bus_strategy, max_iterations=100):
    """
    Compute the expected discounted future rewards (value function) for all states
    using dynamic programming with current player strategies
    """
    # Initialize value functions for both players
    car_values = defaultdict(float)
    bus_values = defaultdict(float)
    
    # Iterate to converge to stable values
    for _ in range(max_iterations):
        new_car_values = defaultdict(float)
        new_bus_values = defaultdict(float)
        
        # Iterate through all possible state combinations
        for car_pos, bus_pos in product(product(range(game.grid_size), repeat=2),
                                      product(range(game.grid_size), repeat=2)):
            state = (car_pos, bus_pos)
            car_value = 0
            bus_value = 0
            
            # Calculate expected value by considering all possible action combinations
            for car_action_idx, car_action in enumerate(game.actions):
                for bus_action_idx, bus_action in enumerate(game.actions):
                    # Get action probabilities from current strategies
                    car_prob = car_strategy[state][car_action_idx]
                    bus_prob = bus_strategy[state][bus_action_idx]
                    
                    # Calculate next state after both players move
                    next_car_pos = game.get_next_position(car_pos, car_action)
                    next_bus_pos = game.get_next_position(bus_pos, bus_action)
                    next_state = (next_car_pos, next_bus_pos)
                    
                    # Get immediate rewards for this action combination
                    car_immediate, bus_immediate = game.get_reward(next_car_pos, next_bus_pos)
                    
                    # Calculate joint probability and update expected values
                    prob = car_prob * bus_prob
                    # Value = immediate reward + discounted future value
                    car_value += prob * (car_immediate + game.discount * car_values[next_state])
                    bus_value += prob * (bus_immediate + game.discount * bus_values[next_state])
            
            # Store computed values for this state
            new_car_values[state] = car_value
            new_bus_values[state] = bus_value
        
        # Update value functions
        car_values = new_car_values
        bus_values = new_bus_values
    
    return car_values, bus_values

def fictitious_play(game, iterations=1000):
    """
    Implementation of fictitious play algorithm to find approximate Nash equilibrium
    by iteratively computing best responses to average historical play
    """
    # Initialize uniform random strategies for both players
    car_strategy = defaultdict(lambda: np.ones(len(game.actions)) / len(game.actions))
    bus_strategy = defaultdict(lambda: np.ones(len(game.actions)) / len(game.actions))
    
    # Track historical action counts to compute average strategies
    car_counts = defaultdict(lambda: np.zeros(len(game.actions)))
    bus_counts = defaultdict(lambda: np.zeros(len(game.actions)))

    # Main fictitious play loop
    for _ in range(iterations):
        # Compute value functions for current strategies
        car_values, bus_values = compute_value_function(game, car_strategy, bus_strategy)
        
        # Update best responses for all possible states
        for car_pos, bus_pos in product(product(range(game.grid_size), repeat=2),
                                      product(range(game.grid_size), repeat=2)):
            state = (car_pos, bus_pos)
            
            # Calculate car's best response considering future rewards
            car_payoffs = np.zeros(len(game.actions))
            for car_action_idx, car_action in enumerate(game.actions):
                next_car_pos = game.get_next_position(car_pos, car_action)
                expected_reward = 0
                
                # Consider all possible bus actions and their probabilities
                for bus_action_idx, bus_action in enumerate(game.actions):
                    next_bus_pos = game.get_next_position(bus_pos, bus_action)
                    next_state = (next_car_pos, next_bus_pos)
                    car_immediate, _ = game.get_reward(next_car_pos, next_bus_pos)
                    future_value = car_values[next_state]
                    
                    # Combine immediate and future rewards with discount
                    reward = car_immediate + game.discount * future_value
                    expected_reward += reward * bus_strategy[state][bus_action_idx]
                    
                car_payoffs[car_action_idx] = expected_reward
            
            # Update car's strategy based on best response
            best_car_action = np.argmax(car_payoffs)
            car_counts[state][best_car_action] += 1
            car_strategy[state] = car_counts[state] / np.sum(car_counts[state])

            # Calculate bus's best response considering future rewards
            bus_payoffs = np.zeros(len(game.actions))
            for bus_action_idx, bus_action in enumerate(game.actions):
                next_bus_pos = game.get_next_position(bus_pos, bus_action)
                expected_reward = 0
                
                # Consider all possible car actions and their probabilities
                for car_action_idx, car_action in enumerate(game.actions):
                    next_car_pos = game.get_next_position(car_pos, car_action)
                    next_state = (next_car_pos, next_bus_pos)
                    _, bus_immediate = game.get_reward(next_car_pos, next_bus_pos)
                    future_value = bus_values[next_state]
                    
                    # Combine immediate and future rewards with discount
                    reward = bus_immediate + game.discount * future_value
                    expected_reward += reward * car_strategy[state][car_action_idx]
                    
                bus_payoffs[bus_action_idx] = expected_reward
            
            # Update bus's strategy based on best response
            best_bus_action = np.argmax(bus_payoffs)
            bus_counts[state][best_bus_action] += 1
            bus_strategy[state] = bus_counts[state] / np.sum(bus_counts[state])

    return car_strategy, bus_strategy

def main():
    # Create game instance
    game = CarBusGame()
    # Display game parameters
    print("Reward grid:")
    print(game.reward_grid)
    print("\nCollision reward (bus):", game.collision_reward['bus'])
    print("Collision penalty (car):", game.collision_reward['car'])
    
    # Compute optimal strategies using fictitious play
    car_strategy, bus_strategy = fictitious_play(game)
    
    # Display example optimal strategies for a specific state
    example_state = ((0, 0), (1, 1))  # Car at (0,0), Bus at (1,1)
    print(f"\nState: Car at {example_state[0]}, Bus at {example_state[1]}")
    print("Car strategy:", dict(zip(game.actions, car_strategy[example_state])))
    print("Bus strategy:", dict(zip(game.actions, bus_strategy[example_state])))

if __name__ == "__main__":
    main()
