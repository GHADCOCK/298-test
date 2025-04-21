import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 3
NUM_ACTIONS = 4  # UP, DOWN, LEFT, RIGHT (removed STAY)
GAMMA = 0.99
LEARNING_RATE = 0.001
NUM_EPISODES = 1000
MAX_STEPS = 100

class PolicyNetwork(nn.Module):
    def __init__(self, shared=False):
        super().__init__()
        self.shared = shared
        input_dim = 4  # [chaser_x, chaser_y, escaper_x, escaper_y]
        hidden_dim = 64
        output_dim = NUM_ACTIONS
        
        if shared:
            # Shared network with two heads
            self.shared_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.chaser_head = nn.Linear(hidden_dim, output_dim)
            self.escaper_head = nn.Linear(hidden_dim, output_dim)
        else:
            # Separate networks
            self.chaser_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.escaper_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(self, state, agent_type, env=None):
        if self.shared:
            features = self.shared_net(state)
            if agent_type == 'chaser':
                logits = self.chaser_head(features)
            else:
                logits = self.escaper_head(features)
        else:
            if agent_type == 'chaser':
                logits = self.chaser_net(state)
            else:
                logits = self.escaper_net(state)
        
        # Get position from state
        if agent_type == 'chaser':
            pos = (int(state[0].item() * (GRID_SIZE-1)), int(state[1].item() * (GRID_SIZE-1)))
        else:
            pos = (int(state[2].item() * (GRID_SIZE-1)), int(state[3].item() * (GRID_SIZE-1)))
        
        # Create action mask
        valid_actions = env.get_valid_actions(pos)
        action_mask = torch.tensor(valid_actions, dtype=torch.float)
        
        # Apply mask and numerical stability in a differentiable way
        masked_logits = logits + (1 - action_mask) * -1e9  # Large negative number for invalid actions
        masked_logits = masked_logits - masked_logits.max()  # For numerical stability
        exp_logits = torch.exp(masked_logits)
        probs = exp_logits * action_mask  # Zero out invalid actions
        probs = probs / (probs.sum() + 1e-10)  # Normalize
        
        return probs

class GridWorld:
    def __init__(self):
        self.size = GRID_SIZE
        # Define tile rewards for 3x3 grid
        self.tile_rewards = np.array([
            [0.0,  5.0,  0.0],  # Row 0
            [5.0,  10.0, 5.0],  # Row 1
            [0.0,  5.0,  0.0]   # Row 2
        ])
        # Define capture rewards (larger than any tile reward)
        self.CAPTURE_REWARD = 20.0  # Greater than max tile reward (10.0)
        self.reset()
    
    def reset(self):
        # Random starting positions (not overlapping)
        while True:
            self.chaser_pos = np.random.randint(0, self.size, size=2)
            self.escaper_pos = np.random.randint(0, self.size, size=2)
            if not np.array_equal(self.chaser_pos, self.escaper_pos):
                break
        return self._get_state()
    
    def _get_state(self):
        # Normalize positions to [0,1]
        state = np.concatenate([
            self.chaser_pos / (self.size - 1),
            self.escaper_pos / (self.size - 1)
        ])
        return torch.FloatTensor(state)
    
    def step(self, chaser_action, escaper_action):
        # Action mapping: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        moves = {
            0: [-1, 0],  # UP
            1: [1, 0],   # DOWN
            2: [0, -1],  # LEFT
            3: [0, 1],   # RIGHT
        }
        
        # Store old positions for reward calculation
        old_chaser_pos = self.chaser_pos.copy()
        old_escaper_pos = self.escaper_pos.copy()
        
        # Move agents
        self.chaser_pos = np.clip(
            self.chaser_pos + moves[chaser_action],
            0, self.size - 1
        )
        self.escaper_pos = np.clip(
            self.escaper_pos + moves[escaper_action],
            0, self.size - 1
        )
        
        # Check if caught
        caught = np.array_equal(self.chaser_pos, self.escaper_pos)
        
        # Calculate rewards - but don't end the episode
        if caught:
            chaser_reward = self.CAPTURE_REWARD
            escaper_reward = -self.CAPTURE_REWARD
        else:
            # Tile-based rewards
            chaser_reward = self.tile_rewards[self.chaser_pos[0], self.chaser_pos[1]]
            escaper_reward = self.tile_rewards[self.escaper_pos[0], self.escaper_pos[1]]
            
            # Small time penalty/reward to encourage action
            chaser_reward -= 0.1
            escaper_reward += 0.1
        
        return self._get_state(), chaser_reward, escaper_reward, caught

    def get_valid_actions(self, pos):
        """Return a boolean mask of valid actions for a given position"""
        valid = np.ones(NUM_ACTIONS, dtype=bool)  # All actions start as valid
        
        # Check boundaries
        if pos[0] == 0:  # Top edge
            valid[0] = False  # Can't go UP
        if pos[0] == self.size - 1:  # Bottom edge
            valid[1] = False  # Can't go DOWN
        if pos[1] == 0:  # Left edge
            valid[2] = False  # Can't go LEFT
        if pos[1] == self.size - 1:  # Right edge
            valid[3] = False  # Can't go RIGHT
        
        return valid

def train_reinforce(env, policy_net, shared=False):
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    
    metrics = {
        'chaser_returns': [],
        'escaper_returns': [],
        'episode_lengths': [],
        'capture_rates': []
    }
    
    running_captures = 0
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        log_probs = {'chaser': [], 'escaper': []}
        rewards = {'chaser': [], 'escaper': []}
        captures_this_episode = 0
        
        # Always run for full MAX_STEPS
        for t in range(MAX_STEPS):
            # Get action probabilities
            with torch.no_grad():
                chaser_probs = policy_net(state, 'chaser', env)
                escaper_probs = policy_net(state, 'escaper', env)
            
            # Sample actions
            chaser_action = torch.multinomial(chaser_probs, 1).item()
            escaper_action = torch.multinomial(escaper_probs, 1).item()
            
            # Compute log probabilities with fresh forward passes
            chaser_probs = policy_net(state, 'chaser', env)
            escaper_probs = policy_net(state, 'escaper', env)
            log_probs['chaser'].append(torch.log(chaser_probs[chaser_action] + 1e-10))
            log_probs['escaper'].append(torch.log(escaper_probs[escaper_action] + 1e-10))
            
            # Take actions
            next_state, c_reward, e_reward, caught = env.step(chaser_action, escaper_action)
            
            rewards['chaser'].append(c_reward)
            rewards['escaper'].append(e_reward)
            
            if caught:
                captures_this_episode += 1
            
            state = next_state
        
        running_captures += captures_this_episode
        
        # Compute returns
        returns = {'chaser': [], 'escaper': []}
        for agent in ['chaser', 'escaper']:
            G = 0
            for r in reversed(rewards[agent]):
                G = r + GAMMA * G
                returns[agent].insert(0, G)
            returns[agent] = torch.tensor(returns[agent])
            if len(returns[agent]) > 1:
                returns[agent] = (returns[agent] - returns[agent].mean()) / (returns[agent].std() + 1e-8)
        
        # Update policy
        optimizer.zero_grad()
        
        chaser_loss = -torch.stack(log_probs['chaser']) * returns['chaser']
        escaper_loss = -torch.stack(log_probs['escaper']) * returns['escaper']
        
        loss = chaser_loss.mean() + escaper_loss.mean()
        if shared:
            loss = loss * 0.5
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()
        
        # Update metrics
        metrics['chaser_returns'].append(sum(rewards['chaser']))
        metrics['escaper_returns'].append(sum(rewards['escaper']))
        metrics['episode_lengths'].append(MAX_STEPS)
        metrics['capture_rates'].append(running_captures / ((episode + 1) * MAX_STEPS))
        
        if episode % 100 == 0:
            print(f"Episode {episode}")
            print(f"Capture rate: {metrics['capture_rates'][-1]:.3f}")
            print(f"Average episode length: {np.mean(metrics['episode_lengths'][-100:]):.1f}")
            print(f"Captures this episode: {captures_this_episode}")
    
    return metrics

def visualize_interesting_states(policy_net, filename=None):
    """Visualize policy decisions on specific game states and optionally save to file"""
    env = GridWorld()  # Create environment instance
    interesting_states = [
        [0, 0, 1, 1],  # Chaser at corner, Escaper at center
        [1, 1, 0, 0],  # Chaser at center, Escaper at corner
        [0, 0, 2, 2],  # Opposite corners
        [1, 1, 1, 2],  # Adjacent positions
    ]
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']  # Removed STAY
    
    # If filename provided, open file for writing
    if filename:
        f = open(filename, 'w')
        f.write("Policy Network Probabilities\n")
        f.write("===========================\n\n")
    
    for state in interesting_states:
        state_tensor = torch.FloatTensor(np.array(state) / (GRID_SIZE - 1))
        chaser_probs = policy_net(state_tensor, 'chaser', env).detach().numpy()  # Added env parameter
        escaper_probs = policy_net(state_tensor, 'escaper', env).detach().numpy()  # Added env parameter
        
        state_desc = f"\nState: Chaser at ({state[0]}, {state[1]}), Escaper at ({state[2]}, {state[3]})"
        chaser_desc = "Chaser actions: " + ", ".join([f"{name}: {prob:.3f}" for name, prob in zip(action_names, chaser_probs)])
        escaper_desc = "Escaper actions: " + ", ".join([f"{name}: {prob:.3f}" for name, prob in zip(action_names, escaper_probs)])
        
        # Print to console
        print(state_desc)
        print(chaser_desc)
        print(escaper_desc)
        
        # Write to file if filename provided
        if filename:
            f.write(state_desc + "\n")
            f.write(chaser_desc + "\n")
            f.write(escaper_desc + "\n")
            f.write("-" * 50 + "\n")
    
    if filename:
        f.close()

def visualize_all_states(policy_net, filename):
    env = GridWorld()  # Create env instance for valid action checking
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']  # Removed STAY
    
    with open(filename, 'w') as f:
        f.write("Policy Network Probabilities - All States\n")
        f.write("=======================================\n\n")
        
        for chaser_x in range(GRID_SIZE):
            for chaser_y in range(GRID_SIZE):
                for escaper_x in range(GRID_SIZE):
                    for escaper_y in range(GRID_SIZE):
                        state = [chaser_x, chaser_y, escaper_x, escaper_y]
                        state_tensor = torch.FloatTensor(np.array(state) / (GRID_SIZE - 1))
                        
                        # Get probabilities with valid action masking
                        chaser_probs = policy_net(state_tensor, 'chaser', env).detach().numpy()
                        escaper_probs = policy_net(state_tensor, 'escaper', env).detach().numpy()
                        
                        # Get valid action masks
                        chaser_valid = env.get_valid_actions((chaser_x, chaser_y))
                        escaper_valid = env.get_valid_actions((escaper_x, escaper_y))
                        
                        # Add indicator if this is a capture state
                        capture_state = "(CAPTURE STATE)" if chaser_x == escaper_x and chaser_y == escaper_y else ""
                        f.write(f"\nState: Chaser at ({chaser_x}, {chaser_y}), Escaper at ({escaper_x}, {escaper_y}) {capture_state}\n")
                        
                        f.write("\nChaser Actions:\n")
                        f.write("--------------\n")
                        for action, prob, valid in zip(action_names, chaser_probs, chaser_valid):
                            validity = "VALID" if valid else "INVALID"
                            f.write(f"{action:>6}: {prob:.3f} ({validity})\n")
                        
                        f.write("\nEscaper Actions:\n")
                        f.write("--------------\n")
                        for action, prob, valid in zip(action_names, escaper_probs, escaper_valid):
                            validity = "VALID" if valid else "INVALID"
                            f.write(f"{action:>6}: {prob:.3f} ({validity})\n")
                        
                        f.write("\n" + "="*50 + "\n")

def save_probabilities_standard_format(policy_net, filename):
    env = GridWorld()
    
    with open(filename, 'w') as f:
        # Write header
        f.write("X_1,Y_1,X_2,Y_2:U_0,D_0,L_0,R_0,U_1,D_1,L_1,R_1\n")
        
        # Iterate through all possible states
        for chaser_x in range(GRID_SIZE):
            for chaser_y in range(GRID_SIZE):
                for escaper_x in range(GRID_SIZE):
                    for escaper_y in range(GRID_SIZE):
                        state = [chaser_x, chaser_y, escaper_x, escaper_y]
                        state_tensor = torch.FloatTensor(np.array(state) / (GRID_SIZE - 1))
                        
                        # Get probabilities
                        with torch.no_grad():
                            chaser_probs = policy_net(state_tensor, 'chaser', env).detach().numpy()
                            escaper_probs = policy_net(state_tensor, 'escaper', env).detach().numpy()
                        
                        # Format line: state:chaser_probs,escaper_probs
                        probs_str = ','.join([f"{p:.16f}" for p in np.concatenate([chaser_probs, escaper_probs])])
                        state_str = ','.join(map(str, state))
                        f.write(f"{state_str}:{probs_str}\n")

def main():
    env = GridWorld()
    
    # Train separate networks version
    print("\nTraining separate networks...")
    separate_policy = PolicyNetwork(shared=False)
    separate_metrics = train_reinforce(env, separate_policy, shared=False)
    
    # Train shared network version
    print("\nTraining shared network...")
    shared_policy = PolicyNetwork(shared=True)
    shared_metrics = train_reinforce(env, shared_policy, shared=True)
    
    # Save probabilities in different formats
    print("\nSaving probabilities...")
    visualize_all_states(separate_policy, "unsharedProbabilities_all_states.txt")
    visualize_all_states(shared_policy, "sharedNetworkProbabilities_all_states.txt")
    save_probabilities_standard_format(shared_policy, "sharedNetworkProbabilitiesStd.txt")
    
    # Original visualization for interesting states
    print("\nSeparate Networks Policy (Interesting States):")
    visualize_interesting_states(separate_policy, "unsharedProbabilities_interesting.txt")
    
    print("\nShared Network Policy (Interesting States):")
    visualize_interesting_states(shared_policy, "sharedNetworkProbabilities_interesting.txt")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(separate_metrics['capture_rates'], label='Separate')
    plt.plot(shared_metrics['capture_rates'], label='Shared')
    plt.title('Capture Rate')
    plt.xlabel('Episode')
    plt.legend()
    
    plt.subplot(132)
    plt.plot(separate_metrics['episode_lengths'], label='Separate')
    plt.plot(shared_metrics['episode_lengths'], label='Shared')
    plt.title('Episode Length')
    plt.xlabel('Episode')
    plt.legend()
    
    plt.subplot(133)
    plt.plot(separate_metrics['chaser_returns'], label='Chaser')
    plt.plot(separate_metrics['escaper_returns'], label='Escaper')
    plt.title('Returns (Separate Networks)')
    plt.xlabel('Episode')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()