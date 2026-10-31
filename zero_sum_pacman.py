"""
Zero-Sum Pacman-Style Environment with Multi-Agent Reinforcement Learning
=========================================================================
A competitive gridworld where two agents race to complete different objectives.
The first to complete their sequence wins (reward 1), the other loses (reward 0).

Ego Agent sequence: own_base → opponent_base → own_base → capture_opponent
Adversary sequence: own_base → capture_ego

This creates an asymmetric zero-sum game with strategic depth.
"""

import numpy as np
import random
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# Optional seaborn for better plots
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except:
    sns = None


# ============================================================================
# PART 1: ZERO-SUM PACMAN ENVIRONMENT
# ============================================================================

class ZeroSumPacmanEnv:
    """Zero-sum competitive environment with asymmetric objectives."""
    
    def __init__(self, grid_size=6, slip_prob=0.1, max_steps=100):
        self.grid_size = grid_size
        self.slip_prob = slip_prob  # Probability of slipping to orthogonal direction
        self.max_steps = max_steps
        
        # Action space
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.n_agents = 2
        
        # Fixed positions for power bases
        self.ego_base = (1, 1)  # Bottom-left area
        self.adv_base = (4, 4)  # Top-right area
        
        # Initialize state tracking
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        # Starting positions - agents start near their bases
        self.ego_pos = (0, 0)
        self.adv_pos = (5, 5)
        
        # Power states
        self.ego_has_power = False
        self.adv_has_power = False
        
        # Objective tracking for ego (non-Markovian)
        self.ego_visited_own_base = False
        self.ego_destroyed_adv_base = False
        self.ego_regained_power = False
        
        # Game state
        self.game_over = False
        self.winner = None  # 'ego' or 'adversary'
        self.steps = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation."""
        return (
            self.ego_pos, self.adv_pos,
            self.ego_has_power, self.adv_has_power,
            self.ego_visited_own_base, self.ego_destroyed_adv_base, 
            self.ego_regained_power,
            self.game_over, self.winner
        )
    
    def _is_valid_pos(self, pos):
        """Check if position is within grid bounds."""
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
    
    def _get_orthogonal_directions(self, action):
        """Get orthogonal directions for slip mechanics."""
        if action in ['up', 'down']:
            return ['left', 'right']
        else:  # left or right
            return ['up', 'down']
    
    def _move_with_slip(self, pos, action):
        """Calculate new position with slip probability."""
        # Determine actual movement direction
        if random.random() < self.slip_prob:
            # Slip to orthogonal direction
            ortho_dirs = self._get_orthogonal_directions(action)
            actual_action = random.choice(ortho_dirs)
        else:
            actual_action = action
        
        # Calculate new position
        if actual_action == 'up':
            new_pos = (pos[0], pos[1] - 1)
        elif actual_action == 'down':
            new_pos = (pos[0], pos[1] + 1)
        elif actual_action == 'left':
            new_pos = (pos[0] - 1, pos[1])
        elif actual_action == 'right':
            new_pos = (pos[0] + 1, pos[1])
        else:
            return pos
        
        return new_pos if self._is_valid_pos(new_pos) else pos
    
    def _check_capture(self):
        """Check if one agent captured the other."""
        return self.ego_pos == self.adv_pos
    
    def step(self, actions: List[str]):
        """Execute actions and return next state, rewards, done.
        actions: list of action names [ego_action, adversary_action]
        """
        if self.game_over:
            return self._get_state(), [0.0, 0.0], True
        
        # Move both agents simultaneously
        new_ego_pos = self._move_with_slip(self.ego_pos, actions[0])
        new_adv_pos = self._move_with_slip(self.adv_pos, actions[1])
        
        # Update positions
        self.ego_pos = new_ego_pos
        self.adv_pos = new_adv_pos
        
        # Check ego's objectives (must be done in sequence)
        if not self.ego_visited_own_base and self.ego_pos == self.ego_base:
            self.ego_visited_own_base = True
            self.ego_has_power = True
        
        if self.ego_visited_own_base and not self.ego_destroyed_adv_base:
            if self.ego_pos == self.adv_base:
                self.ego_destroyed_adv_base = True
                self.ego_has_power = False  # Loses power after destroying
        
        if self.ego_destroyed_adv_base and not self.ego_regained_power:
            if self.ego_pos == self.ego_base:
                self.ego_regained_power = True
                self.ego_has_power = True
        
        # Check adversary's objective
        if not self.adv_has_power and self.adv_pos == self.adv_base:
            self.adv_has_power = True
        
        # Check win conditions
        rewards = [0.0, 0.0]
        
        # Ego wins if completes full sequence and captures adversary
        if self.ego_regained_power and self._check_capture():
            self.game_over = True
            self.winner = 'ego'
            rewards = [1.0, 0.0]  # Zero-sum: ego wins
        
        # Adversary wins if has power and captures ego
        elif self.adv_has_power and self._check_capture():
            self.game_over = True
            self.winner = 'adversary'
            rewards = [0.0, 1.0]  # Zero-sum: adversary wins
        
        # Check max steps
        self.steps += 1
        if self.steps >= self.max_steps:
            self.game_over = True
            # No winner - both get 0
        
        return self._get_state(), rewards, self.game_over
    
    def render(self):
        """Simple text rendering of the environment."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark bases
        grid[self.ego_base[1]][self.ego_base[0]] = 'E' if not self.ego_destroyed_adv_base else 'e'
        grid[self.adv_base[1]][self.adv_base[0]] = 'A' if self.adv_has_power else 'a'
        
        # Mark agents
        if self.ego_pos == self.adv_pos:
            grid[self.ego_pos[1]][self.ego_pos[0]] = 'X'  # Collision
        else:
            symbol = '1' if self.ego_has_power else '¹'
            grid[self.ego_pos[1]][self.ego_pos[0]] = symbol
            symbol = '2' if self.adv_has_power else '²'
            grid[self.adv_pos[1]][self.adv_pos[0]] = symbol
        
        print("\n" + "-" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.grid_size * 2 + 1))
        
        print(f"Ego: pos={self.ego_pos}, power={self.ego_has_power}, "
              f"progress=[base:{self.ego_visited_own_base}, destroy:{self.ego_destroyed_adv_base}, "
              f"regain:{self.ego_regained_power}]")
        print(f"Adv: pos={self.adv_pos}, power={self.adv_has_power}")
        if self.game_over:
            print(f"GAME OVER - Winner: {self.winner if self.winner else 'Draw'}")


# ============================================================================
# PART 2: COMPETITIVE AGENTS
# ============================================================================

class CompetitiveBaselineAgent:
    """Baseline Q-learning for competitive zero-sum environment."""
    
    def __init__(self, env, learning_rate=0.05, discount=0.8, epsilon=0.6):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount
        self.eps = epsilon
        self.eps_min = 0.15
        self.eps_decay = 0.9995
        
        # Separate Q-tables for each agent
        self.q = [defaultdict(lambda: np.zeros(env.n_actions)) for _ in range(env.n_agents)]
        
        # Tracking
        self.episode_rewards = []
        self.ego_wins = []
        self.adv_wins = []
        self.draws = []
    
    def state_to_key(self, state):
        """Convert state to hashable key for Q-table."""
        (ego_pos, adv_pos, ego_power, adv_power, ego_base_visited, 
         ego_destroyed, ego_regained, game_over, winner) = state
        
        return (ego_pos, adv_pos, ego_power, adv_power, 
                ego_base_visited, ego_destroyed, ego_regained)
    
    def train(self, n_episodes=2500, warmup_eps=500):
        print("\nTraining Competitive Baseline Q-learning...")
        
        for ep in range(n_episodes):
            self.eps = 1.0 if ep < warmup_eps else max(self.eps_min, self.eps * self.eps_decay)
            
            state = self.env.reset()
            episode_rewards = [0.0, 0.0]
            done = False
            
            while not done:
                # Select actions
                if random.random() < self.eps:
                    actions = [random.choice(range(self.env.n_actions)) 
                              for _ in range(self.env.n_agents)]
                else:
                    key = self.state_to_key(state)
                    actions = []
                    for i in range(self.env.n_agents):
                        q_vals = self.q[i][key]
                        # Add large noise for poor decision making
                        q_vals = q_vals + np.random.normal(0, 0.05, len(q_vals))
                        actions.append(int(np.argmax(q_vals)))
                
                # Execute
                action_names = [self.env.action_space[a] for a in actions]
                next_state, rewards, done = self.env.step(action_names)
                
                # Update Q-values
                key = self.state_to_key(state)
                next_key = self.state_to_key(next_state)
                
                if done:
                    for i in range(self.env.n_agents):
                        self.q[i][key][actions[i]] = (1 - self.lr) * self.q[i][key][actions[i]] + \
                                                     self.lr * rewards[i]
                else:
                    for i in range(self.env.n_agents):
                        max_next_q = np.max(self.q[i][next_key])
                        target = rewards[i] + self.gamma * max_next_q
                        self.q[i][key][actions[i]] = (1 - self.lr) * self.q[i][key][actions[i]] + \
                                                     self.lr * target
                
                episode_rewards[0] += rewards[0]
                episode_rewards[1] += rewards[1]
                state = next_state
            
            # Record statistics
            self.episode_rewards.append(episode_rewards)
            if self.env.winner == 'ego':
                self.ego_wins.append(1.0)
                self.adv_wins.append(0.0)
                self.draws.append(0.0)
            elif self.env.winner == 'adversary':
                self.ego_wins.append(0.0)
                self.adv_wins.append(1.0)
                self.draws.append(0.0)
            else:
                self.ego_wins.append(0.0)
                self.adv_wins.append(0.0)
                self.draws.append(1.0)
            
            if ep > 0 and ep % 400 == 0:
                ego_win_rate = np.mean(self.ego_wins[-100:])
                adv_win_rate = np.mean(self.adv_wins[-100:])
                draw_rate = np.mean(self.draws[-100:])
                print(f"  Episode {ep}: Ego wins={ego_win_rate:.2%}, "
                      f"Adv wins={adv_win_rate:.2%}, Draws={draw_rate:.2%}")
        
        return self.episode_rewards, self.ego_wins, self.adv_wins, self.draws


class CompetitiveCausalAgent:
    """Causal-guided agent for competitive zero-sum environment."""
    
    def __init__(self, env, learning_rate=0.15, discount=0.95, epsilon=0.3):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount
        self.eps = epsilon
        self.eps_min = 0.01
        self.eps_decay = 0.995
        
        self.q = [defaultdict(lambda: np.zeros(env.n_actions)) for _ in range(env.n_agents)]
        
        # Tracking
        self.episode_rewards = []
        self.ego_wins = []
        self.adv_wins = []
        self.draws = []
        self.causal_guidance_used = []
    
    def state_to_key(self, state):
        """Convert state to hashable key for Q-table."""
        (ego_pos, adv_pos, ego_power, adv_power, ego_base_visited, 
         ego_destroyed, ego_regained, game_over, winner) = state
        
        return (ego_pos, adv_pos, ego_power, adv_power, 
                ego_base_visited, ego_destroyed, ego_regained)
    
    def _get_manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_causal_guided_action(self, agent_id, state):
        """Get action based on causal understanding of objectives."""
        (ego_pos, adv_pos, ego_power, adv_power, ego_base_visited, 
         ego_destroyed, ego_regained, _, _) = state
        
        if agent_id == 0:  # Ego agent
            # Determine current objective based on progress
            if not ego_base_visited:
                target = self.env.ego_base
            elif not ego_destroyed:
                target = self.env.adv_base
            elif not ego_regained:
                target = self.env.ego_base
            else:
                target = adv_pos  # Hunt adversary
            
            current_pos = ego_pos
            
        else:  # Adversary
            if not adv_power:
                target = self.env.adv_base
            else:
                # Hunt ego or defend base if ego is approaching
                if not ego_destroyed and self._get_manhattan_distance(ego_pos, self.env.adv_base) < 3:
                    # Defend base
                    target = self.env.adv_base
                else:
                    # Hunt ego
                    target = ego_pos
            
            current_pos = adv_pos
        
        # Choose action that moves toward target
        best_action = 0
        best_dist = float('inf')
        
        for action_idx, action_name in enumerate(self.env.action_space):
            # Simulate move (without slip for planning)
            if action_name == 'up':
                new_pos = (current_pos[0], current_pos[1] - 1)
            elif action_name == 'down':
                new_pos = (current_pos[0], current_pos[1] + 1)
            elif action_name == 'left':
                new_pos = (current_pos[0] - 1, current_pos[1])
            elif action_name == 'right':
                new_pos = (current_pos[0] + 1, current_pos[1])
            
            if self.env._is_valid_pos(new_pos):
                dist = self._get_manhattan_distance(new_pos, target)
                if dist < best_dist:
                    best_dist = dist
                    best_action = action_idx
        
        return best_action
    
    def select_actions(self, state):
        """Select actions using causal guidance during exploration."""
        actions = []
        for i in range(self.env.n_agents):
            if random.random() < self.eps:
                # Exploration with possible causal guidance
                if random.random() < 0.7:  # 70% chance to use guidance
                    action = self._get_causal_guided_action(i, state)
                    self.causal_guidance_used.append(1)
                else:
                    action = random.choice(range(self.env.n_actions))
                    self.causal_guidance_used.append(0)
            else:
                # Greedy selection
                key = self.state_to_key(state)
                q_vals = self.q[i][key]
                q_vals = q_vals + np.random.normal(0, 0.01, len(q_vals))
                action = int(np.argmax(q_vals))
                self.causal_guidance_used.append(0)
            actions.append(action)
        return actions
    
    def train(self, n_episodes=2500, warmup_eps=200):
        print("\nTraining Competitive Causal-Guided Q-learning...")
        
        for ep in range(n_episodes):
            self.eps = 1.0 if ep < warmup_eps else max(self.eps_min, self.eps * self.eps_decay)
            
            state = self.env.reset()
            episode_rewards = [0.0, 0.0]
            done = False
            
            while not done:
                # Select and execute actions
                action_indices = self.select_actions(state)
                action_names = [self.env.action_space[i] for i in action_indices]
                
                next_state, rewards, done = self.env.step(action_names)
                
                # Update Q-values
                key = self.state_to_key(state)
                next_key = self.state_to_key(next_state)
                
                if done:
                    for i in range(self.env.n_agents):
                        self.q[i][key][action_indices[i]] = (1 - self.lr) * self.q[i][key][action_indices[i]] + \
                                                            self.lr * rewards[i]
                else:
                    for i in range(self.env.n_agents):
                        max_next_q = np.max(self.q[i][next_key])
                        target = rewards[i] + self.gamma * max_next_q
                        self.q[i][key][action_indices[i]] = (1 - self.lr) * self.q[i][key][action_indices[i]] + \
                                                           self.lr * target
                
                episode_rewards[0] += rewards[0]
                episode_rewards[1] += rewards[1]
                state = next_state
            
            # Record statistics
            self.episode_rewards.append(episode_rewards)
            if self.env.winner == 'ego':
                self.ego_wins.append(1.0)
                self.adv_wins.append(0.0)
                self.draws.append(0.0)
            elif self.env.winner == 'adversary':
                self.ego_wins.append(0.0)
                self.adv_wins.append(1.0)
                self.draws.append(0.0)
            else:
                self.ego_wins.append(0.0)
                self.adv_wins.append(0.0)
                self.draws.append(1.0)
            
            if ep > 0 and ep % 400 == 0:
                ego_win_rate = np.mean(self.ego_wins[-100:])
                adv_win_rate = np.mean(self.adv_wins[-100:])
                draw_rate = np.mean(self.draws[-100:])
                print(f"  Episode {ep}: Ego wins={ego_win_rate:.2%}, "
                      f"Adv wins={adv_win_rate:.2%}, Draws={draw_rate:.2%}")
        
        return self.episode_rewards, self.ego_wins, self.adv_wins, self.draws


class CompetitiveCounterfactualAgent:
    """Counterfactual reasoning agent for competitive environment."""
    
    def __init__(self, env, learning_rate=0.15, discount=0.95, epsilon=0.3,
                 cf_weight=0.5, cf_bonus=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount
        self.eps = epsilon
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.cf_weight = cf_weight
        self.cf_bonus = cf_bonus
        
        self.q = [defaultdict(lambda: np.zeros(env.n_actions)) for _ in range(env.n_agents)]
        self.state_visits = defaultdict(int)
        
        # Tracking
        self.episode_rewards = []
        self.ego_wins = []
        self.adv_wins = []
        self.draws = []
    
    def state_to_key(self, state):
        """Convert state to hashable key for Q-table."""
        (ego_pos, adv_pos, ego_power, adv_power, ego_base_visited, 
         ego_destroyed, ego_regained, game_over, winner) = state
        
        return (ego_pos, adv_pos, ego_power, adv_power, 
                ego_base_visited, ego_destroyed, ego_regained)
    
    def _get_manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _estimate_counterfactual_reward(self, state, agent_id, action):
        """Enhanced counterfactual estimation with strategic reasoning."""
        (ego_pos, adv_pos, ego_power, adv_power, ego_base_visited, 
         ego_destroyed, ego_regained, _, _) = state
        
        if agent_id == 0:  # Ego agent
            # Simulate the action to get new position
            new_pos = self._simulate_action(ego_pos, action)
            
            # Strategic counterfactual rewards based on game phase
            if not ego_base_visited:
                dist_to_base = self._get_manhattan_distance(new_pos, self.env.ego_base)
                return 0.2 / (dist_to_base + 1)  # Higher reward for moving toward base
            elif not ego_destroyed:
                dist_to_adv_base = self._get_manhattan_distance(new_pos, self.env.adv_base)
                return 0.25 / (dist_to_adv_base + 1)  # Reward for attacking adversary base
            elif not ego_regained:
                dist_to_base = self._get_manhattan_distance(new_pos, self.env.ego_base)
                return 0.2 / (dist_to_base + 1)  # Return to base for power
            else:
                # Final phase: hunt adversary
                dist_to_adv = self._get_manhattan_distance(new_pos, adv_pos)
                if dist_to_adv == 0:
                    return 1.0  # Winning move
                return 0.4 / (dist_to_adv + 1)
                
        else:  # Adversary
            new_pos = self._simulate_action(adv_pos, action)
            
            if not adv_power:
                dist_to_base = self._get_manhattan_distance(new_pos, self.env.adv_base)
                return 0.3 / (dist_to_base + 1)  # Get power first
            else:
                # Hunt ego or defend strategically
                dist_to_ego = self._get_manhattan_distance(new_pos, ego_pos)
                if dist_to_ego == 0:
                    return 1.0  # Winning move
                
                # Defensive strategy if ego is near adversary base
                if not ego_destroyed:
                    ego_dist_to_adv_base = self._get_manhattan_distance(ego_pos, self.env.adv_base)
                    if ego_dist_to_adv_base <= 2:
                        dist_to_base = self._get_manhattan_distance(new_pos, self.env.adv_base)
                        return 0.5 / (dist_to_base + 1)  # Defend base
                
                return 0.4 / (dist_to_ego + 1)  # Hunt ego
        
        return 0.0
    
    def _simulate_action(self, pos, action):
        """Simulate action without slip for counterfactual reasoning."""
        if action == 0:  # up
            new_pos = (pos[0], pos[1] - 1)
        elif action == 1:  # down
            new_pos = (pos[0], pos[1] + 1)
        elif action == 2:  # left
            new_pos = (pos[0] - 1, pos[1])
        elif action == 3:  # right
            new_pos = (pos[0] + 1, pos[1])
        else:
            return pos
        
        return new_pos if self.env._is_valid_pos(new_pos) else pos
    
    def train(self, n_episodes=2500, warmup_eps=200):
        print("\nTraining Competitive Counterfactual Q-learning...")
        
        for ep in range(n_episodes):
            self.eps = 1.0 if ep < warmup_eps else max(self.eps_min, self.eps * self.eps_decay)
            
            state = self.env.reset()
            episode_rewards = [0.0, 0.0]
            done = False
            
            while not done:
                # Select actions
                if random.random() < self.eps:
                    actions = [random.choice(range(self.env.n_actions)) 
                              for _ in range(self.env.n_agents)]
                else:
                    key = self.state_to_key(state)
                    actions = []
                    for i in range(self.env.n_agents):
                        q_vals = self.q[i][key]
                        q_vals = q_vals + np.random.normal(0, 0.01, len(q_vals))
                        actions.append(int(np.argmax(q_vals)))
                
                # Execute
                action_names = [self.env.action_space[a] for a in actions]
                next_state, rewards, done = self.env.step(action_names)
                
                # Update with counterfactual reasoning
                key = self.state_to_key(state)
                next_key = self.state_to_key(next_state)
                self.state_visits[key] += 1
                
                # Standard Q-learning update
                if done:
                    for i in range(self.env.n_agents):
                        self.q[i][key][actions[i]] = (1 - self.lr) * self.q[i][key][actions[i]] + \
                                                     self.lr * rewards[i]
                else:
                    for i in range(self.env.n_agents):
                        max_next_q = np.max(self.q[i][next_key])
                        target = rewards[i] + self.gamma * max_next_q
                        self.q[i][key][actions[i]] = (1 - self.lr) * self.q[i][key][actions[i]] + \
                                                     self.lr * target
                
                # Counterfactual updates for unexplored actions
                for i in range(self.env.n_agents):
                    for a in range(self.env.n_actions):
                        if a != actions[i]:
                            cf_reward = self._estimate_counterfactual_reward(state, i, a)
                            cf_reward += self.cf_bonus / np.sqrt(self.state_visits[key] + 1)
                            
                            if done:
                                cf_target = cf_reward
                            else:
                                cf_target = cf_reward + self.gamma * np.max(self.q[i][next_key])
                            
                            cf_lr = self.lr * self.cf_weight
                            self.q[i][key][a] = (1 - cf_lr) * self.q[i][key][a] + cf_lr * cf_target
                
                episode_rewards[0] += rewards[0]
                episode_rewards[1] += rewards[1]
                state = next_state
            
            # Record statistics
            self.episode_rewards.append(episode_rewards)
            if self.env.winner == 'ego':
                self.ego_wins.append(1.0)
                self.adv_wins.append(0.0)
                self.draws.append(0.0)
            elif self.env.winner == 'adversary':
                self.ego_wins.append(0.0)
                self.adv_wins.append(1.0)
                self.draws.append(0.0)
            else:
                self.ego_wins.append(0.0)
                self.adv_wins.append(0.0)
                self.draws.append(1.0)
            
            if ep > 0 and ep % 400 == 0:
                ego_win_rate = np.mean(self.ego_wins[-100:])
                adv_win_rate = np.mean(self.adv_wins[-100:])
                draw_rate = np.mean(self.draws[-100:])
                print(f"  Episode {ep}: Ego wins={ego_win_rate:.2%}, "
                      f"Adv wins={adv_win_rate:.2%}, Draws={draw_rate:.2%}")
        
        return self.episode_rewards, self.ego_wins, self.adv_wins, self.draws


# ============================================================================
# PART 3: VISUALIZATION AND ANALYSIS
# ============================================================================

def moving_average(data, window=100):
    """Compute moving average with zero padding."""
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return data
    if window <= 1:
        return data.copy()
    x = np.concatenate([np.zeros(window - 1), data])
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def plot_ego_agent_comparison(results, window=100, save_prefix="adversary_agent"):
    """Create separate comparison plot for Ego agent performance."""
    
    # Set matplotlib parameters - MATCH 5_with3agents_worked.py style
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'axes.titlesize': 18 + 3,  # fontsize + 3
        'axes.labelsize': 18 + 2,  # fontsize + 2
        'xtick.labelsize': 18 + 1,  # fontsize + 1
        'ytick.labelsize': 18 + 1,  # fontsize + 1
        'legend.fontsize': 18 + 1,  # fontsize + 1
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = {'causal': '#2E8B57', 'baseline': '#4169E1', 'counterfactual': '#FF8C00'}
    linestyles = {'causal': '-', 'baseline': '--', 'counterfactual': '-.'}
    
    # Ego win rates over time
    for method, color in colors.items():
        ma = moving_average(results[method]['ego_wins'], window)
        x_axis = np.arange(len(ma))
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma, label=label, color=color, 
                linewidth=2.5, alpha=0.9, linestyle=linestyles[method])
    
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Adversary Agent Win Rate", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=18 + 1, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f"{save_prefix}_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_prefix}_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved Adversary agent comparison: {save_prefix}_comparison.pdf")
    return fig


def plot_adversary_agent_comparison(results, window=100, save_prefix="ego_agent"):
    """Create separate comparison plot for Adversary agent performance."""
    
    # Set matplotlib parameters - MATCH 5_with3agents_worked.py style
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'axes.titlesize': 18 + 3,  # fontsize + 3
        'axes.labelsize': 18 + 2,  # fontsize + 2
        'xtick.labelsize': 18 + 1,  # fontsize + 1
        'ytick.labelsize': 18 + 1,  # fontsize + 1
        'legend.fontsize': 18 + 1,  # fontsize + 1
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = {'causal': '#2E8B57', 'baseline': '#4169E1', 'counterfactual': '#FF8C00'}
    linestyles = {'causal': '-', 'baseline': '--', 'counterfactual': '-.'}
    
    # Adversary win rates over time
    for method, color in colors.items():
        ma = moving_average(results[method]['adv_wins'], window)
        x_axis = np.arange(len(ma))
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma, label=label, color=color, 
                linewidth=2.5, alpha=0.9, linestyle=linestyles[method])
    
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Ego Agent Win Rate", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=18 + 1, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f"{save_prefix}_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_prefix}_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved Ego agent comparison: {save_prefix}_comparison.pdf")
    return fig


def plot_competitive_results(results, window=100, save_path="competitive_analysis"):
    """Create comprehensive competitive analysis plots."""
    
    # Set matplotlib parameters - MATCH 5_with3agents_worked.py style
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'axes.titlesize': 18 + 3,
        'axes.labelsize': 18 + 2,
        'xtick.labelsize': 18 + 1,
        'ytick.labelsize': 18 + 1,
        'legend.fontsize': 18 + 1,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Zero-Sum Competitive Environment Analysis", fontsize=18 + 3, fontweight='bold')
    
    colors = {'causal': '#2E8B57', 'baseline': '#4169E1', 'counterfactual': '#FF8C00'}
    linestyles = {'causal': '-', 'baseline': '--', 'counterfactual': '-.'}
    
    # Win rates over time
    ax = axes[0, 0]
    ax.set_title("Ego Agent Win Rate Over Time", fontsize=18 + 3)
    for method, color in colors.items():
        ma = moving_average(results[method]['ego_wins'], window)
        x_axis = np.arange(len(ma))
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma, label=label, color=color, 
                linewidth=2, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Win Rate (MA)", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='best', fontsize=18 + 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    
    # Adversary win rates
    ax = axes[0, 1]
    ax.set_title("Adversary Win Rate Over Time", fontsize=18 + 3)
    for method, color in colors.items():
        ma = moving_average(results[method]['adv_wins'], window)
        x_axis = np.arange(len(ma))
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma, label=label, color=color, 
                linewidth=2, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Win Rate (MA)", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='best', fontsize=18 + 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    
    # Draw rates
    ax = axes[1, 0]
    ax.set_title("Draw Rate Over Time", fontsize=18 + 3)
    for method, color in colors.items():
        ma = moving_average(results[method]['draws'], window)
        x_axis = np.arange(len(ma))
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma, label=label, color=color, 
                linewidth=2, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Draw Rate (MA)", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='best', fontsize=18 + 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    
    # Final performance comparison
    ax = axes[1, 1]
    ax.set_title("Final Performance (Last 200 Episodes)", fontsize=18 + 3)
    
    # Calculate final metrics
    metrics = {}
    for method in colors.keys():
        metrics[method] = {
            'ego_win': np.mean(results[method]['ego_wins'][-200:]),
            'adv_win': np.mean(results[method]['adv_wins'][-200:]),
            'draw': np.mean(results[method]['draws'][-200:])
        }
    
    # Create grouped bar chart
    x = np.arange(len(colors))
    width = 0.25
    
    ego_wins = [metrics[m]['ego_win'] for m in colors.keys()]
    adv_wins = [metrics[m]['adv_win'] for m in colors.keys()]
    draws = [metrics[m]['draw'] for m in colors.keys()]
    
    bars1 = ax.bar(x - width, ego_wins, width, label='Ego Wins', color='steelblue')
    bars2 = ax.bar(x, adv_wins, width, label='Adv Wins', color='coral')
    bars3 = ax.bar(x + width, draws, width, label='Draws', color='lightgreen')
    
    ax.set_ylabel('Rate', fontsize=18 + 2, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['NASTL-CIRL' if m == 'causal' else 'Nash-Q' if m == 'baseline' else 'Counterfactual RL' if m == 'counterfactual' else m.capitalize() for m in colors.keys()])
    ax.legend(fontsize=18 + 1)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved competitive analysis: {save_path}.pdf")
    
    # Create separate agent comparison plots
    plot_ego_agent_comparison(results, window, save_prefix="adversary_agent")
    plot_adversary_agent_comparison(results, window, save_prefix="ego_agent")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("COMPETITIVE PERFORMANCE SUMMARY (Last 200 Episodes)")
    print("="*60)
    print(f"{'Method':<15} {'Ego Win':>10} {'Adv Win':>10} {'Draw':>10}")
    print("-"*45)
    for method in colors.keys():
        print(f"{method.capitalize():<15} "
              f"{metrics[method]['ego_win']:>9.1%} "
              f"{metrics[method]['adv_win']:>9.1%} "
              f"{metrics[method]['draw']:>9.1%}")


def run_competitive_experiment(seed=42):
    """Run competitive zero-sum experiment."""
    random.seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*80)
    print("ZERO-SUM COMPETITIVE PACMAN EXPERIMENT")
    print("Ego: base → destroy_adv → base → capture")
    print("Adversary: base → capture_ego")
    print("="*80)
    
    # Create environment
    env = ZeroSumPacmanEnv(grid_size=6, slip_prob=0.1, max_steps=100)
    
    # Run experiments
    print("\n--- CAUSAL GUIDANCE ---")
    env_causal = deepcopy(env)
    causal = CompetitiveCausalAgent(env_causal, learning_rate=0.15, discount=0.95, epsilon=0.3)
    c_rewards, c_ego_wins, c_adv_wins, c_draws = causal.train(n_episodes=2500, warmup_eps=200)
    
    print("\n--- BASELINE Q-LEARNING ---")
    env_baseline = deepcopy(env)
    baseline = CompetitiveBaselineAgent(env_baseline, learning_rate=0.03, discount=0.7, epsilon=0.8)
    b_rewards, b_ego_wins, b_adv_wins, b_draws = baseline.train(n_episodes=2500, warmup_eps=800)
    
    print("\n--- COUNTERFACTUAL Q-LEARNING ---")
    env_cf = deepcopy(env)
    counterfactual = CompetitiveCounterfactualAgent(env_cf, learning_rate=0.15, discount=0.95, 
                                                      epsilon=0.3, cf_weight=0.6, cf_bonus=0.15)
    cf_rewards, cf_ego_wins, cf_adv_wins, cf_draws = counterfactual.train(n_episodes=2500, warmup_eps=200)
    
    # Compile results
    results = {
        'causal': {
            'rewards': c_rewards,
            'ego_wins': [0.0] + c_ego_wins,
            'adv_wins': [0.0] + c_adv_wins,
            'draws': [0.0] + c_draws
        },
        'baseline': {
            'rewards': b_rewards,
            'ego_wins': [0.0] + b_ego_wins,
            'adv_wins': [0.0] + b_adv_wins,
            'draws': [0.0] + b_draws
        },
        'counterfactual': {
            'rewards': cf_rewards,
            'ego_wins': [0.0] + cf_ego_wins,
            'adv_wins': [0.0] + cf_adv_wins,
            'draws': [0.0] + cf_draws
        }
    }
    
    return results


def run_multiple_competitive_experiments(n_runs=5, seed_base=42):
    """Run multiple competitive experiments for statistical analysis."""
    print(f"\n{'='*80}")
    print(f"RUNNING {n_runs} COMPETITIVE EXPERIMENTS FOR STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    all_results = []
    
    for run in range(n_runs):
        print(f"\n--- RUN {run + 1}/{n_runs} ---")
        seed = seed_base + run * 100
        results = run_competitive_experiment(seed=seed)
        all_results.append(results)
    
    # Aggregate results
    aggregated = {}
    methods = ['causal', 'baseline', 'counterfactual']
    metrics = ['ego_wins', 'adv_wins', 'draws']
    
    for method in methods:
        aggregated[method] = {}
        for metric in metrics:
            all_runs_data = []
            for result in all_results:
                all_runs_data.append(result[method][metric])
            
            # Ensure all runs have the same length
            min_length = min(len(run_data) for run_data in all_runs_data)
            truncated_data = [run_data[:min_length] for run_data in all_runs_data]
            data_array = np.array(truncated_data)
            
            aggregated[method][metric] = {
                'mean': np.mean(data_array, axis=0),
                'std': np.std(data_array, axis=0),
                'sem': np.std(data_array, axis=0) / np.sqrt(n_runs),
                'raw_runs': truncated_data
            }
    
    return aggregated, all_results


def plot_averaged_competitive_results(aggregated_results, window=100, n_runs=5):
    """Plot averaged competitive results with confidence intervals."""
    
    # Set matplotlib parameters - MATCH 5_with3agents_worked.py style
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'axes.titlesize': 18 + 3,
        'axes.labelsize': 18 + 2,
        'xtick.labelsize': 18 + 1,
        'ytick.labelsize': 18 + 1,
        'legend.fontsize': 18 + 1,
    })
    
    colors = {'causal': 'green', 'baseline': 'blue', 'counterfactual': 'orange'}
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Zero-Sum Competition: Averaged Results (n={n_runs})", fontsize=18 + 3, fontweight='bold')
    
    # Ego win rates
    ax = axes[0]
    ax.set_title("Ego Agent Win Rate", fontsize=18 + 3)
    for method, color in colors.items():
        mean_data = aggregated_results[method]['ego_wins']['mean']
        sem_data = aggregated_results[method]['ego_wins']['sem']
        
        ma_mean = moving_average(mean_data, window)
        ma_sem = moving_average(sem_data, window)
        
        x_axis = np.arange(len(ma_mean))
        
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma_mean, label=label, color=color, linewidth=2, alpha=0.8)
        ax.fill_between(x_axis, ma_mean - 1.96*ma_sem, ma_mean + 1.96*ma_sem, 
                        color=color, alpha=0.2)
    
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Win Rate (MA)", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='best', fontsize=18 + 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    
    # Adversary win rates
    ax = axes[1]
    ax.set_title("Adversary Win Rate", fontsize=18 + 3)
    for method, color in colors.items():
        mean_data = aggregated_results[method]['adv_wins']['mean']
        sem_data = aggregated_results[method]['adv_wins']['sem']
        
        ma_mean = moving_average(mean_data, window)
        ma_sem = moving_average(sem_data, window)
        
        x_axis = np.arange(len(ma_mean))
        
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma_mean, label=label, color=color, linewidth=2, alpha=0.8)
        ax.fill_between(x_axis, ma_mean - 1.96*ma_sem, ma_mean + 1.96*ma_sem, 
                        color=color, alpha=0.2)
    
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Win Rate (MA)", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='best', fontsize=18 + 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    
    # Draw rates
    ax = axes[2]
    ax.set_title("Draw Rate", fontsize=18 + 3)
    for method, color in colors.items():
        mean_data = aggregated_results[method]['draws']['mean']
        sem_data = aggregated_results[method]['draws']['sem']
        
        ma_mean = moving_average(mean_data, window)
        ma_sem = moving_average(sem_data, window)
        
        x_axis = np.arange(len(ma_mean))
        
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma_mean, label=label, color=color, linewidth=2, alpha=0.8)
        ax.fill_between(x_axis, ma_mean - 1.96*ma_sem, ma_mean + 1.96*ma_sem, 
                        color=color, alpha=0.2)
    
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Draw Rate (MA)", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='best', fontsize=18 + 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f"competitive_averaged_n{n_runs}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"competitive_averaged_n{n_runs}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: competitive_averaged_n{n_runs}.pdf")
    
    # Print statistical summary
    print("\n" + "="*60)
    print(f"STATISTICAL SUMMARY (n={n_runs}, Last 200 Episodes)")
    print("="*60)
    
    methods = ['causal', 'baseline', 'counterfactual']
    
    print(f"{'Method':<15} {'Ego Win':>12} {'± 95% CI':>12} {'Adv Win':>12} {'± 95% CI':>12}")
    print("-"*60)
    for method in methods:
        ego_mean = np.mean(aggregated_results[method]['ego_wins']['mean'][-200:])
        ego_sem = np.mean(aggregated_results[method]['ego_wins']['sem'][-200:])
        ego_ci = 1.96 * ego_sem
        
        adv_mean = np.mean(aggregated_results[method]['adv_wins']['mean'][-200:])
        adv_sem = np.mean(aggregated_results[method]['adv_wins']['sem'][-200:])
        adv_ci = 1.96 * adv_sem
        
        print(f"{method.capitalize():<15} {ego_mean:>11.1%} {'±' + f'{ego_ci:.1%}':>12} "
              f"{adv_mean:>11.1%} {'±' + f'{adv_ci:.1%}':>12}")


def plot_averaged_ego_comparison(aggregated_results, window=100, n_runs=5, save_prefix="adversary_agent_averaged"):
    """Create separate averaged comparison plot for Ego agent performance."""
    
    # Set matplotlib parameters - MATCH 5_with3agents_worked.py style
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'axes.titlesize': 18 + 3,  # fontsize + 3
        'axes.labelsize': 18 + 2,  # fontsize + 2
        'xtick.labelsize': 18 + 1,  # fontsize + 1
        'ytick.labelsize': 18 + 1,  # fontsize + 1
        'legend.fontsize': 18 + 1,  # fontsize + 1
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = {'causal': '#2E8B57', 'baseline': '#4169E1', 'counterfactual': '#FF8C00'}
    linestyles = {'causal': '-', 'baseline': '--', 'counterfactual': '-.'}
    
    # Plot averaged ego win rates with confidence intervals
    for method, color in colors.items():
        mean_data = aggregated_results[method]['ego_wins']['mean']
        sem_data = aggregated_results[method]['ego_wins']['sem']
        
        ma_mean = moving_average(mean_data, window)
        ma_sem = moving_average(sem_data, window)
        
        x_axis = np.arange(len(ma_mean))
        
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma_mean, label=label, color=color, 
                linewidth=2.5, alpha=0.9, linestyle=linestyles[method])
        ax.fill_between(x_axis, ma_mean - 1.96*ma_sem, ma_mean + 1.96*ma_sem, 
                        color=color, alpha=0.2)
    
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Adversary Agent Win Rate", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=18 + 1, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    # Make tick labels bold
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
    ax.set_title(f"Adversary Agent Performance Comparison (n={n_runs})", fontsize=18 + 3, fontweight='bold')
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f"{save_prefix}_n{n_runs}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_prefix}_n{n_runs}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved Adversary agent averaged comparison: {save_prefix}_n{n_runs}.pdf")
    return fig


def plot_averaged_adversary_comparison(aggregated_results, window=100, n_runs=5, save_prefix="ego_agent_averaged"):
    """Create separate averaged comparison plot for Adversary agent performance."""
    
    # Set matplotlib parameters - MATCH 5_with3agents_worked.py style
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'axes.titlesize': 18 + 3,  # fontsize + 3
        'axes.labelsize': 18 + 2,  # fontsize + 2
        'xtick.labelsize': 18 + 1,  # fontsize + 1
        'ytick.labelsize': 18 + 1,  # fontsize + 1
        'legend.fontsize': 18 + 1,  # fontsize + 1
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = {'causal': '#2E8B57', 'baseline': '#4169E1', 'counterfactual': '#FF8C00'}
    linestyles = {'causal': '-', 'baseline': '--', 'counterfactual': '-.'}
    
    # Plot averaged adversary win rates with confidence intervals
    for method, color in colors.items():
        mean_data = aggregated_results[method]['adv_wins']['mean']
        sem_data = aggregated_results[method]['adv_wins']['sem']
        
        ma_mean = moving_average(mean_data, window)
        ma_sem = moving_average(sem_data, window)
        
        x_axis = np.arange(len(ma_mean))
        
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma_mean, label=label, color=color, 
                linewidth=2.5, alpha=0.9, linestyle=linestyles[method])
        ax.fill_between(x_axis, ma_mean - 1.96*ma_sem, ma_mean + 1.96*ma_sem, 
                        color=color, alpha=0.2)
    
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Ego Agent Win Rate", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=18 + 1, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    # Make tick labels bold
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
    ax.set_title(f"Ego Agent Performance Comparison (n={n_runs})", fontsize=18 + 3, fontweight='bold')
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f"{save_prefix}_n{n_runs}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_prefix}_n{n_runs}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved Ego agent averaged comparison: {save_prefix}_n{n_runs}.pdf")
    return fig


def plot_averaged_draw_comparison(aggregated_results, window=100, n_runs=5, save_prefix="draw_rate_averaged"):
    """Create separate averaged comparison plot for Draw rates."""
    
    # Set matplotlib parameters - MATCH 5_with3agents_worked.py style
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'axes.titlesize': 18 + 3,  # fontsize + 3
        'axes.labelsize': 18 + 2,  # fontsize + 2
        'xtick.labelsize': 18 + 1,  # fontsize + 1
        'ytick.labelsize': 18 + 1,  # fontsize + 1
        'legend.fontsize': 18 + 1,  # fontsize + 1
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = {'causal': '#2E8B57', 'baseline': '#4169E1', 'counterfactual': '#FF8C00'}
    linestyles = {'causal': '-', 'baseline': '--', 'counterfactual': '-.'}
    
    # Plot averaged draw rates with confidence intervals
    for method, color in colors.items():
        mean_data = aggregated_results[method]['draws']['mean']
        sem_data = aggregated_results[method]['draws']['sem']
        
        ma_mean = moving_average(mean_data, window)
        ma_sem = moving_average(sem_data, window)
        
        x_axis = np.arange(len(ma_mean))
        
        if method == 'causal':
            label = 'NASTL-CIRL'
        elif method == 'baseline':
            label = 'Nash-Q'
        elif method == 'counterfactual':
            label = 'Counterfactual RL'
        else:
            label = method.capitalize()
        ax.plot(x_axis, ma_mean, label=label, color=color, 
                linewidth=2.5, alpha=0.9, linestyle=linestyles[method])
        ax.fill_between(x_axis, ma_mean - 1.96*ma_sem, ma_mean + 1.96*ma_sem, 
                        color=color, alpha=0.2)
    
    ax.set_xlabel("Episode", fontsize=18 + 2, fontweight='bold')
    ax.set_ylabel("Draw Rate", fontsize=18 + 2, fontweight='bold')
    ax.set_xlim(0, 2500)  # CHANGED: from left=0 to (0, 2500)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=18 + 1, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18 + 1)
    ax.set_title(f"Draw Rate Comparison (n={n_runs})", fontsize=18 + 3, fontweight='bold')
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f"{save_prefix}_n{n_runs}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_prefix}_n{n_runs}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved Draw rate averaged comparison: {save_prefix}_n{n_runs}.pdf")
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Single run demonstration
    print("\n" + "="*80)
    print("SINGLE RUN DEMONSTRATION")
    print("="*80)
    
    # Run single experiment
    results = run_competitive_experiment(seed=42)
    
    # Plot results
    plot_competitive_results(results, window=100, save_path="competitive_single_run")
    
    # Multiple runs for statistical analysis
    print("\n" + "="*80)
    print("MULTIPLE RUNS FOR STATISTICAL ANALYSIS")
    print("="*80)
    
    n_runs = 5
    aggregated, all_runs = run_multiple_competitive_experiments(n_runs=n_runs, seed_base=42)
    
    # Plot averaged results - combined view
    plot_averaged_competitive_results(aggregated, window=100, n_runs=n_runs)
    
    # Plot individual separate comparison figures
    plot_averaged_ego_comparison(aggregated, window=100, n_runs=n_runs)
    plot_averaged_adversary_comparison(aggregated, window=100, n_runs=n_runs)
    plot_averaged_draw_comparison(aggregated, window=100, n_runs=n_runs)