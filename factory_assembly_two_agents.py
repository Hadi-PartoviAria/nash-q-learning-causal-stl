"""
Key-Door-Part Environment with Per-Agent Analysis
=================================================
Complete code for multi-agent reinforcement learning with causal guidance,
including per-agent reward convergence and completion comparison.

Environment: Two agents (T1, T2) must:
- T1: Get base_key → unlock base_door → collect base_part
- T2: Get head_key → unlock head_door → collect head_part
- Together: Assemble parts at assembly station

This code compares three approaches:
1. Causal Guidance: Uses causal patterns to guide exploration
2. Baseline: Standard Q-learning
3. Counterfactual: Q-learning with counterfactual reasoning
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

# Optional tikzplotlib for LaTeX export
try:
    import tikzplotlib
    TIKZ_AVAILABLE = True
except ImportError:
    TIKZ_AVAILABLE = False
    print("Warning: tikzplotlib not available. Install with 'pip install tikzplotlib' for LaTeX export.")

# ==========================================================================
# Shared plotting style (match zero.py)
# ==========================================================================
PLOT_COLORS = {'causal': '#2E8B57', 'baseline': '#4169E1', 'counterfactual': '#FF8C00'}
PLOT_LINESTYLES = {'causal': '-', 'baseline': '--', 'counterfactual': '-.'}

def method_label(method: str) -> str:
    if method == 'causal':
        return 'NASTL-CIRL'
    elif method == 'baseline':
        return 'Nash-Q'
    elif method == 'counterfactual':
        return 'Counterfactual RL'
    else:
        return method.capitalize()

def apply_zero_style(use_tex: bool = True):
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'serif',
        'text.usetex': use_tex,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0
    })


# ============================================================================
# PART 1: KEY-DOOR-PART ENVIRONMENT
# ============================================================================

class KeyDoorPartEnv:
    """Multi-agent environment with keys, doors, and parts."""
    
    def __init__(self, grid_size=8, step_cost=0.01, max_steps=100, t2_far=False):
        self.grid_size = grid_size
        self.step_cost = step_cost
        self.max_steps = max_steps
        
        # Action space
        self.action_space = ['up', 'down', 'left', 'right', 
                             'pickup_key', 'unlock_door', 
                             'pickup_part', 'assemble', 'noop']
        self.n_actions = len(self.action_space)
        self.n_agents = 2
        
        # Fixed positions
        self.base_key_pos = (1, 1)
        self.head_key_pos = (6, 1)
        self.base_door_pos = (1, 3)
        self.head_door_pos = (6, 3)
        self.base_area = [(1, 5), (1, 6), (2, 5), (2, 6)]
        self.head_area = [(5, 5), (5, 6), (6, 5), (6, 6)]
        self.assembly_pos = (3, 3)
        
        # Rewards
        self.pickup_key_reward = 1.0
        self.unlock_door_reward = 1.0
        self.pickup_part_reward = 3.0
        self.assembly_reward = 5.0

        if t2_far:
            # push T2’s objectives to the far bottom-right
            self.head_key_pos = (7, 7)   # farther from T2’s start (7,0)
            self.head_door_pos = (6, 7)  # near the corner
            # (optional) tighten head area to the lower-right; comment out if you
            # want to keep the old head_area.
            self.head_area = [(5, 6), (6, 6), (5, 7), (6, 7)]
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        # Agent positions
        self.t1_pos = (0, 0)
        self.t2_pos = (7, 0)
        
        # Item states
        self.t1_has_base_key = False
        self.t1_has_base = False
        self.t2_has_head_key = False
        self.t2_has_head = False
        
        # Door states
        self.base_door_open = False
        self.head_door_open = False
        
        # Item availability
        self.base_key_available = True
        self.head_key_available = True
        self.base_part_available = True
        self.head_part_available = True
        
        # Task completion
        self.parts_assembled = False
        self.assembly_completed = False
        
        self.steps = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation."""
        return (
            self.t1_pos, self.t2_pos,
            self.t1_has_base_key, self.t1_has_base,
            self.t2_has_head_key, self.t2_has_head,
            self.base_door_open, self.head_door_open,
            self.base_key_available, self.head_key_available,
            self.base_part_available, self.head_part_available,
            self.parts_assembled, self.assembly_completed
        )
    
    def _is_valid_pos(self, pos):
        """Check if position is within grid bounds."""
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
    
    def _move(self, pos, action):
        """Calculate new position after movement."""
        if action == 'up':
            new_pos = (pos[0], pos[1] - 1)
        elif action == 'down':
            new_pos = (pos[0], pos[1] + 1)
        elif action == 'left':
            new_pos = (pos[0] - 1, pos[1])
        elif action == 'right':
            new_pos = (pos[0] + 1, pos[1])
        else:
            return pos
        
        return new_pos if self._is_valid_pos(new_pos) else pos
    
    def legal_actions(self, agent_id, state=None):
        """Get legal actions for an agent in current state. Returns action indices."""
        if state is None:
            state = self._get_state()
        
        (t1_pos, t2_pos, t1_has_base_key, t1_has_base, 
         t2_has_head_key, t2_has_head, base_door_open, head_door_open,
         base_key_avail, head_key_avail, base_part_avail, head_part_avail,
         parts_assembled, assembly_completed) = state
        
        legal = []
        
        if agent_id == 0:  # T1
            pos = t1_pos
            
            # Movement actions (always legal)
            legal.extend([0, 1, 2, 3])  # up, down, left, right
            
            # Pickup base key
            if pos == self.base_key_pos and base_key_avail and not t1_has_base_key:
                legal.append(4)  # pickup_key
            
            # Unlock base door
            if pos == self.base_door_pos and t1_has_base_key and not base_door_open:
                legal.append(5)  # unlock_door
            
            # Pickup base part
            if pos in self.base_area and base_door_open and base_part_avail and not t1_has_base:
                legal.append(6)  # pickup_part
            
            # Assemble
            if pos == self.assembly_pos and t1_has_base and t2_has_head and not parts_assembled:
                legal.append(7)  # assemble
            
            legal.append(8)  # noop
            
        else:  # T2
            pos = t2_pos
            
            # Movement actions
            legal.extend([0, 1, 2, 3])
            
            # Pickup head key
            if pos == self.head_key_pos and head_key_avail and not t2_has_head_key:
                legal.append(4)  # pickup_key
            
            # Unlock head door
            if pos == self.head_door_pos and t2_has_head_key and not head_door_open:
                legal.append(5)  # unlock_door
            
            # Pickup head part
            if pos in self.head_area and head_door_open and head_part_avail and not t2_has_head:
                legal.append(6)  # pickup_part
            
            # Assemble
            if pos == self.assembly_pos and t1_has_base and t2_has_head and not parts_assembled:
                legal.append(7)  # assemble
            
            legal.append(8)  # noop
        
        return legal
    
    def step(self, actions: List[str]):
        """Execute actions and return next state, rewards, done.
        actions: list of action names of length n_agents
        """
        rewards = [0, 0]
        
        # T1 action
        if actions[0] in ['up', 'down', 'left', 'right']:
            self.t1_pos = self._move(self.t1_pos, actions[0])
        elif actions[0] == 'pickup_key':
            if self.t1_pos == self.base_key_pos and self.base_key_available and not self.t1_has_base_key:
                self.t1_has_base_key = True
                self.base_key_available = False
                rewards[0] += self.pickup_key_reward
        elif actions[0] == 'unlock_door':
            if self.t1_pos == self.base_door_pos and self.t1_has_base_key and not self.base_door_open:
                self.base_door_open = True
                rewards[0] += self.unlock_door_reward
        elif actions[0] == 'pickup_part':
            if self.t1_pos in self.base_area and self.base_door_open and self.base_part_available and not self.t1_has_base:
                self.t1_has_base = True
                self.base_part_available = False
                rewards[0] += self.pickup_part_reward
        elif actions[0] == 'assemble':
            if self.t1_pos == self.assembly_pos and self.t1_has_base and self.t2_has_head and not self.parts_assembled:
                self.parts_assembled = True
                self.assembly_completed = True
                rewards[0] += self.assembly_reward
                rewards[1] += self.assembly_reward
        
        # T2 action
        if actions[1] in ['up', 'down', 'left', 'right']:
            self.t2_pos = self._move(self.t2_pos, actions[1])
        elif actions[1] == 'pickup_key':
            if self.t2_pos == self.head_key_pos and self.head_key_available and not self.t2_has_head_key:
                self.t2_has_head_key = True
                self.head_key_available = False
                rewards[1] += self.pickup_key_reward
        elif actions[1] == 'unlock_door':
            if self.t2_pos == self.head_door_pos and self.t2_has_head_key and not self.head_door_open:
                self.head_door_open = True
                rewards[1] += self.unlock_door_reward
        elif actions[1] == 'pickup_part':
            if self.t2_pos in self.head_area and self.head_door_open and self.head_part_available and not self.t2_has_head:
                self.t2_has_head = True
                self.head_part_available = False
                rewards[1] += self.pickup_part_reward
        elif actions[1] == 'assemble':
            if self.t2_pos == self.assembly_pos and self.t1_has_base and self.t2_has_head and not self.parts_assembled:
                self.parts_assembled = True
                self.assembly_completed = True
                rewards[0] += self.assembly_reward
                rewards[1] += self.assembly_reward
        
        self.steps += 1
        done = self.assembly_completed or self.steps >= self.max_steps
        
        return self._get_state(), rewards, done


# ============================================================================
# PART 2: ENHANCED AGENTS WITH PER-AGENT TRACKING
# ============================================================================

class EnhancedBaselineAgent:
    """Baseline Q-learning with per-agent tracking."""
    
    def __init__(self, env, learning_rate=0.15, discount=0.95, epsilon=0.3):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount
        self.eps = epsilon
        self.eps_min = 0.01
        self.eps_decay = 0.995
        
        self.q = [defaultdict(lambda: np.zeros(env.n_actions)) for _ in range(env.n_agents)]
        
        # Overall tracking - DON'T initialize with zero (will be added in plotting)
        self.episode_rewards = []
        self.completion_rates = []
        self.parts_collected = []
        
        # Per-agent tracking - DON'T initialize with zero (will be added in plotting)
        self.t1_rewards = []
        self.t2_rewards = []
        self.t1_has_part_history = []
        self.t2_has_part_history = []
    
    def state_to_key(self, state):
        """Convert state to hashable key for Q-table."""
        (t1_pos, t2_pos, t1_has_base_key, t1_has_base, t2_has_head_key, t2_has_head,
         base_door_open, head_door_open, base_key_avail, head_key_avail,
         base_part_avail, head_part_avail, parts_assembled, assembly_done) = state
        
        return (t1_pos, t2_pos, t1_has_base_key, t1_has_base, 
                t2_has_head_key, t2_has_head, base_door_open, head_door_open,
                parts_assembled, assembly_done)
    
    def train(self, n_episodes=1000, warmup_eps=100):
        print("\nTraining Baseline Q-learning with Per-Agent Tracking...")
        
        for ep in range(n_episodes):
            self.eps = 1.0 if ep < warmup_eps else max(self.eps_min, self.eps * self.eps_decay)
            
            state = self.env.reset()
            episode_rewards = [0.0, 0.0]
            t1_ep_reward = 0.0
            t2_ep_reward = 0.0
            done = False
            
            while not done:
                # Select actions
                if random.random() < self.eps:
                    actions = [random.choice(self.env.legal_actions(i, state)) 
                              for i in range(self.env.n_agents)]
                else:
                    key = self.state_to_key(state)
                    actions = []
                    for i in range(self.env.n_agents):
                        legal = self.env.legal_actions(i, state)
                        q_vals = self.q[i][key]
                        masked_q = np.full_like(q_vals, -np.inf)
                        masked_q[legal] = q_vals[legal]
                        masked_q[legal] += np.random.normal(0, 0.01, len(legal))
                        actions.append(int(np.argmax(masked_q)))
                
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
                        legal_next = self.env.legal_actions(i, next_state)
                        if legal_next:
                            max_next_q = max([self.q[i][next_key][a] for a in legal_next])
                        else:
                            max_next_q = 0.0
                        target = rewards[i] + self.gamma * max_next_q
                        self.q[i][key][actions[i]] = (1 - self.lr) * self.q[i][key][actions[i]] + \
                                                     self.lr * target
                
                t1_ep_reward += rewards[0]
                t2_ep_reward += rewards[1]
                episode_rewards[0] += rewards[0]
                episode_rewards[1] += rewards[1]
                state = next_state
            
            # Record statistics
            self.t1_rewards.append(t1_ep_reward)
            self.t2_rewards.append(t2_ep_reward)
            self.t1_has_part_history.append(1.0 if self.env.t1_has_base else 0.0)
            self.t2_has_part_history.append(1.0 if self.env.t2_has_head else 0.0)
            self.episode_rewards.append(episode_rewards)
            parts_collected = int(self.env.t1_has_base) + int(self.env.t2_has_head)
            self.parts_collected.append(parts_collected)
            self.completion_rates.append(1.0 if self.env.assembly_completed else 0.0)
            
            if ep > 0 and ep % 200 == 0:
                avg_t1 = np.mean(self.t1_rewards[-100:])
                avg_t2 = np.mean(self.t2_rewards[-100:])
                avg_t1_part = np.mean(self.t1_has_part_history[-100:])
                avg_t2_part = np.mean(self.t2_has_part_history[-100:])
                avg_completion = np.mean(self.completion_rates[-100:])
                print(f"  Episode {ep}: T1 reward={avg_t1:.3f}, T2 reward={avg_t2:.3f}, "
                      f"T1 part={avg_t1_part:.2%}, T2 part={avg_t2_part:.2%}, "
                      f"Completion={avg_completion:.2%}")
        
        return self.episode_rewards, self.completion_rates, self.parts_collected


class EnhancedCounterfactualAgent:
    """Counterfactual Q-learning with per-agent tracking."""
    
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
        self.state_visits = defaultdict(int)
        
        self.q = [defaultdict(lambda: np.zeros(env.n_actions)) for _ in range(env.n_agents)]
        
        # Overall tracking - DON'T initialize with zero (will be added in plotting)
        self.episode_rewards = []
        self.completion_rates = []
        self.parts_collected = []
        
        # Per-agent tracking - DON'T initialize with zero (will be added in plotting)
        self.t1_rewards = []
        self.t2_rewards = []
        self.t1_has_part_history = []
        self.t2_has_part_history = []
    
    def state_to_key(self, state):
        """Convert state to hashable key for Q-table."""
        (t1_pos, t2_pos, t1_has_base_key, t1_has_base, t2_has_head_key, t2_has_head,
         base_door_open, head_door_open, base_key_avail, head_key_avail,
         base_part_avail, head_part_avail, parts_assembled, assembly_done) = state
        
        return (t1_pos, t2_pos, t1_has_base_key, t1_has_base, 
                t2_has_head_key, t2_has_head, base_door_open, head_door_open,
                parts_assembled, assembly_done)
    
    def _estimate_counterfactual_reward(self, state, agent_id, action):
        """Estimate immediate reward for an alternative action (shaped)."""
        (t1_pos, t2_pos, t1_has_base_key, t1_has_base, t2_has_head_key, t2_has_head,
         base_door_open, head_door_open, base_key_avail, head_key_avail,
         base_part_avail, head_part_avail, parts_assembled, assembly_done) = state
        
        action_name = self.env.action_space[action]
        base_reward = 0
        
        if agent_id == 0:  # T1
            if action_name == 'pickup_key' and t1_pos == self.env.base_key_pos and not t1_has_base_key:
                base_reward += 0.8
            elif action_name == 'unlock_door' and t1_pos == self.env.base_door_pos and t1_has_base_key:
                base_reward += 0.8
            elif action_name == 'pickup_part' and t1_pos in self.env.base_area and t1_has_base_key and base_door_open:
                base_reward += 2.5
            elif action_name == 'assemble' and t1_has_base and t2_has_head and t1_pos == self.env.assembly_pos:
                base_reward += 4.0
        else:  # T2
            if action_name == 'pickup_key' and t2_pos == self.env.head_key_pos and not t2_has_head_key:
                base_reward += 0.8
            elif action_name == 'unlock_door' and t2_pos == self.env.head_door_pos and t2_has_head_key:
                base_reward += 0.8
            elif action_name == 'pickup_part' and t2_pos in self.env.head_area and t2_has_head_key and head_door_open:
                base_reward += 2.5
            elif action_name == 'assemble' and t1_has_base and t2_has_head and t2_pos == self.env.assembly_pos:
                base_reward += 4.0
        
        return base_reward
    
    def train(self, n_episodes=1000, warmup_eps=100):
        print("\nTraining Counterfactual Q-learning with Per-Agent Tracking...")
        
        for ep in range(n_episodes):
            self.eps = 1.0 if ep < warmup_eps else max(self.eps_min, self.eps * self.eps_decay)
            
            state = self.env.reset()
            episode_rewards = [0.0, 0.0]
            t1_ep_reward = 0.0
            t2_ep_reward = 0.0
            done = False
            
            while not done:
                # Select actions
                if random.random() < self.eps:
                    actions = [random.choice(self.env.legal_actions(i, state)) 
                              for i in range(self.env.n_agents)]
                else:
                    key = self.state_to_key(state)
                    actions = []
                    for i in range(self.env.n_agents):
                        legal = self.env.legal_actions(i, state)
                        q_vals = self.q[i][key]
                        masked_q = np.full_like(q_vals, -np.inf)
                        masked_q[legal] = q_vals[legal]
                        masked_q[legal] += np.random.normal(0, 0.01, len(legal))
                        actions.append(int(np.argmax(masked_q)))
                
                # Execute
                action_names = [self.env.action_space[a] for a in actions]
                next_state, rewards, done = self.env.step(action_names)
                
                # Update with counterfactual reasoning
                key = self.state_to_key(state)
                next_key = self.state_to_key(next_state)
                self.state_visits[key] += 1
                
                # Standard update
                if done:
                    for i in range(self.env.n_agents):
                        self.q[i][key][actions[i]] = (1 - self.lr) * self.q[i][key][actions[i]] + \
                                                     self.lr * rewards[i]
                else:
                    for i in range(self.env.n_agents):
                        legal_next = self.env.legal_actions(i, next_state)
                        if legal_next:
                            max_next_q = max([self.q[i][next_key][a] for a in legal_next])
                        else:
                            max_next_q = 0.0
                        target = rewards[i] + self.gamma * max_next_q
                        self.q[i][key][actions[i]] = (1 - self.lr) * self.q[i][key][actions[i]] + \
                                                     self.lr * target
                
                # Counterfactual updates
                for i in range(self.env.n_agents):
                    legal = self.env.legal_actions(i, state)
                    for a in legal:
                        if a != actions[i]:
                            cf_reward = self._estimate_counterfactual_reward(state, i, a)
                            cf_reward += self.cf_bonus / np.sqrt(self.state_visits[key] + 1)
                            
                            if done:
                                cf_target = cf_reward
                            else:
                                legal_next = self.env.legal_actions(i, next_state)
                                if legal_next:
                                    cf_target = cf_reward + self.gamma * max([self.q[i][next_key][la] 
                                                                               for la in legal_next])
                                else:
                                    cf_target = cf_reward
                            
                            cf_lr = self.lr * self.cf_weight
                            self.q[i][key][a] = (1 - cf_lr) * self.q[i][key][a] + cf_lr * cf_target
                
                t1_ep_reward += rewards[0]
                t2_ep_reward += rewards[1]
                episode_rewards[0] += rewards[0]
                episode_rewards[1] += rewards[1]
                state = next_state
            
            # Record statistics
            self.t1_rewards.append(t1_ep_reward)
            self.t2_rewards.append(t2_ep_reward)
            self.t1_has_part_history.append(1.0 if self.env.t1_has_base else 0.0)
            self.t2_has_part_history.append(1.0 if self.env.t2_has_head else 0.0)
            self.episode_rewards.append(episode_rewards)
            parts_collected = int(self.env.t1_has_base) + int(self.env.t2_has_head)
            self.parts_collected.append(parts_collected)
            self.completion_rates.append(1.0 if self.env.assembly_completed else 0.0)
            
            if ep > 0 and ep % 200 == 0:
                avg_t1 = np.mean(self.t1_rewards[-100:])
                avg_t2 = np.mean(self.t2_rewards[-100:])
                avg_t1_part = np.mean(self.t1_has_part_history[-100:])
                avg_t2_part = np.mean(self.t2_has_part_history[-100:])
                avg_completion = np.mean(self.completion_rates[-100:])
                print(f"  Episode {ep}: T1 reward={avg_t1:.3f}, T2 reward={avg_t2:.3f}, "
                      f"T1 part={avg_t1_part:.2%}, T2 part={avg_t2_part:.2%}, "
                      f"Completion={avg_completion:.2%}")
        
        return self.episode_rewards, self.completion_rates, self.parts_collected


class EnhancedCausalGuidedAgent:
    """Causal-guided Q-learning with per-agent tracking."""
    
    def __init__(self, env, learning_rate=0.15, discount=0.95, epsilon=0.3):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount
        self.eps = epsilon
        self.eps_min = 0.01
        self.eps_decay = 0.995
        
        self.q = [defaultdict(lambda: np.zeros(env.n_actions)) for _ in range(env.n_agents)]
        
        # Overall tracking - DON'T initialize with zero (will be added in plotting)
        self.episode_rewards = []
        self.completion_rates = []
        self.parts_collected = []
        self.causal_guidance_used = []
        
        # Per-agent tracking - DON'T initialize with zero (will be added in plotting)
        self.t1_rewards = []
        self.t2_rewards = []
        self.t1_has_part_history = []
        self.t2_has_part_history = []
    
    def state_to_key(self, state):
        """Convert state to hashable key for Q-table."""
        (t1_pos, t2_pos, t1_has_base_key, t1_has_base, t2_has_head_key, t2_has_head,
         base_door_open, head_door_open, base_key_avail, head_key_avail,
         base_part_avail, head_part_avail, parts_assembled, assembly_done) = state
        
        return (t1_pos, t2_pos, t1_has_base_key, t1_has_base, 
                t2_has_head_key, t2_has_head, base_door_open, head_door_open,
                parts_assembled, assembly_done)
    
    def _should_use_causal_guidance(self, agent_id, state):
        """Determine if causal guidance should be used."""
        (t1_pos, t2_pos, t1_has_base_key, t1_has_base, t2_has_head_key, t2_has_head,
         base_door_open, head_door_open, _, _, _, _, _, _) = state
        
        if agent_id == 0:  # T1
            return not t1_has_base
        else:  # T2
            return not t2_has_head
    
    def _get_causal_guided_action(self, agent_id, state):
        """Get action index based on causal understanding."""
        (t1_pos, t2_pos, t1_has_base_key, t1_has_base, t2_has_head_key, t2_has_head,
         base_door_open, head_door_open, base_key_avail, head_key_avail,
         base_part_avail, head_part_avail, parts_assembled, assembly_done) = state
        
        legal = self.env.legal_actions(agent_id, state)
        legal_names = [self.env.action_space[a] for a in legal]
        
        if agent_id == 0:  # T1
            # Priority 1: Pick up key if don't have it
            if not t1_has_base_key and 'pickup_key' in legal_names:
                return self.env.action_space.index('pickup_key')
            # Priority 2: Unlock door if have key but door closed
            if t1_has_base_key and not base_door_open and 'unlock_door' in legal_names:
                return self.env.action_space.index('unlock_door')
            # Priority 3: Pick up part if have access
            if t1_has_base_key and base_door_open and not t1_has_base and 'pickup_part' in legal_names:
                return self.env.action_space.index('pickup_part')
            # Priority 4: Move toward next objective
            if not t1_has_base_key:
                target = self.env.base_key_pos
            elif not base_door_open:
                target = self.env.base_door_pos
            elif not t1_has_base:
                target = (1, 5)  # a base area cell
            else:
                target = self.env.assembly_pos
            
            current_pos = t1_pos
        else:  # T2
            if not t2_has_head_key and 'pickup_key' in legal_names:
                return self.env.action_space.index('pickup_key')
            if t2_has_head_key and not head_door_open and 'unlock_door' in legal_names:
                return self.env.action_space.index('unlock_door')
            if t2_has_head_key and head_door_open and not t2_has_head and 'pickup_part' in legal_names:
                return self.env.action_space.index('pickup_part')
            if not t2_has_head_key:
                target = self.env.head_key_pos
            elif not head_door_open:
                target = self.env.head_door_pos
            elif not t2_has_head:
                target = (6, 6)  # a head area cell
            else:
                target = self.env.assembly_pos
            
            current_pos = t2_pos
        
        # Choose movement toward target
        best_action = None
        best_dist = float('inf')
        for action_idx in legal:
            action_name = self.env.action_space[action_idx]
            if action_name in ['up', 'down', 'left', 'right']:
                new_pos = self.env._move(current_pos, action_name)
                dist = abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])
                if dist < best_dist:
                    best_dist = dist
                    best_action = action_idx
        return best_action if best_action is not None else random.choice(legal)
    
    def select_actions(self, state):
        """Select action indices using causal guidance during exploration."""
        actions = []
        for i in range(self.env.n_agents):
            if random.random() < self.eps:
                # Exploration with possible causal guidance
                if self._should_use_causal_guidance(i, state) and random.random() < 0.7:
                    action = self._get_causal_guided_action(i, state)
                    self.causal_guidance_used.append(1)
                else:
                    legal = self.env.legal_actions(i, state)
                    action = random.choice(legal)
                    self.causal_guidance_used.append(0)
            else:
                # Greedy selection using Q-values
                key = self.state_to_key(state)
                q_vals = self.q[i][key]
                legal = self.env.legal_actions(i, state)
                masked_q = np.full_like(q_vals, -np.inf)
                masked_q[legal] = q_vals[legal]
                masked_q[legal] += np.random.normal(0, 0.01, len(legal))
                action = int(np.argmax(masked_q))
                self.causal_guidance_used.append(0)
            actions.append(action)
        return actions
    
    def update(self, state, actions, rewards, next_state, done):
        """Independent Q-learning update (no reward shaping)."""
        key = self.state_to_key(state)
        next_key = self.state_to_key(next_state)
        
        if done:
            for i in range(self.env.n_agents):
                self.q[i][key][actions[i]] = (1 - self.lr) * self.q[i][key][actions[i]] + \
                                             self.lr * rewards[i]
        else:
            for i in range(self.env.n_agents):
                legal_next = self.env.legal_actions(i, next_state)
                next_q_vals = self.q[i][next_key]
                max_next_q = max([next_q_vals[a] for a in legal_next]) if legal_next else 0.0
                target = rewards[i] + self.gamma * max_next_q
                self.q[i][key][actions[i]] = (1 - self.lr) * self.q[i][key][actions[i]] + \
                                             self.lr * target
    
    def train(self, n_episodes=1000, warmup_eps=100):
        """Train the agent with causal guidance and per-agent tracking."""
        print("\nTraining with Causal Guidance and Per-Agent Tracking...")
        
        for ep in range(n_episodes):
            # Epsilon scheduling
            self.eps = 1.0 if ep < warmup_eps else max(self.eps_min, self.eps * self.eps_decay)
            
            state = self.env.reset()
            episode_rewards = [0.0, 0.0]
            t1_ep_reward = 0.0
            t2_ep_reward = 0.0
            done = False
            
            while not done:
                # Select and execute actions
                action_indices = self.select_actions(state)
                action_names = [self.env.action_space[i] for i in action_indices]
                
                next_state, rewards, done = self.env.step(action_names)
                
                # Update Q-values
                self.update(state, action_indices, rewards, next_state, done)
                
                t1_ep_reward += rewards[0]
                t2_ep_reward += rewards[1]
                episode_rewards[0] += rewards[0]
                episode_rewards[1] += rewards[1]
                
                state = next_state
            
            # Record episode statistics
            self.t1_rewards.append(t1_ep_reward)
            self.t2_rewards.append(t2_ep_reward)
            self.t1_has_part_history.append(1.0 if self.env.t1_has_base else 0.0)
            self.t2_has_part_history.append(1.0 if self.env.t2_has_head else 0.0)
            self.episode_rewards.append(episode_rewards)
            parts_collected = int(self.env.t1_has_base) + int(self.env.t2_has_head)
            self.parts_collected.append(parts_collected)
            self.completion_rates.append(1.0 if self.env.assembly_completed else 0.0)
            
            # Progress reporting
            if ep > 0 and ep % 200 == 0:
                avg_t1 = np.mean(self.t1_rewards[-100:])
                avg_t2 = np.mean(self.t2_rewards[-100:])
                avg_t1_part = np.mean(self.t1_has_part_history[-100:])
                avg_t2_part = np.mean(self.t2_has_part_history[-100:])
                avg_completion = np.mean(self.completion_rates[-100:])
                guidance_rate = np.mean(self.causal_guidance_used[-500:]) if self.causal_guidance_used else 0.0
                print(f"  Episode {ep}: T1 reward={avg_t1:.3f}, T2 reward={avg_t2:.3f}, "
                      f"T1 part={avg_t1_part:.2%}, T2 part={avg_t2_part:.2%}, "
                      f"Completion={avg_completion:.2%}, Guidance={guidance_rate:.1%}")
        
        return self.episode_rewards, self.completion_rates, self.parts_collected


# ============================================================================
# PART 3: PLOT SAVING UTILITIES AND ANALYSIS FUNCTIONS
# ============================================================================

def save_plot_multi_format(fig, filename_base, formats=["png", "pdf"]):
    """Save plot in multiple formats for academic publications."""
    saved_files = []
    
    for fmt in formats:
        output_path = f"{filename_base}.{fmt}"
        
        # Adjust DPI for different formats
        dpi = 300 if fmt == 'png' else None  # PDF is vector, doesn't need DPI
        
        fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        saved_files.append(output_path)
        print(f"Saved: {output_path}")
    
    return saved_files


def save_plot_with_tikz(fig, filename_base):
    """Save plot as PDF and generate LaTeX code using tikzplotlib."""
    saved_files = []
    
    # Save as PDF
    pdf_filename = f"{filename_base}.pdf"
    fig.savefig(pdf_filename, format='pdf', bbox_inches='tight', 
               pad_inches=0.1, facecolor='white', edgecolor='none')
    saved_files.append(pdf_filename)
    print(f"Saved: {pdf_filename}")
    
    # Generate LaTeX code if tikzplotlib is available
    if TIKZ_AVAILABLE:
        try:
            tikz_filename = f"{filename_base}.tex"
            # Configure tikzplotlib for better LaTeX output
            tikzplotlib.save(tikz_filename, 
                           figure=fig,
                           axis_width='10cm',  # Adjust as needed for your journal
                           axis_height='6cm',   # Adjust as needed for your journal
                           tex_relative_path_to_data='./figures/')
            saved_files.append(tikz_filename)
            print(f"Saved LaTeX code: {tikz_filename}")
            
            # Also save a standalone LaTeX document
            standalone_filename = f"{filename_base}_standalone.tex"
            with open(standalone_filename, 'w') as f:
                f.write("\\documentclass{standalone}\n")
                f.write("\\usepackage{pgfplots}\n")
                f.write("\\pgfplotsset{compat=1.16}\n")
                f.write("\\begin{document}\n")
                with open(tikz_filename, 'r') as tikz_file:
                    f.write(tikz_file.read())
                f.write("\\end{document}\n")
            saved_files.append(standalone_filename)
            print(f"Saved standalone LaTeX: {standalone_filename}")
            
        except Exception as e:
            print(f"Warning: Could not generate LaTeX code: {e}")
    else:
        print("Warning: tikzplotlib not available for LaTeX export")
    
    return saved_files


def create_publication_figure(width=10, height=6, fontsize=12):
    """Create a matplotlib figure optimized for publications."""
    fig = plt.figure(figsize=(width, height))
    
    # Set publication-quality parameters
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.titlesize': fontsize + 1,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize - 1,
        'ytick.labelsize': fontsize - 1,
        'legend.fontsize': fontsize - 1,
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3
    })
    
    return fig


# def moving_average(data, window=100):
#     """Compute moving average (1D list/array)."""
#     data = np.asarray(data, dtype=float)
#     if len(data) == 0:
#         return data
#     if len(data) < window:
#         return np.cumsum(data) / (np.arange(len(data)) + 1)
#     kernel = np.ones(window) / window
#     return np.convolve(data, kernel, mode='valid')


def plot_per_agent_results(results, window=100, save_path="per_agent_analysis.png"):
    """Create comprehensive per-agent analysis plots."""
    
    apply_zero_style(use_tex=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Per-Agent Analysis: Reward Convergence and Task Completion", fontsize=16)
    
    # Colors and linestyles to match zero.py
    colors = PLOT_COLORS
    linestyles = PLOT_LINESTYLES
    
    # T1 Reward Convergence
    ax = axes[0, 0]
    ax.set_title("T1 (Base Part) - Reward Convergence")
    for method, color in colors.items():
        data = results[method]['t1_rewards']
        ma = moving_average(data, window)
        x_axis = np.arange(len(ma))
        ax.plot(x_axis, ma, label=method_label(method), color=color, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward (MA)")
    ax.set_xlim(left=0)
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # T2 Reward Convergence
    ax = axes[1, 0]
    ax.set_title("T2 (Head Part) - Reward Convergence")
    for method, color in colors.items():
        data = results[method]['t2_rewards']
        ma = moving_average(data, window)
        x_axis = np.arange(len(ma))
        ax.plot(x_axis, ma, label=method_label(method), color=color, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward (MA)")
    ax.set_xlim(left=0)
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # T1 Part Collection Rate
    ax = axes[0, 1]
    ax.set_title("T1 - Base Part Collection Rate")
    for method, color in colors.items():
        ma = ema(results[method]['t1_parts'], alpha=2/(window+1))
        ax.plot(ma, label=method_label(method), color=color, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Collection Rate (MA)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # T2 Part Collection Rate
    ax = axes[1, 1]
    ax.set_title("T2 - Head Part Collection Rate")
    for method, color in colors.items():
        ma = ema(results[method]['t2_parts'], alpha=2/(window+1))
        ax.plot(ma, label=method_label(method), color=color, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Collection Rate (MA)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Agent Comparison - Final Performance
    ax = axes[0, 2]
    ax.set_title("Final Performance Comparison (Last 100 Episodes)")
    
    metrics_t1 = {}
    metrics_t2 = {}
    for method in colors.keys():
        metrics_t1[method] = {
            'reward': np.mean(results[method]['t1_rewards'][-100:]),
            'part_rate': np.mean(results[method]['t1_parts'][-100:])
        }
        metrics_t2[method] = {
            'reward': np.mean(results[method]['t2_rewards'][-100:]),
            'part_rate': np.mean(results[method]['t2_parts'][-100:])
        }
    
    x = np.arange(len(colors))
    width = 0.35
    rewards = [metrics_t1[m]['reward'] for m in colors.keys()]
    parts = [metrics_t1[m]['part_rate'] * 10 for m in colors.keys()]
    
    bars1 = ax.bar(x - width/2, rewards, width, label='Avg Reward', color='steelblue')
    bars2 = ax.bar(x + width/2, parts, width, label='Part Rate (×10)', color='coral')
    
    ax.set_ylabel('Value')
    ax.set_title('T1 Final Performance (Last 100 Episodes)')
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in colors.keys()])
    ax.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height/10:.2%}', ha='center', va='bottom', fontsize=9)
    
    # Bar plot for T2
    ax = axes[1, 2]
    rewards = [metrics_t2[m]['reward'] for m in colors.keys()]
    parts = [metrics_t2[m]['part_rate'] * 10 for m in colors.keys()]
    
    bars1 = ax.bar(x - width/2, rewards, width, label='Avg Reward', color='steelblue')
    bars2 = ax.bar(x + width/2, parts, width, label='Part Rate (×10)', color='coral')
    
    ax.set_ylabel('Value')
    ax.set_title('T2 Final Performance (Last 100 Episodes)')
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in colors.keys()])
    ax.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height/10:.2%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save in multiple formats
    base_name = save_path.replace('.png', '')
    save_plot_multi_format(fig, base_name, formats=["png", "pdf"])
    save_plot_with_tikz(fig, base_name)
    plt.show()
    print(f"\nSaved per-agent analysis to {save_path} and PDF/LaTeX formats")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PER-AGENT PERFORMANCE SUMMARY (Last 100 Episodes)")
    print("="*60)
    
    for agent_name, agent_metrics in [("T1 (Base Part)", metrics_t1), ("T2 (Head Part)", metrics_t2)]:
        print(f"\n{agent_name}:")
        print("-"*40)
        print(f"{'Method':<15} {'Avg Reward':>12} {'Part Rate':>12}")
        print("-"*40)
        for method in colors.keys():
            print(f"{method_label(method):<15} "
                  f"{agent_metrics[method]['reward']:>12.3f} "
                  f"{agent_metrics[method]['part_rate']:>11.1%}")

def plot_agent_coordination(results, window=50, save_path="agent_coordination.png"):
    """Plot coordination metrics between agents."""
    
    apply_zero_style(use_tex=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Agent Coordination Analysis", fontsize=16)
    
    colors = PLOT_COLORS
    linestyles = PLOT_LINESTYLES
    
    # Both agents have parts (coordination metric)
    ax = axes[0, 0]
    ax.set_title("Both Agents Have Parts (Coordination Success)")
    for method, color in colors.items():
        both_have = [int(t1 and t2) for t1, t2 in 
                     zip(results[method]['t1_parts'], results[method]['t2_parts'])]
        ma = moving_average(both_have, window)
        ax.plot(ma, label=method_label(method), color=color, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (MA)")
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Assembly completion rate
    ax = axes[0, 1]
    ax.set_title("Assembly Completion Rate")
    for method, color in colors.items():
        ma = moving_average(results[method]['completion'], window)
        ax.plot(ma, label=method_label(method), color=color, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Completion Rate (MA)")
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Reward balance between agents
    ax = axes[1, 0]
    ax.set_title("Reward Balance (T1 vs T2)")
    for method, color in colors.items():
        balance = [abs(t1 - t2) for t1, t2 in 
                   zip(results[method]['t1_rewards'], results[method]['t2_rewards'])]
        ma = moving_average(balance, window)
        ax.plot(ma, label=method_label(method), color=color, alpha=0.9, linestyle=linestyles[method])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward Difference |T1-T2| (MA)")
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Efficiency metric
    ax = axes[1, 1]
    ax.set_title("Learning Efficiency Metrics")
    
    efficiency_metrics = {}
    for method in colors.keys():
        both_have = [int(t1 and t2) for t1, t2 in 
                     zip(results[method]['t1_parts'], results[method]['t2_parts'])]
        first_coord = next((i for i, v in enumerate(both_have) if v), None)
        ma = moving_average(results[method]['completion'], window)
        ep_80 = next((i for i, v in enumerate(ma) if v >= 0.8), None)
        efficiency_metrics[method] = {
            'first_coordination': first_coord if first_coord else 1500,
            'episodes_to_80': ep_80 + window - 1 if ep_80 else 1500
        }
    
    x = np.arange(len(colors))
    width = 0.35
    first_coord_vals = [efficiency_metrics[m]['first_coordination'] for m in colors.keys()]
    ep_80_vals = [efficiency_metrics[m]['episodes_to_80'] for m in colors.keys()]
    
    bars1 = ax.bar(x - width/2, first_coord_vals, width, label='First Coordination', color='steelblue')
    bars2 = ax.bar(x + width/2, ep_80_vals, width, label='Episodes to 80%', color='coral')
    
    ax.set_ylabel('Episodes')
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in colors.keys()])
    ax.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    base_name = save_path.replace('.png', '')
    save_plot_multi_format(fig, base_name, formats=["png", "pdf"])
    save_plot_with_tikz(fig, base_name)
    plt.show()
    print(f"\nSaved coordination analysis to {save_path} and PDF/LaTeX formats")


def ema(data, alpha=0.05):
    data = np.asarray(data, dtype=float)
    if len(data) == 0: 
        return data
    out = np.empty_like(data, dtype=float)
    m = 0.0
    for i, x in enumerate(data):
        m = alpha * x + (1 - alpha) * m
        out[i] = m
    return out


def run_key_door_experiment_with_agent_tracking(seed=42):
    """Run complete experiment with per-agent tracking."""
    random.seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*80)
    print("KEY-DOOR-PART EXPERIMENT WITH PER-AGENT TRACKING")
    print("Causal Guidance vs Counterfactual vs Baseline")
    print("T1: access1 → has_base, T2: access2 → has_head")
    print("="*80)
    
    # Create environment
    env = KeyDoorPartEnv(grid_size=8, step_cost=0.0, t2_far=True)
    
    # Run experiments - all start fresh from episode 0
    print("\n--- CAUSAL GUIDANCE ---")
    env_causal = deepcopy(env)
    causal = EnhancedCausalGuidedAgent(env_causal, learning_rate=0.15, discount=0.95, epsilon=0.3)
    c_rewards, c_completion, c_parts = causal.train(n_episodes=1500, warmup_eps=150)
    
    print("\n--- BASELINE Q-LEARNING ---")
    env_baseline = deepcopy(env)
    baseline = EnhancedBaselineAgent(env_baseline, learning_rate=0.15, discount=0.95, epsilon=0.3)
    baseline_rewards, baseline_completion, baseline_parts = baseline.train(n_episodes=1500, warmup_eps=150)
    
    print("\n--- COUNTERFACTUAL Q-LEARNING ---")
    env_cf = deepcopy(env)
    counterfactual = EnhancedCounterfactualAgent(env_cf, learning_rate=0.15, discount=0.95, epsilon=0.3, 
                                                  cf_weight=0.5, cf_bonus=0.1)
    cf_rewards, cf_completion, cf_parts = counterfactual.train(n_episodes=1500, warmup_eps=150)
    
    # Compile results with per-agent data - add zero at the beginning for all methods
    results = {
        'causal': {
            'rewards': [0.0] + [r[0] + r[1] for r in c_rewards],
            'parts': [0] + c_parts,
            'completion': [0.0] + c_completion,
            't1_rewards': [0.0] + causal.t1_rewards,
            't2_rewards': [0.0] + causal.t2_rewards,
            't1_parts': [0.0] + causal.t1_has_part_history,
            't2_parts': [0.0] + causal.t2_has_part_history
        },
        'baseline': {
            'rewards': [0.0] + [r[0] + r[1] for r in baseline_rewards],
            'parts': [0] + baseline_parts,
            'completion': [0.0] + baseline_completion,
            't1_rewards': [0.0] + baseline.t1_rewards,
            't2_rewards': [0.0] + baseline.t2_rewards,
            't1_parts': [0.0] + baseline.t1_has_part_history,
            't2_parts': [0.0] + baseline.t2_has_part_history
        },
        'counterfactual': {
            'rewards': [0.0] + [r[0] + r[1] for r in cf_rewards],
            'parts': [0] + cf_parts,
            'completion': [0.0] + cf_completion,
            't1_rewards': [0.0] + counterfactual.t1_rewards,
            't2_rewards': [0.0] + counterfactual.t2_rewards,
            't1_parts': [0.0] + counterfactual.t1_has_part_history,
            't2_parts': [0.0] + counterfactual.t2_has_part_history
        }
    }
    return results


def run_multiple_experiments(n_runs=5, seed_base=42):
    """Run experiment multiple times and return aggregated results."""
    print(f"\n{'='*80}")
    print(f"RUNNING {n_runs} EXPERIMENTS FOR STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    all_results = []
    
    for run in range(n_runs):
        print(f"\n--- RUN {run + 1}/{n_runs} ---")
        seed = seed_base + run * 100  # Use different seeds for each run
        results = run_key_door_experiment_with_agent_tracking(seed=seed)
        all_results.append(results)
    
    # Aggregate results
    aggregated = {}
    methods = ['causal', 'baseline', 'counterfactual']
    metrics = ['rewards', 'parts', 'completion', 't1_rewards', 't2_rewards', 't1_parts', 't2_parts']
    
    for method in methods:
        aggregated[method] = {}
        for metric in metrics:
            # Collect all runs for this method and metric
            all_runs_data = []
            for result in all_results:
                all_runs_data.append(result[method][metric])
            
            # Convert to numpy array and calculate statistics
            # Ensure all runs have the same length (they should)
            min_length = min(len(run_data) for run_data in all_runs_data)
            truncated_data = [run_data[:min_length] for run_data in all_runs_data]
            data_array = np.array(truncated_data)
            
            aggregated[method][metric] = {
                'mean': np.mean(data_array, axis=0),
                'std': np.std(data_array, axis=0),
                'sem': np.std(data_array, axis=0) / np.sqrt(n_runs),  # Standard error of mean
                'raw_runs': truncated_data
            }
    
    return aggregated, all_results


def moving_average(data, window=100):
    """Fixed-window moving average that starts at episode 0 via left zero-padding."""
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return data
    if window <= 1:
        return data.copy()
    x = np.concatenate([np.zeros(window - 1), data])           # pad on the left
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid') 


def plot_averaged_agent_comparisons(aggregated_results, window=100, n_runs=5):
    """Create 4 separate figures with averaged results and confidence intervals."""
    
    apply_zero_style(use_tex=True)
    colors = PLOT_COLORS
    linestyles = PLOT_LINESTYLES
    
    # Figure 1: T1 Average Reward Comparison
    fig1 = plt.figure(figsize=(12, 6))
    plt.title(f"T1 Agent (Base Part) - Average Reward Comparison (n={n_runs})", fontsize=14, fontweight='bold')
    
    for method, color in colors.items():
        mean_data = aggregated_results[method]['t1_rewards']['mean']
        sem_data = aggregated_results[method]['t1_rewards']['sem']
        ma_mean = moving_average(mean_data, window)
        ma_sem = moving_average(sem_data, window)
        x_axis = np.arange(len(ma_mean))
        plt.plot(x_axis, ma_mean, label=method_label(method), color=color, linewidth=2, alpha=0.9, linestyle=linestyles[method])
        plt.fill_between(x_axis, ma_mean - 1.96*ma_sem, ma_mean + 1.96*ma_sem, 
                        color=color, alpha=0.18, label=None)
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward (Moving Average)", fontsize=12)
    plt.xlim(left=0)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_multi_format(fig1, f"t1_reward_comparison_averaged_n{n_runs}", formats=["png", "pdf"])
    save_plot_with_tikz(fig1, f"t1_reward_comparison_averaged_n{n_runs}")
    plt.show()
    
    # Figure 2: T2 Average Reward Comparison
    fig2 = plt.figure(figsize=(12, 6))
    plt.title(f"T2 Agent (Head Part) - Average Reward Comparison (n={n_runs})", fontsize=14, fontweight='bold')
    
    for method, color in colors.items():
        mean_data = aggregated_results[method]['t2_rewards']['mean']
        sem_data = aggregated_results[method]['t2_rewards']['sem']
        ma_mean = moving_average(mean_data, window)
        ma_sem = moving_average(sem_data, window)
        x_axis = np.arange(len(ma_mean))
        plt.plot(x_axis, ma_mean, label=method_label(method), color=color, linewidth=2, alpha=0.9, linestyle=linestyles[method])
        plt.fill_between(x_axis, ma_mean - 1.96*ma_sem, ma_mean + 1.96*ma_sem, 
                        color=color, alpha=0.18, label=None)
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward (Moving Average)", fontsize=12)
    plt.xlim(left=0)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_multi_format(fig2, f"t2_reward_comparison_averaged_n{n_runs}", formats=["png", "pdf"])
    save_plot_with_tikz(fig2, f"t2_reward_comparison_averaged_n{n_runs}")
    plt.show()
    
    EMA_ALPHA_SUCCESS = 0.2

    # Figure 3: T1 Success Rate Comparison
    fig3 = plt.figure(figsize=(12, 6))
    plt.title(f"T1 Agent - Success Rate (Base Part Collection) (n={n_runs})", fontsize=14, fontweight='bold')

    for method, color in colors.items():
        mean_data = np.asarray(aggregated_results[method]['t1_parts']['mean'], dtype=float)
        sem_data  = np.asarray(aggregated_results[method]['t1_parts']['sem'],  dtype=float)
        ma_mean = ema(mean_data, alpha=EMA_ALPHA_SUCCESS)
        ma_sem  = ema(sem_data,  alpha=EMA_ALPHA_SUCCESS)
        x_axis = np.arange(len(ma_mean))
        lower = np.clip(ma_mean - 1.96 * ma_sem, 0.0, 1.0)
        upper = np.clip(ma_mean + 1.96 * ma_sem, 0.0, 1.0)
        plt.plot(x_axis, ma_mean, label=method_label(method), color=color, linewidth=2, alpha=0.9, linestyle=linestyles[method])
        plt.fill_between(x_axis, lower, upper, color=color, alpha=0.18, label=None)

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Success Rate (EMA)", fontsize=12)
    plt.xlim(left=0); plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_multi_format(fig3, f"t1_success_rate_comparison_averaged_n{n_runs}", formats=["png", "pdf"])
    save_plot_with_tikz(fig3, f"t1_success_rate_comparison_averaged_n{n_runs}")
    plt.show()

    # Figure 4: T2 Success Rate Comparison
    fig4 = plt.figure(figsize=(12, 6))
    plt.title(f"T2 Agent - Success Rate (Head Part Collection) (n={n_runs})", fontsize=14, fontweight='bold')

    for method, color in colors.items():
        mean_data = np.asarray(aggregated_results[method]['t2_parts']['mean'], dtype=float)
        sem_data  = np.asarray(aggregated_results[method]['t2_parts']['sem'],  dtype=float)
        ma_mean = ema(mean_data, alpha=EMA_ALPHA_SUCCESS)
        ma_sem  = ema(sem_data,  alpha=EMA_ALPHA_SUCCESS)
        x_axis = np.arange(len(ma_mean))
        lower = np.clip(ma_mean - 1.96 * ma_sem, 0.0, 1.0)
        upper = np.clip(ma_mean + 1.96 * ma_sem, 0.0, 1.0)
        plt.plot(x_axis, ma_mean, label=method_label(method), color=color, linewidth=2, alpha=0.9, linestyle=linestyles[method])
        plt.fill_between(x_axis, lower, upper, color=color, alpha=0.18, label=None)

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Success Rate (EMA)", fontsize=12)
    plt.xlim(left=0); plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_multi_format(fig4, f"t2_success_rate_comparison_averaged_n{n_runs}", formats=["png", "pdf"])
    save_plot_with_tikz(fig4, f"t2_success_rate_comparison_averaged_n{n_runs}")
    plt.show()


def print_averaged_statistics(aggregated_results, n_runs=5):
    """Print detailed statistics for averaged results."""
    print("\n" + "="*60)
    print(f"AVERAGED PERFORMANCE SUMMARY (n={n_runs}, Last 100 Episodes)")
    print("="*60)
    
    methods = ['causal', 'baseline', 'counterfactual']
    
    print("\nT1 Agent (Base Part):")
    print("-"*60)
    print(f"{'Method':<15} {'Avg Reward':>12} {'± 95% CI':>12} {'Success Rate':>12} {'± 95% CI':>12}")
    print("-"*60)
    for method in methods:
        # Calculate final performance (last 100 episodes)
        reward_mean = np.mean(aggregated_results[method]['t1_rewards']['mean'][-100:])
        reward_sem = np.mean(aggregated_results[method]['t1_rewards']['sem'][-100:])
        reward_ci = 1.96 * reward_sem
        
        success_mean = np.mean(aggregated_results[method]['t1_parts']['mean'][-100:])
        success_sem = np.mean(aggregated_results[method]['t1_parts']['sem'][-100:])
        success_ci = 1.96 * success_sem
        
        print(f"{method_label(method):<15} {reward_mean:>12.3f} {'±' + str(reward_ci)[:5]:>12} "
              f"{success_mean:>11.1%} {'±' + f'{success_ci:.1%}':>12}")
    
    print("\nT2 Agent (Head Part):")
    print("-"*60)
    print(f"{'Method':<15} {'Avg Reward':>12} {'± 95% CI':>12} {'Success Rate':>12} {'± 95% CI':>12}")
    print("-"*60)
    for method in methods:
        reward_mean = np.mean(aggregated_results[method]['t2_rewards']['mean'][-100:])
        reward_sem = np.mean(aggregated_results[method]['t2_rewards']['sem'][-100:])
        reward_ci = 1.96 * reward_sem
        
        success_mean = np.mean(aggregated_results[method]['t2_parts']['mean'][-100:])
        success_sem = np.mean(aggregated_results[method]['t2_parts']['sem'][-100:])
        success_ci = 1.96 * success_sem
        
        print(f"{method_label(method):<15} {reward_mean:>12.3f} {'±' + str(reward_ci)[:5]:>12} "
              f"{success_mean:>11.1%} {'±' + f'{success_ci:.1%}':>12}")
    
    # Overall completion rates
    print("\nOverall Task Completion Rate:")
    print("-"*40)
    print(f"{'Method':<15} {'Completion Rate':>15} {'± 95% CI':>12}")
    print("-"*40)
    for method in methods:
        completion_mean = np.mean(aggregated_results[method]['completion']['mean'][-100:])
        completion_sem = np.mean(aggregated_results[method]['completion']['sem'][-100:])
        completion_ci = 1.96 * completion_sem
        
        print(f"{method_label(method):<15} {completion_mean:>14.1%} {'±' + f'{completion_ci:.1%}':>12}")


# Update the plot_agent_comparisons function to handle both single and multiple runs
def plot_agent_comparisons(results, window=100, is_averaged=False, n_runs=1):
    """Create 4 separate figures comparing agents' performance."""
    if is_averaged:
        plot_averaged_agent_comparisons(results, window, n_runs)
        return
    
    apply_zero_style(use_tex=True)
    colors = PLOT_COLORS
    linestyles = PLOT_LINESTYLES
    
    # Figure 1: T1 Average Reward Comparison
    fig1 = plt.figure(figsize=(10, 6))
    plt.title("T1 Agent (Base Part) - Average Reward Comparison", fontsize=14, fontweight='bold')
    
    for method, color in colors.items():
        data = results[method]['t1_rewards']
        if len(data) == 0 or abs(data[0]) > 1e-10:
            data = [0.0] + list(data)
        ma = moving_average(data, window)
        x_axis = np.arange(len(ma))
        plt.plot(x_axis, ma, label=method_label(method), color=color, linewidth=2, alpha=0.9, linestyle=linestyles[method])
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward (Moving Average)", fontsize=12)
    plt.xlim(left=0)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_multi_format(fig1, "t1_reward_comparison", formats=["png", "pdf"])
    save_plot_with_tikz(fig1, "t1_reward_comparison")
    plt.show()
    
    # Figure 2: T2 Average Reward Comparison
    fig2 = plt.figure(figsize=(10, 6))
    plt.title("T2 Agent (Head Part) - Average Reward Comparison", fontsize=14, fontweight='bold')
    
    for method, color in colors.items():
        data = results[method]['t2_rewards']
        if len(data) == 0 or abs(data[0]) > 1e-10:
            data = [0.0] + list(data)
        ma = moving_average(data, window)
        x_axis = np.arange(len(ma))
        plt.plot(x_axis, ma, label=method_label(method), color=color, linewidth=2, alpha=0.9, linestyle=linestyles[method])
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward (Moving Average)", fontsize=12)
    plt.xlim(left=0)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_multi_format(fig2, "t2_reward_comparison", formats=["png", "pdf"])
    save_plot_with_tikz(fig2, "t2_reward_comparison")
    plt.show()
    
    # Figure 3: T1 Success Rate (Part Collection) Comparison
    fig3 = plt.figure(figsize=(10, 6))
    plt.title("T1 Agent - Success Rate (Base Part Collection)", fontsize=14, fontweight='bold')
    
    for method, color in colors.items():
        data = results[method]['t1_parts']
        if len(data) == 0 or abs(data[0]) > 1e-10:
            data = [0.0] + list(data)
        ma = ema(results[method]['t1_parts'], alpha=2/(window+1))
        x_axis = np.arange(len(ma))
        plt.plot(x_axis, ma, label=method_label(method), color=color, linewidth=2, alpha=0.9, linestyle=linestyles[method])
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Success Rate (Moving Average)", fontsize=12)
    plt.xlim(left=0)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_multi_format(fig3, "t1_success_rate_comparison", formats=["png", "pdf"])
    save_plot_with_tikz(fig3, "t1_success_rate_comparison")
    plt.show()
    
    # Figure 4: T2 Success Rate (Part Collection) Comparison
    fig4 = plt.figure(figsize=(10, 6))
    plt.title("T2 Agent - Success Rate (Head Part Collection)", fontsize=14, fontweight='bold')
    
    for method, color in colors.items():
        data = results[method]['t2_parts']
        if len(data) == 0 or abs(data[0]) > 1e-10:
            data = [0.0] + list(data)
        ma = ema(results[method]['t2_parts'], alpha=2/(window+1))
        x_axis = np.arange(len(ma))
       
        plt.plot(x_axis, ma, label=method_label(method), color=color, linewidth=2, alpha=0.9, linestyle=linestyles[method])
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Success Rate (Moving Average)", fontsize=12)
    plt.xlim(left=0)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_multi_format(fig4, "t2_success_rate_comparison", formats=["png", "pdf"])
    save_plot_with_tikz(fig4, "t2_success_rate_comparison")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY (Last 100 Episodes)")
    print("="*60)
    
    print("\nT1 Agent (Base Part):")
    print("-"*40)
    print(f"{'Method':<15} {'Avg Reward':>12} {'Success Rate':>12}")
    print("-"*40)
    for method in colors.keys():
        avg_reward = np.mean(results[method]['t1_rewards'][-100:] if len(results[method]['t1_rewards']) > 0 else [0])
        success_rate = np.mean(results[method]['t1_parts'][-100:] if len(results[method]['t1_parts']) > 0 else [0])
        print(f"{method_label(method):<15} {avg_reward:>12.3f} {success_rate:>11.1%}")
    
    print("\nT2 Agent (Head Part):")
    print("-"*40)
    print(f"{'Method':<15} {'Avg Reward':>12} {'Success Rate':>12}")
    print("-"*40)
    for method in colors.keys():
        avg_reward = np.mean(results[method]['t2_rewards'][-100:] if len(results[method]['t2_rewards']) > 0 else [0])
        success_rate = np.mean(results[method]['t2_parts'][-100:] if len(results[method]['t2_parts']) > 0 else [0])
        print(f"{method_label(method):<15} {avg_reward:>12.3f} {success_rate:>11.1%}")

# Usage example (to be added at the end of your main code):

if __name__ == "__main__":
    # Run experiment with per-agent tracking
    results = run_key_door_experiment_with_agent_tracking(seed=42)
    
    # Create the 4 comparison plots
    plot_agent_comparisons(results, window=100)



# ============================================================================
# MAIN EXECUTION
# ============================================================================

# if __name__ == "__main__":
#     # Run experiment with per-agent tracking
#     results = run_key_door_experiment_with_agent_tracking(seed=42)
    
#     # Create per-agent analysis plots
#     plot_per_agent_results(results, window=100, save_path="per_agent_analysis.png")
    
#     # Create coordination analysis plots
#     plot_agent_coordination(results, window=50, save_path="agent_coordination.png")
    
#     # Print detailed statistics
#     print_detailed_statistics(results)

if __name__ == "__main__":
    # Run multiple experiments for statistical analysis
    n_runs = 5
    aggregated_results, all_individual_results = run_multiple_experiments(n_runs=n_runs, seed_base=42)
    
    # Create averaged plots with confidence intervals
    plot_averaged_agent_comparisons(aggregated_results, window=100, n_runs=n_runs)
    
    # Print averaged statistics
    print_averaged_statistics(aggregated_results, n_runs=n_runs)