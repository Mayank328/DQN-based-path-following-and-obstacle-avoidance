import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import os

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Network specifically designed for circular path following
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.LayerNorm(128),  # Normalize inputs for stability
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # Separate paths for path following and obstacle avoidance
        self.path_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        self.obstacle_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # Combine both streams for final output
        self.output_layer = nn.Linear(256, output_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        path_features = self.path_stream(shared_features)
        obstacle_features = self.obstacle_stream(shared_features)
        combined = torch.cat([path_features, obstacle_features], dim=-1)
        return self.output_layer(combined)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Larger memory for better experience diversity
        self.memory = deque(maxlen=100_000)
        self.batch_size = 256  # Larger batch size
        
        # Adjusted parameters for circular path
        self.gamma = 0.99      # Higher discount for long-term path following
        self.epsilon = 1.0
        self.epsilon_min = 0.02  # Higher minimum exploration
        self.epsilon_decay = 0.998  # Slower decay
        self.learning_rate = 0.0003
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Use Adam optimizer with custom parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                  lr=self.learning_rate,
                                  betas=(0.9, 0.999),
                                  eps=1e-8)
        
        # Experience replay with reward scaling
        self.reward_scale = 0.1
        self.min_reward = float('inf')
        self.max_reward = float('-inf')

    def remember(self, state, action, reward, next_state, done):
        """Store experience with reward scaling"""
        # Update reward bounds
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        
        # Scale reward for better learning
        if self.max_reward > self.min_reward:
            scaled_reward = self.reward_scale * (reward - self.min_reward) / (self.max_reward - self.min_reward)
        else:
            scaled_reward = reward
        
        self.memory.append((state, action, scaled_reward, next_state, done))

    def act(self, state, evaluation=False):
        """Choose action using epsilon-greedy with noise for better exploration"""
        if not evaluation and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            
            # Add small noise during training
            if not evaluation:
                noise = torch.randn_like(q_values) * 0.05
                q_values += noise
            
            return q_values.argmax().item()

    def replay(self):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values with Double DQN
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            # Get Q-values from target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # Huber loss for robustness
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

        # Optimization step with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Soft update target network"""
        tau = 0.005  # Soft update parameter
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def save_model(self, filepath):
        """Save model weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath):
        """Load model weights"""
        self.policy_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.policy_net.state_dict())