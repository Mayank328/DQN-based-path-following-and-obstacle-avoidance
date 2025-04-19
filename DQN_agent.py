# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# import random
# import numpy as np
# import os

# class DQN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_size)
#         )

#     def forward(self, x):
#         return self.network(x)


# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size

#         # Initialize replay memory
#         self.memory = deque(maxlen=10000)
#         self.batch_size = 32
#         self.gamma = 0.95
#         self.epsilon = 1.0
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001

#         # Set up device
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize networks
#         self.policy_net = DQN(state_size, action_size).to(self.device)
#         self.target_net = DQN(state_size, action_size).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())

#         # Initialize optimizer
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

#     def remember(self, state, action, reward, next_state, done):
#         """Store experience in replay memory"""
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         """Choose action using epsilon-greedy policy"""
#         if random.random() < self.epsilon:
#             return random.randrange(self.action_size)
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             return self.policy_net(state).argmax().item()

#     def replay(self):
#         """Train on a batch of experiences"""
#         if len(self.memory) < self.batch_size:
#             return

#         # Sample batch
#         batch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         states = torch.FloatTensor(states).to(self.device)
#         actions = torch.LongTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)

#         # Compute Q values
#         current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
#         next_q_values = self.target_net(next_states).max(1)[0]
#         target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

#         # Loss and optimization
#         loss = nn.MSELoss()(current_q_values, target_q_values)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # Decay epsilon
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     def update_target_network(self):
#         """Update the target network"""
#         self.target_net.load_state_dict(self.policy_net.state_dict())

#     def save_model(self, filepath):
#         """Save model weights"""
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
#         torch.save(self.policy_net.state_dict(), filepath)

#     def load_model(self, filepath):
#         """Load model weights"""
#         self.policy_net.load_state_dict(torch.load(filepath))
#         self.target_net.load_state_dict(self.policy_net.state_dict())

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
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize replay memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def replay(self):
        """Train on a batch of experiences and return the loss value"""
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples to train

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss and optimization
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        """Update the target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath):
        """Save model weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath):
        """Load model weights"""
        self.policy_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.policy_net.state_dict())
