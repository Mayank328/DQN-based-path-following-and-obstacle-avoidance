import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os

# Prioritized Replay Buffer Components
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PERMemory:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.0

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = []
        ISWeights = np.empty((n, 1))

        priority_segment = self.tree.total() / n
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        if min_prob == 0:
            min_prob = 1e-6

        for i in range(n):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            prob = p / self.tree.total()
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i]= idx
            b_memory.append(data)

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += 1e-6
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

# Dueling DQN network
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

# DQN Agent with Double DQN, PER, and Entropy Regularization
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = PERMemory(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.entropy_coeff = 0.01
        self.epsilon_min = 0.01
        self.learning_rate = 1e-4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        self.tau = 0.005  # Soft update factor

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        probs = torch.softmax(q_values / self.entropy_coeff, dim=1)
        action = torch.multinomial(probs, 1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def replay(self):
        if self.memory.tree.write < self.batch_size:
            return

        idxs, batch, ISWeights = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        ISWeights = torch.FloatTensor(ISWeights).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions)

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        self.memory.batch_update(idxs, td_errors)

        loss = (ISWeights * (q_values - target_q_values).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update_target()

    def soft_update_target(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.policy_net.state_dict())
