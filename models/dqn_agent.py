import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from models.per_buffer import PrioritizedReplayBuffer

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.buffer = PrioritizedReplayBuffer(capacity=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act_vector(self, state_vec):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def remember_vector(self, state_vec, action, reward, next_state_vec, done):
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        next_tensor = torch.tensor(next_state_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            target = self.model(state_tensor)[0].clone()
            if done:
                td_error = reward - target[action].item()
            else:
                next_q = self.target_model(next_tensor)[0].max().item()
                td_error = reward + self.gamma * next_q - target[action].item()

        self.buffer.push((state_vec, action, reward, next_state_vec, done), td_error)

    def replay_vector(self, beta=0.4):
        if len(self.buffer.buffer) < self.batch_size:
            return

        minibatch, indices, weights = self.buffer.sample(self.batch_size, beta)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = self.model(state_tensor)[0].detach().clone()
            with torch.no_grad():
                if done:
                    target[action] = reward
                else:
                    target[action] = reward + self.gamma * self.target_model(next_tensor)[0].max()

            prediction = self.model(state_tensor)[0]
            weight = torch.tensor(weights[i], dtype=torch.float32)
            loss = self.criterion(prediction, target) * weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
