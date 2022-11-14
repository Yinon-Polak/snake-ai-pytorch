import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import wandb
from torch.optim.lr_scheduler import StepLR


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, init_kaiming_normal=False):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        if init_kaiming_normal:
            nn.init.kaiming_normal(self.linear1.weight, mode='fan_in')
            nn.init.kaiming_normal(self.linear2.weight, mode='fan_in')

        self.timestamp = datetime.now().isoformat()
        self.activations_layer_1 = None
        self.initial_activations = None
        self.initial_weights = None
        self.initial_x = None
        self.initial_bias = None

    def forward(self, input_x):
        x = F.relu(self.linear1(input_x))
        with torch.no_grad():
            self.activations_layer_1 = x.detach().clone()
            if not torch.is_tensor(self.initial_activations):
                self.initial_x = input_x
                self.initial_weights = self.linear1.weight.detach().clone()
                self.initial_bias = self.linear1.bias
                self.initial_activations = x.detach().clone()

        x = self.linear2(x)
        return x

    def save(self, file_name):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

class QTrainer:
    def __init__(self, model, lr, gamma, scheduler_step_size, scheduler_gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.lr_scheduler = StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.criterion = nn.MSELoss()
        self.checkpoint = self.save_state_checkpoint()

    def reset_state_to_last_checkpoint(self):
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

    def save_state_checkpoint(self):
        self.checkpoint = {
            'model_state_dict': copy.deepcopy(self.model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
            'scheduler_state_dict': copy.deepcopy(self.lr_scheduler.state_dict()),
        }
        return self.checkpoint

    def train_step(self, state, action, reward, next_state, done):
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if isinstance(state, tuple):
            state = torch.stack(state)
            next_state = torch.stack(next_state)
        else:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
