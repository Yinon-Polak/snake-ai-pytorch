import enum

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        last_activation = F.relu(self.linear1(x))
        last_layer = self.linear2(last_activation)
        return last_activation, last_layer

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Ensemble(nn.Module):
    def __init__(self, model1, model2, hidden_size, output_size):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.norm_layer = nn.LayerNorm(hidden_size * 2)
        self.ensmbele_linear1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.ensmbele_linear2 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x1, x2):
        x1_relu_activation, _ = self.model1(x1)
        x2_relu_activation, _ = self.model2(x2)
        x = torch.cat((x1_relu_activation, x2_relu_activation), dim=-1)
        x = self.norm_layer(x)
        second_last_layer = F.relu(self.ensmbele_linear1(x))
        last_layer = self.ensmbele_linear2(x)
        return second_last_layer, last_layer

    def save(self, file_name='ensmble-model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class ModelType(enum.Enum):
    LINEAR_QNET = 0
    ENSEMBLE = 1


class QTrainer:
    def __init__(self, model1, ensmble, lr, gamma):
        self.n_games = 0
        self.n_games_model_change = 1501

        self.lr = lr
        self.gamma = gamma

        self.model1 = model1
        self.ensmble = ensmble

        self.model = model1
        self.model_type = ModelType.LINEAR_QNET
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def model_wrapper(self, x, index=20):
        if self.model_type == ModelType.LINEAR_QNET:
            return self.model(x[:, :index]) if len(x.shape) > 1 else self.model(x[:index])

        if self.model_type == ModelType.ENSEMBLE:
            return self.model(x[:, :index], x[:, index:]) if len(x.shape) > 1 else self.model(x[:index], x[index:])

        raise Exception("Unsported type ModelType: {}")

    def set_model(self, model):
        self.model.save(f'model-{self.n_games}.pth')
        self.model = model
        self.model_type = ModelType.ENSEMBLE

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def increment_n_games(self):
        self.n_games += 1
        if self.n_games == self.n_games_model_change:
            self.set_model(self.ensmble)

    def train_step(self, state, action, reward, next_state, done):
        action = torch.tensor(action, dtype=torch.long)  # todo create only once
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
        second_last_layer, pred = self.model_wrapper(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                second_last_layer_idx, pred_idx = self.model_wrapper(next_state[idx])
                Q_new = reward[idx] + self.gamma * torch.max(pred_idx)

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
