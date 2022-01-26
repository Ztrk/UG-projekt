from collections import namedtuple
import torch
from torch import nn


class LPGNetwork(nn.Module):
    LPGInput = namedtuple('LPGInput', ['reward', 'done', 'discount_factor', 'action_proba', 'y', 'y_next'])

    def __init__(self, hidden_size=32, y_size=30):
        super(LPGNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_size)
        self.y_embedding = nn.Sequential(
            nn.Linear(y_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.dense_pi = nn.Linear(hidden_size, 1)
        self.dense_y = nn.Linear(hidden_size, y_size)

    def forward(self, x: LPGInput, hidden=None):
        # x - tuple(reward, done, discount_factor, action_proba, y, next_y)
        y = self.y_embedding(x.y).squeeze(1)
        y_next = self.y_embedding(x.y_next).squeeze(1)
        x_embedded = x._replace(y=y, y_next=y_next)

        x = torch.stack(x_embedded, dim=1).flip(1).unsqueeze(1)

        out, hidden = self.lstm(x, hidden)
        pi = self.dense_pi(out).squeeze(1).squeeze(1)
        y = self.dense_y(out).squeeze(1).squeeze(1)
        return (pi, y), hidden
