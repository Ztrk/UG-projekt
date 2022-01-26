import torch
from torch import nn

class Tabular(nn.Module):
    def __init__(self, n_states, n_actions, y_dim=30) -> None:
        super(Tabular, self).__init__()
        self.policy = nn.Parameter(torch.randn((n_states, n_actions)))
        self.y = nn.Parameter(torch.randn((n_states, y_dim)))
        
    def forward(self, state):
        return torch.index_select(self.policy, 0, state), torch.index_select(self.y, 0, state)


class TabularCritic(nn.Module):
    def __init__(self, n_states) -> None:
        super(TabularCritic, self).__init__()
        self.value = nn.Parameter(torch.randn(n_states))
        
    def forward(self, state):
        return torch.index_select(self.value, 0, state)
