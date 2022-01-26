import time
from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

class TabularA2C(nn.Module):
    def __init__(self, n_states, n_actions) -> None:
        super(TabularA2C, self).__init__()
        self.policy = nn.Parameter(torch.randn((n_states, n_actions)))
        
    def forward(self, state):
        return torch.index_select(self.policy, 0, state)


class TabularCriticA2C(nn.Module):
    def __init__(self, n_states) -> None:
        super(TabularCriticA2C, self).__init__()
        self.value = nn.Parameter(torch.randn(n_states))
        
    def forward(self, state):
        return torch.index_select(self.value, 0, state)
        

def plot_mean_rewards(rewards, window):
    cumsum = np.cumsum(rewards)
    avg = (cumsum[window:] - cumsum[:-window]) / window

    plt.plot(avg)
    plt.xlabel('Liczba epizodów')
    plt.ylabel('Suma nagród')
    plt.savefig('plots/mean-rewards.pdf')
    plt.show()

def get_returns(rewards, discount_factor):
    ret = 0
    for reward in rewards[-1::-1]:
        ret = reward + discount_factor * ret
    return ret

class A2C():
    MemoryEntry = namedtuple('MemoryEntry', ['state', 'action', 'reward', 'next_state', 'done'])

    def __init__(self, env):
        self.env = env
        self.agent = TabularA2C(env.observation_space.n, 9)
        self.agent_optimizer = torch.optim.SGD(self.agent.parameters(), lr=0.1)

        self.critic = TabularCriticA2C(env.observation_space.n)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.05)

        self.discount_factor = 0.99
        self.agent_update_steps = 10

        self.memory = []

    def update_agent(self):
        n = len(self.memory)
        states = torch.zeros(n, dtype=torch.int)
        actions = torch.zeros(n, dtype=torch.long)
        rewards = torch.zeros(n)
        next_states = torch.zeros(n, dtype=torch.int)
        dones = torch.zeros(n, dtype=torch.int)

        for i, entry in enumerate(self.memory):
            states[i] = entry.state
            actions[i] = entry.action
            rewards[i] = entry.reward
            next_states[i] = entry.next_state
            dones[i] = entry.done
        
        policy_logit = self.agent(states)

        next_reward = rewards + (1 - dones) * self.discount_factor * self.critic(next_states).detach()
        td = next_reward - self.critic(states)

        policy_loss = -td.detach() * torch.take_along_dim(F.log_softmax(policy_logit, dim=1), actions.unsqueeze(1), dim=1).squeeze(1)
        critic_loss = td * td
        loss = policy_loss.mean() + critic_loss.mean()

        self.agent_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.agent_optimizer.step()
        self.critic_optimizer.step()

        self.memory = []

    def train(self, n_steps):
        rewards = []
        returns = []
        episode_rewards = []
        state = self.env.reset()

        for _ in tqdm(range(n_steps)):
            logits = self.agent(torch.tensor(state))
            action_probas = F.softmax(logits, dim=1)
            action = torch.multinomial(action_probas, 1).item()

            next_state, reward, done, _ = self.env.step(action)

            self.memory.append(self.MemoryEntry(state, action, reward, next_state, done))

            if len(self.memory) >= self.agent_update_steps or done:
                self.update_agent()

            state = next_state
            episode_rewards.append(reward)

            if done:
                rewards.append(np.sum(episode_rewards))
                returns.append(get_returns(episode_rewards, self.discount_factor))
                episode_rewards = []
                state = self.env.reset()
        
        return rewards, returns
