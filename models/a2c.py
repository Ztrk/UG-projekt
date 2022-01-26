import time
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax, log_softmax
import matplotlib.pyplot as plt
from envs import get_tabular_grid

class TabularA2C(nn.Module):
    def __init__(self, n_states, n_actions) -> None:
        super(TabularA2C, self).__init__()
        self.policy = nn.Parameter(torch.randn((n_states, n_actions)))
        
    def forward(self, state):
        return self.policy[state]


class TabularCriticA2C(nn.Module):
    def __init__(self, n_states) -> None:
        super(TabularCriticA2C, self).__init__()
        self.value = nn.Parameter(torch.randn(n_states))
        
    def forward(self, state):
        return self.value[state]

def plot_mean_rewards(rewards, window):
    cumsum = np.cumsum(rewards)
    avg = (cumsum[window:] - cumsum[:-window]) / window

    plt.plot(avg)
    plt.savefig('plots/mean-rewards.pdf')
    plt.show()

def a2c():
    env = get_tabular_grid('small-sparse')
    agent = TabularA2C(env.observation_space.n, 9)
    critic = TabularCriticA2C(env.observation_space.n)

    agent_optimizer = torch.optim.SGD(agent.parameters(), lr=0.1)
    critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.05)

    discount_factor = 0.9
    rewards = []

    start = time.time()
    for epoch in range(10000):
        state = env.reset()
        done = False
        while not done:
            logits = agent(state)
            action_probas = softmax(logits, dim=0)
            action = torch.multinomial(action_probas, 1).item()

            next_state, reward, done, _ = env.step(action)
            # print(state, action, reward, next_state, done)

            next_reward = reward + (1 - done) * discount_factor * critic(next_state).detach()
            td = next_reward - critic(state)
            policy_loss = -td.detach() * log_softmax(logits, dim=0)[action]
            critic_loss = td * td
            loss = policy_loss + critic_loss

            agent_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            loss.backward()

            agent_optimizer.step()
            critic_optimizer.step()

            state = next_state
        rewards.append(reward)
    print(env.grid)
    print(agent.policy[:35])
    print(critic.value[:35].reshape((5, 7)))
    print(f'Elapsed time: {time.time() - start:.3f} s')
    plot_mean_rewards(rewards, window=20)
