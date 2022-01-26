import time
from collections import namedtuple
import torch
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.special import entr as entropy
from higher import innerloop_ctx
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs import get_tabular_grid
from models import Tabular, TabularCritic


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

def plot_mean_rewards(rewards):
    plt.plot(rewards)
    plt.savefig('plots/mean-rewards.pdf')
    plt.show()

def plot_lifetime_rewards(rewards, window):
    for reward in rewards:
        cumsum = np.cumsum(reward)
        avg = (cumsum[window:] - cumsum[:-window]) / window

        plt.plot(avg)

    plt.savefig('plots/lifetime-rewards.pdf')
    plt.show()

class LPGAgent():
    MemoryEntry = namedtuple('MemoryEntry', ['state', 'action', 'reward', 'next_state', 'done'])

    def __init__(self, lpg, env):
        self.env = env
        self.agent = Tabular(env.observation_space.n, 9)
        self.agent_optimizer = torch.optim.SGD(self.agent.parameters(), lr=0.01)

        if lpg.training:
            self.critic = TabularCritic(env.observation_space.n)
            self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.05)

        self.discount_factor = 0.99

        self.agent_update_steps = 10
        self.agent_update_cnt = 0
        self.lpg_update_steps = 5

        self.lpg = lpg
        self.memory = []

        self.create_func_model()

    def create_func_model(self):
        with innerloop_ctx(self.agent, self.agent_optimizer, copy_initial_weights=False, track_higher_grads=self.lpg.training) as (agent, agent_optimizer):
            self.fagent = agent
            self.fagent_optimizer = agent_optimizer

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
        
        policy_logit, y_logit = self.fagent(states)
        _, y_logit_next = self.fagent(next_states)

        action_probas = torch.take_along_dim(F.softmax(policy_logit, dim=1), actions.unsqueeze(1), dim=1).squeeze(1)
        x = LPGNetwork.LPGInput(rewards, dones, torch.full((n, ), self.discount_factor),
            action_probas, F.softmax(y_logit, dim=1), F.softmax(y_logit_next, dim=1))
        (policy_target, y_target), hidden = self.lpg.lpg(x, None)

        policy_loss = -policy_target * torch.take_along_dim(F.log_softmax(policy_logit, dim=1), actions.unsqueeze(1), dim=1).squeeze(1)
        kl_loss = F.kl_div(F.log_softmax(y_logit, dim=1), F.log_softmax(y_target, dim=1), log_target=True, reduction='batchmean')
        loss = policy_loss.mean() + 0.5 * kl_loss

        self.fagent_optimizer.step(loss)

        if self.lpg.training:
            # Update critic
            next_reward = rewards + (1 - dones) * self.discount_factor * self.critic(next_states).detach()
            td = next_reward - self.critic(states)

            critic_loss = (td * td).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.lpg.lpg_memory.append(self.lpg.MemoryEntry(states, actions, rewards, next_states, dones, policy_target, y_target))

        self.memory = []

    def train(self, n_steps):
        rewards = []
        episode_rewards = 0
        state = self.env.reset()

        for _ in tqdm(range(n_steps)):
            logits, _ = self.fagent(torch.tensor(state))
            action_probas = F.softmax(logits, dim=1)
            action = torch.multinomial(action_probas, 1).item()

            next_state, reward, done, _ = self.env.step(action)

            self.memory.append(self.MemoryEntry(state, action, reward, next_state, done))

            if len(self.memory) >= self.agent_update_steps or done:
                self.agent_update_cnt += 1
                self.update_agent()

                if self.lpg.training and self.agent_update_cnt % self.lpg_update_steps == 0:
                    self.lpg.compute_lpg_gradient()

            state = next_state
            episode_rewards += reward

            if done:
                rewards.append(episode_rewards)
                episode_rewards = 0
                state = self.env.reset()
        
        return rewards

class LPG():
    MemoryEntry = namedtuple('LPGMemoryEntry', ['state', 'action', 'reward', 'next_state', 'done', 'policy_target', 'y_target'])

    def __init__(self):
        self.lpg = LPGNetwork()
        self.lpg_optimizer = torch.optim.Adam(self.lpg.parameters(), lr=0.005)

        self.discount_factor = 0.99

        self.lpg_memory = []
        self.training = True

    def compute_lpg_gradient(self):
        states = torch.cat([e.state for e in self.lpg_memory])
        actions = torch.cat([e.action for e in self.lpg_memory])
        rewards = torch.cat([e.reward for e in self.lpg_memory])
        next_states = torch.cat([e.next_state for e in self.lpg_memory])
        dones = torch.cat([e.done for e in self.lpg_memory])
        policy_target = torch.cat([e.policy_target for e in self.lpg_memory])
        y_target = torch.cat([e.y_target for e in self.lpg_memory])

        # for i in range(len(returns) - 1, 0, -1):
            # returns[i - 1] += (1 - dones[i - 1]) * self.discount_factor * returns[i]

        logits, y = self.agent.fagent(states)

        next_reward = rewards + self.discount_factor * (1 - dones) * self.agent.critic(next_states)
        td = next_reward - self.agent.critic(states)
        policy_loss = -td.detach() * torch.take_along_dim(F.log_softmax(logits, dim=1), actions.unsqueeze(1), dim=1).squeeze(1)

        # Entropy regularization
        policy_entropy = -0.01 * entropy(F.softmax(logits, dim=1)).sum(dim=1).mean()
        pred_entropy = -0.001 * entropy(F.softmax(y, dim=1)).sum(dim=1).mean()

        # L2 regularization
        policy_l2 = 0.001 * (policy_target * policy_target).mean()
        y_target_softmax = F.softmax(y_target, dim=1)
        y_l2 = 0.001 * (y_target_softmax * y_target_softmax).sum(dim=1).mean()

        loss = policy_loss.mean() + policy_entropy + pred_entropy + policy_l2 + y_l2

        loss.backward()

        self.lpg_memory = []

        self.agent.agent.load_state_dict(self.agent.fagent.state_dict())
        self.agent.create_func_model()

    def train(self, lifetimes, n_steps):
        rewards = []
        mean_rewards = []

        start = time.time()
        for epoch in range(lifetimes):
            env = get_tabular_grid('very-small')
            self.lpg_memory = []
            self.agent = LPGAgent(self, env)

            rewards = self.agent.train(n_steps)

            self.lpg_optimizer.step()
            self.lpg_optimizer.zero_grad()

            mean_rewards.append(np.mean(rewards))
            print(f'Epoch: {epoch + 1}/{lifetimes} Lifetime reward: {mean_rewards[-1]}')
            
        elapsed = time.time() - start
        print(f'Elapsed time: {elapsed // 3600:.0f}h {(elapsed // 60) % 60:.0f}m {elapsed % 60:.3f}s')
        plot_mean_rewards(mean_rewards)
        plot_lifetime_rewards(rewards, window=20)

        torch.save(self.lpg.state_dict(), 'results/lpg.model')
