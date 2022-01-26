import time
from collections import namedtuple
from itertools import chain
import numpy as np
import torch
import torch.nn.functional as F
from torch.special import entr as entropy
from higher import innerloop_ctx
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs import get_tabular_grid
from models import Tabular, TabularCritic, LPGNetwork

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

MemoryEntry = namedtuple('MemoryEntry', ['state', 'action', 'reward', 'next_state', 'done'])
LPGMemoryEntry = namedtuple('LPGMemoryEntry', ['state', 'action', 'reward', 'next_state', 'done', 'policy_target', 'y_target'])

def update_model(model):
    params = [p.clone().detach().requires_grad_(p.requires_grad) for p in model.parameters()]
    model.update_params(params)

class LPG():
    def __init__(self):
        self.lpg = LPGNetwork()
        self.lpg_optimizer = torch.optim.Adam(self.lpg.parameters(), lr=0.005)

        self.discount_factor = 0.99

        self.agent_update_steps = 10
        self.agent_update_cnt = 0
        self.lpg_update_steps = 5

        self.memory = []
        self.lpg_memory = []

    def create_func_models(self):
        with innerloop_ctx(self.agent, self.agent_optimizer, copy_initial_weights=False) as (agent, agent_optimizer):
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
        (policy_target, y_target), hidden = self.lpg(x, None)

        policy_loss = -policy_target * torch.take_along_dim(F.log_softmax(policy_logit, dim=1), actions.unsqueeze(1), dim=1).squeeze(1)
        kl_loss = F.kl_div(F.log_softmax(y_logit, dim=1), F.log_softmax(y_target, dim=1), log_target=True, reduction='batchmean')
        loss = policy_loss.mean() + 0.5 * kl_loss

        self.fagent_optimizer.step(loss)

        # Update critic
        next_reward = rewards + (1 - dones) * self.discount_factor * self.critic(next_states).detach()
        td = next_reward - self.critic(states)
        critic_loss = (td * td).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.lpg_memory.append(LPGMemoryEntry(states, actions, rewards, next_states, dones, policy_target, y_target))

        self.memory = []

    def update_lpg(self):
        states = torch.cat([e.state for e in self.lpg_memory])
        actions = torch.cat([e.action for e in self.lpg_memory])
        rewards = torch.cat([e.reward for e in self.lpg_memory])
        next_states = torch.cat([e.next_state for e in self.lpg_memory])
        dones = torch.cat([e.done for e in self.lpg_memory])
        policy_target = torch.cat([e.policy_target for e in self.lpg_memory])
        y_target = torch.cat([e.y_target for e in self.lpg_memory])

        # for i in range(len(returns) - 1, 0, -1):
            # returns[i - 1] += (1 - dones[i - 1]) * self.discount_factor * returns[i]

        logits, y = self.fagent(states)

        next_reward = rewards + self.discount_factor * (1 - dones) * self.critic(next_states)
        td = next_reward - self.critic(states)
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

        self.agent.load_state_dict(self.fagent.state_dict())
        self.create_func_models()

    def train(self, lifetimes, n_steps):
        rewards = []
        mean_rewards = []

        start = time.time()
        for epoch in range(lifetimes):
            env = get_tabular_grid('small-sparse')
            state = env.reset()

            self.agent = Tabular(env.observation_space.n, 9)
            self.critic = TabularCritic(env.observation_space.n)
            self.agent_optimizer = torch.optim.SGD(self.agent.parameters(), lr=0.1)
            self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.05)

            self.memory = []
            self.lpg_memory = []

            self.create_func_models()
            rewards.append([])

            for _ in tqdm(range(n_steps)):
                logits, _ = self.fagent(torch.tensor(state))
                action_probas = F.softmax(logits, dim=1)
                action = torch.multinomial(action_probas, 1).item()

                next_state, reward, done, _ = env.step(action)

                self.memory.append(MemoryEntry(state, action, reward, next_state, done))

                if len(self.memory) >= self.agent_update_steps or done:
                    self.agent_update_cnt += 1
                    self.update_agent()

                    if self.agent_update_cnt % self.lpg_update_steps == 0:
                        self.update_lpg()

                state = next_state

                if done:
                    rewards[-1].append(reward)
                    state = env.reset()

            self.lpg_optimizer.step()
            self.lpg_optimizer.zero_grad()
            mean_rewards.append(np.mean(rewards[-1]))
            print(f'Epoch: {epoch + 1}/{lifetimes} Lifetime reward: {mean_rewards[-1]}')
            
        # print('Mean rewards: ', np.mean(rewards))
        # print(env.grid)
        # print(self.agent.policy[:35])
        # print(self.critic.value[:35].reshape((5, 7)))
        elapsed = time.time() - start
        print(f'Elapsed time: {elapsed // 3600:.0f}h {(elapsed // 60) % 60:.0f}m {elapsed % 60:.3f}s')
        plot_mean_rewards(mean_rewards)
        plot_lifetime_rewards(rewards, window=20)

        torch.save(self.lpg.state_dict(), 'results/lpg.model')

def main():
    lpg = LPG()
    lpg.train(10, 10000)

if __name__ == '__main__':
    main()