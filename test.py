import time
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from envs import get_tabular_grid
from models import LPG, LPGAgent, A2C

def plot_lifetime_rewards(rewards, window):
    for reward, label in zip(rewards, ['LPG', 'A2C']):
        cumsum = np.cumsum(reward)
        avg = (cumsum[window:] - cumsum[:-window]) / window

        plt.plot(avg, label=label)

    plt.xlabel('Liczba epizodów')
    plt.ylabel('Zdyskontowana nagroda')
    plt.legend()

    plt.savefig('plots/lifetime-rewards.pdf')
    plt.show()

def plot_mean_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Liczba epizodów')
    plt.ylabel('Zdyskontowana nagroda')

    plt.savefig('plots/mean-rewards.pdf')
    plt.show()

def train(n_steps):
    rewards = []

    lpg = LPG()
    lpg.lpg.load_state_dict(torch.load('results/lpg-trained.model'))
    lpg.training = False

    start = time.time()

    env = get_tabular_grid('dense')

    agent = A2C(env)
    rewards, returns = agent.train(n_steps)

    pickle.dump(rewards, open('results/rewards-a2c.pickle', 'wb'))
    pickle.dump(returns, open('results/returns-a2c.pickle', 'wb'))

    agent2 = LPGAgent(lpg, env)
    rewards2, returns2 = agent2.train(n_steps)

    pickle.dump(rewards2, open('results/rewards-lpg.pickle', 'wb'))
    pickle.dump(returns2, open('results/returns-lpg.pickle', 'wb'))

    elapsed = time.time() - start
    print('Mean rewards:', np.mean(rewards))
    print(f'Elapsed time: {elapsed // 3600:.0f}h {(elapsed // 60) % 60:.0f}m {elapsed % 60:.3f}s')
    plot_lifetime_rewards([returns2, returns], window=20)


def main():
    train(3000000)

if __name__ == '__main__':
    main()
