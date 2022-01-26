import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from envs import get_tabular_grid
from models import LPG, LPGAgent, A2C

def plot_lifetime_rewards(rewards, window):
    for reward in rewards:
        cumsum = np.cumsum(reward)
        avg = (cumsum[window:] - cumsum[:-window]) / window

        plt.plot(avg)

    plt.xlabel('Liczba epizodów')
    plt.ylabel('Suma nagród')
    plt.savefig('plots/lifetime-rewards.pdf')
    plt.show()

def train(n_steps):
    rewards = []

    lpg = LPG()
    lpg.lpg.load_state_dict(torch.load('results/lpg-trained.model'))
    lpg.training = False

    start = time.time()

    env = get_tabular_grid('dense')

    agent = LPGAgent(lpg, env)

    rewards = agent.train(n_steps)

    elapsed = time.time() - start
    print('Mean rewards:', np.mean(rewards))
    print(f'Elapsed time: {elapsed // 3600:.0f}h {(elapsed // 60) % 60:.0f}m {elapsed % 60:.3f}s')
    plot_lifetime_rewards([rewards], window=20)


def main():
    train(3000000)

if __name__ == '__main__':
    main()
