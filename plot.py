import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot_lifetime_rewards(rewards, labels, window):
    plt.figure(figsize=(3.2, 2.4))
    for reward, label in zip(rewards, labels):
        cumsum = np.cumsum(reward)
        avg = (cumsum[window:] - cumsum[:-window]) / window

        plt.plot(avg, label=label)

    plt.tight_layout(pad=2)
    plt.xlabel('Liczba epizodów')
    plt.ylabel('Zdyskontowana nagroda')
    plt.legend()

    # plt.savefig('plots/lifetime-rewards.png', dpi=300)
    plt.show()

def main():
    with open('results/returns-a2c.pickle', 'rb') as f:
        rewards_a2c = pickle.load(f)
    
    with open('results/returns-lpg.pickle', 'rb') as f:
        rewards_lpg = pickle.load(f)

    plot_lifetime_rewards([rewards_lpg, rewards_a2c], ['LPG', 'A2C'], window=100)

if __name__ == '__main__':
    main()
