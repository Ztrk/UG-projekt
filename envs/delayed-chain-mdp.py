import numpy as np
from numpy.random import default_rng
from gym import Env, spaces

class DelayedChainMDP(Env):
    def __init__(self, length, noisy=False):
        self.length = length
        self.noisy = noisy

        self.rng = default_rng()
        self.steps = 0
        self.correct_action = self.rng.integers(0, 2)
        self.chosen_action = 0

        self.action_space = spaces.Discrete(2)
        # observation - correct action, action chosen, index
        self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(length)))

    def step(self, action):
        if self.steps == 0:
            self.chosen_action = action

        self.steps += 1
        if self.steps >= self.length:
            return (self.correct_action, self.chosen_action, self.steps), 2 * (self.chosen_action == self.correct_action) - 1, True, None
        else:
            reward = 2 * self.rng.integers(0, 2) - 1 if self.noisy else 0
            return (self.correct_action, self.chosen_action, self.steps), reward, False, None

            

    def reset(self):
        self.steps = 0
        self.correct_action = self.rng.integers(0, 2)

        return (self.correct_action, self.chosen_action, self.steps)

    def render(self, mode='human'):
        print(self.correct_action, self.chosen_action, self.steps)


def main():
    env = DelayedChainMDP(5, noisy=False)

    observation = env.reset()
    env.render()
    for i in range(10):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        print(observation, reward, done)
        if done:
            break


if __name__ == '__main__':
    main()
