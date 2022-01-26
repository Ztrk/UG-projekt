from typing import List

import numpy as np
from numpy.random import default_rng
from gym import Env, spaces


class Object:
    def __init__(self, reward, eps_term, eps_respawn, position=(0, 0)):
        self.position = position
        self.reward = reward
        self.eps_term = eps_term
        self.eps_respawn = eps_respawn
        self.is_present = True


class TabularGrid(Env):
    objects: List[Object]
    EMPTY = -1
    WALL = -2

    def __init__(self, size, objects: List[Object], max_steps, is_random=False, map=None, collect_actions=False):
        self.is_random = is_random
        self.size = np.array(size)
        self.grid = np.full(size, self.EMPTY)
        if map is not None:
            self.grid = map

        self.rng = default_rng()
        self.objects = objects
        # Init positions of the objects
        positions = self.rng.choice(np.arange(size[0] * size[1]), size=len(objects), replace=False)
        positions = [self.index_to_pos(p) for p in positions]

        for i, obj in enumerate(self.objects):
            obj.position = positions[i]
            self.grid[obj.position] = i

        self.max_steps = max_steps
        self.steps = 0
        self.agent_position = np.array(self.get_empty_position())
        self.collect_actions = collect_actions

        self.action_space = spaces.Discrete(18 if collect_actions else 9)
        self.observation_space = spaces.Discrete(2**len(objects) * size[0] * size[1])
    
    def index_to_pos(self, index):
        return (index // self.size[1], index % self.size[1])

    def state_to_index(self):
        presence = np.sum([(1 - obj.is_present) * 2**i for i, obj in enumerate(self.objects)])
        return presence * self.size[1] * self.size[0] + self.agent_position[0] * self.size[1] + self.agent_position[1]
    
    def get_empty_position(self):
        pos = self.rng.integers(0, self.size[0] * self.size[1])
        pos = self.index_to_pos(pos)
        while self.grid[pos] != self.EMPTY:
            pos = self.rng.integers(0, self.size[0] * self.size[1])
            pos = self.index_to_pos(pos)
        return pos

    def step(self, action: int):
        collect_action = True
        move_action = True
        if self.collect_actions:
            collect_action = action >= 9
            move_action = not collect_action
            action = action % 9
        delta = np.array([action // 3 - 1, action % 3 - 1])
        new_pos = np.maximum(np.minimum(self.agent_position + delta, self.size - 1), [0, 0])
        if move_action:
            self.agent_position = new_pos

        # Collect objects
        reward = 0
        done = False
        index = new_pos[0], new_pos[1]
        if self.grid[index] >= 0 and collect_action:
            cur_object = self.objects[self.grid[index]]
            reward = cur_object.reward
            done = self.rng.random() < cur_object.eps_term
            self.grid[index] = self.EMPTY
            cur_object.is_present = False

        # Respawn objects
        for i, obj in enumerate(self.objects):
            if self.grid[obj.position[0], obj.position[1]] != i and self.rng.random() < obj.eps_respawn:
                # Randomize new position
                if self.is_random:
                    obj.position = self.get_empty_position()
                self.grid[obj.position[0], obj.position[1]] = i
                obj.is_present = True

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self.state_to_index(), reward, done, None

    def reset(self):
        if self.is_random:
            # Randomize new positions of the objects
            positions = self.rng.choice(np.arange(self.size[0] * self.size[1]), size=len(self.objects), replace=False)
            positions = [self.index_to_pos(p) for p in positions]

            # Remove old position and set new one
            for i, obj in enumerate(self.objects):
                self.grid[obj.position] = self.EMPTY
                obj.position = positions[i]

        for i, obj in enumerate(self.objects):
            self.grid[obj.position] = i
            obj.is_present = True

        self.steps = 0
        self.agent_position = np.array(self.get_empty_position())
        return self.state_to_index()

    def render(self, mode="human"):
        print(self.grid)


def get_tabular_grid(env_type, is_random=False, collect_actions=False):
    if not is_random:
        if env_type == 'dense':
            objects = [Object(1, 0, 0.05), Object(1, 0, 0.05), Object(-1, 0.5, 0.1), Object(-1, 0, 0.5)]
            return TabularGrid((11, 11), objects, max_steps=500, collect_actions=collect_actions)
        elif env_type == 'small-sparse':
            objects = [Object(1, 1, 1), Object(-1, 1, 1), Object(-1, 1, 1)]
            return TabularGrid((5, 7), objects, max_steps=50, collect_actions=collect_actions)
        elif env_type == 'very-small':
            objects = [Object(1, 1, 1), Object(-1, 1, 1)]
            return TabularGrid((3, 3), objects, max_steps=10, collect_actions=collect_actions)
        raise ValueError('Wrong evironment type')
    else:
        if env_type == 'dense':
            objects = [Object(1, 0, 0.05), Object(1, 0, 0.05), Object(-1, 0.5, 0.1), Object(-1, 0, 0.5)]
            return TabularGrid((11, 11), objects, max_steps=500, is_random=True, collect_actions=collect_actions)
        raise ValueError('Wrong environment type')


def main():
    env = get_tabular_grid('dense')

    observation = env.reset()
    env.render()
    for i in range(10):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        print(observation, reward, done)
        print(env.agent_position)
        print(env.grid)
        if done:
            break


if __name__ == '__main__':
    main()
