import os

import numpy as np
from tqdm import tqdm

from src.game import Game2048
from src.model import DQNSolver


def train():
    n = 4
    dims = 2
    iterations = 15
    env = Game2048(n=n, dims=dims)
    observation_space = env.board.ravel().shape[0]
    action_space = len(env.actions)
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    best_step = 0
    for _ in tqdm(range(iterations)):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            dqn_solver.experience_replay()
            state = state_next
            if terminal:
                if step > best_step:
                    best_step = step
                    dqn_solver.save(os.path.join('models','best_model.h5'))
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                break

if __name__ == "__main__":
    train()
