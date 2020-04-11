import os

import cv2
import numpy as np

from src.game import Game2048
from src.model import DQNSolver


def test():
    n = 4
    dims = 2
    model_path = os.path.join('models', 'best_model.h5')
    env = Game2048(n=n, dims=dims)
    observation_space = env.board.ravel().shape[0]
    action_space = len(env.actions)
    model = DQNSolver(observation_space, action_space)
    model.load(model_path)
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    if dims == 2:
        video = cv2.VideoWriter(
            os.path.join('videos', f'ai_n_{n}_dims_{dims}.mp4'),
            cv2.VideoWriter_fourcc(*"DIVX"),
            5.,
            (200, 200),
            0
        )
    is_final = False
    while not is_final:
        action_n =  model.act(state)
        state, score, is_final =  env.step(action_n=action_n)
        state = np.reshape(state, [1, observation_space])
        if dims == 2:
            env.show(video=video)
    if dims == 2:
        video.release()
        env.show()

if __name__ == "__main__":
    test()
