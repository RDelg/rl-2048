import cv2
import numpy as np
from PIL import Image

class Game2048(object):
    def __init__(self, n, dims):
        assert n > 3, "n must be greater than 3"
        assert dims > 1, "dims must be greater than 1"
        self.n = n
        self.dims = dims
        self.n_news = ((n**self.dims)//2**4)*2
        self.shape = [n for i in range(self.dims)]
        self.actions = [(dim, 0) for dim in range(self.dims)] + [(dim, 1) for dim in range(self.dims)]
        self.reset()
    
    def reset(self):
        self.board = np.zeros(shape=self.shape)
        self.next_board = np.zeros(shape=self.shape)
        self.score = 0
        self.add_numbers()
        return self.board
    
    def add_numbers(self):
        idx_zero = np.argwhere((self.board == 0))
        if idx_zero.shape[0] > self.n_news:
            insert_idx = idx_zero[np.argsort(np.random.uniform(size=idx_zero.shape[0]))[:self.n_news], :]
        elif idx_zero.shape[0] > 0 and idx_zero.shape[0] <= self.n_news:
            insert_idx = idx_zero
        else:
            return 1
        np.put(self.board, np.ravel_multi_index(insert_idx.T, self.shape), 2, mode='raise')
        return 0
    
    def step(self, action_n):
        assert action_n < self.dims * 2, 'invalid action number'
        score = self.score
        action = self.actions[action_n]
        self.next_board = np.copy(self.board)
        if action[1]:
            it_list = zip(range(self.n-1, 0, -1), range(self.n-2, -1, -1))
        else:
            it_list = zip(range(self.n-1), range(1,self.n))
        for i, next_i in it_list:
            index_base = np.ones(shape=[1 for x in range(self.dims)], dtype=np.int8)
            xi = np.take_along_axis(self.next_board, index_base*i, axis=action[0])
            xiplus = np.take_along_axis(self.next_board, index_base*next_i, axis=action[0])
            with np.nditer([xi, xiplus], flags=[], op_flags=[['readwrite'], ['readwrite']]) as it:
                for j, next_j in it:
                    if j[...] != 0 and next_j[...] == 0:
                        next_j[...] = j[...]
                        j[...] = 0
                    elif j[...] == next_j[...]:
                        next_j *= 2
                        j[...] = 0
                        score += next_j[...]
            np.put_along_axis(self.next_board, index_base*i, xi, axis=action[0])
            np.put_along_axis(self.next_board, index_base*next_i, xiplus, axis=action[0])
        self.board = self.next_board
        diff_score = score - self.score
        self.score = score
        is_final = self.add_numbers()
        if is_final:
            return self.board, self.score, is_final
        else:
            return self.board, diff_score, is_final
    
    def show(self, video=None):       
        n = self.n
        block_size = 50
        
        img = (self.board * (np.iinfo(np.uint8).max/np.max(self.board))).astype(np.uint8)
        img = Image.fromarray(img, "L")
        img_reshaped = img.resize((n * block_size, n * block_size), resample=Image.BOX)
        img_reshaped = np.float32(img_reshaped)
        text_color = (200, 20, 220)
        for i in range(n):
            for j in range(n):
                block_value = str(int(self.board[i, j]))
                cv2.putText(
                    img_reshaped,
                    block_value,
                    (j*block_size+int(block_size / 2)-10*len(block_value), i*block_size+int(block_size/2)+10),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=text_color
                )
        if video is not None:
            video.write(np.uint8(img_reshaped))
        else:
            cv2.imshow('2048',img_reshaped)