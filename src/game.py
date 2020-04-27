import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep

class Env2048(py_environment.PyEnvironment):
  """
  """

  def __init__(self, n: int, dims: int, discount=1.0):
    """Initializes Env2048.
    Args:
      rng: size of the grid.
      dims: number of dims.
      discount: Discount for reward.
    """
    super(Env2048, self).__init__()
    self._n = n
    self._dims = dims
    self._discount = np.asarray(discount, dtype=np.float32)
    self._n_news = ((n**self._dims)//2**4)*2
    self._shape = tuple([self._n for i in range(self._dims)])
    self._actions = [(dim, 0) for dim in range(self._dims)] + [(dim, 1) for dim in range(self._dims)]
    self._states = None
    self._total_score = 0

  def action_spec(self):
    return BoundedArraySpec((1,), np.int32, minimum=[0], maximum=[self._dims*2 - 1])

  def observation_spec(self):
    return BoundedArraySpec([self._n for i in range(self._dims)], np.float32, minimum=0., maximum=1.)

  def _reset(self):
    np.random.seed(123)
    self._total_score = 0.
    self._states = np.zeros(self._shape, np.float32)
    self._add_numbers()
    return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                    self._discount, self._states)

  #def _legal_actions(self, states: np.ndarray):
  #  return list(zip(*np.where(states == 0)))

  def _add_numbers(self):
    idx_zero = np.argwhere((self._states == 0))
    if idx_zero.shape[0] > self._n_news:
      insert_idx = idx_zero[np.argsort(np.random.uniform(size=idx_zero.shape[0]))[:self._n_news], :]
    elif idx_zero.shape[0] > 0 and idx_zero.shape[0] <= self._n_news:
      insert_idx = idx_zero
    else:
      return 1
    np.put(self._states, np.ravel_multi_index(insert_idx.T, self._shape), np.float32(2/2048), mode='raise')
    return 0

  def get_state(self) -> TimeStep:
    return self._current_time_step

  def set_state(self, time_step: TimeStep):
    self._current_time_step = time_step
    self._states = time_step.observation

  def _step(self, action: np.ndarray):
    if self._current_time_step.is_last():
      return self._reset()
    action = self._actions[action[0]]
    score = 0.
    if action[1]:
        it_list = zip(range(self._n-1, 0, -1), range(self._n-2, -1, -1))
    else:
        it_list = zip(range(self._n-1), range(1,self._n))
    for i, next_i in it_list:
      index_base = np.ones(shape=[1 for x in range(self._dims)], dtype=np.int32)
      xi = np.take_along_axis(self._states, index_base*i, axis=action[0])
      xiplus = np.take_along_axis(self._states, index_base*next_i, axis=action[0])
      with np.nditer([xi, xiplus], flags=[], op_flags=[['readwrite'], ['readwrite']]) as it:
        for j, next_j in it:
          if j[...] != 0. and next_j[...] == 0.:
            next_j[...] = j[...]
            j[...] = 0.
          elif j[...] == next_j[...]:
            next_j *= 2.
            j[...] = 0.
            score += next_j[...]
      np.put_along_axis(self._states, index_base*i, xi, axis=action[0])
      np.put_along_axis(self._states, index_base*next_i, xiplus, axis=action[0])
    self._total_score += score
    is_final = self._add_numbers()

    return TimeStep(
        StepType.LAST if is_final else StepType.MID,
        np.asarray(self._total_score, dtype=np.float32) if is_final else np.asarray(0., dtype=np.float32),
        self._discount,
        self._states
    )
    
    # def show(self, video=None):       
    #     n = self.n
    #     block_size = 50
        
    #     img = (self.board * (np.iinfo(np.uint8).max/np.max(self.board))).astype(np.uint8)
    #     img = Image.fromarray(img, "L")
    #     img_reshaped = img.resize((n * block_size, n * block_size), resample=Image.BOX)
    #     img_reshaped = np.float32(img_reshaped)
    #     text_color = (200, 20, 220)
    #     for i in range(n):
    #         for j in range(n):
    #             block_value = str(int(self.board[i, j]))
    #             cv2.putText(
    #                 img_reshaped,
    #                 block_value,
    #                 (j*block_size+int(block_size / 2)-10*len(block_value), i*block_size+int(block_size/2)+10),
    #                 fontFace=cv2.FONT_HERSHEY_DUPLEX,
    #                 fontScale=1,
    #                 color=text_color
    #             )
    #     if video is not None:
    #         video.write(np.uint8(img_reshaped))
    #     else:
    #         cv2.imshow('2048',img_reshaped)