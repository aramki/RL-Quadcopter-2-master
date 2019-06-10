import random
from collections import deque, namedtuple

class ReplayBuffer:
    """
        Buffer to store experience tuples
    """
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.exp = namedtuple("Exp", field_names=["state", "action", "reward", "next_state", "done"])

    def sample(self, batch_size=64):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        ex = self.exp(state, action, reward, next_state, done)
        self.memory.append(ex)