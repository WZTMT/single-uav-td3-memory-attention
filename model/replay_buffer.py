import numpy as np
import torch
import random
from torch.nn.utils import rnn


class ReplayBuffer(object):
    def __init__(self, capacity=int(2**17)):
        self.capacity = capacity
        self.position = 0  # 当前要存储在哪一位
        self.buffer = []  # 缓冲区
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def push(self, history, state, action, next_state, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (history, state, action, reward, next_state, done)  # 替换存入的None
        self.position = (self.position + 1) % self.capacity

    # 返回5项二维tensor，每项的一维长度都是batch_size
    # array, ndarray, ndarray, array, ndarray, array
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        history, state, action, reward, next_state, done = zip(*batch)

        return history, state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    done = [True, False, True]
    print(torch.FloatTensor(1.-np.float32(done)).unsqueeze(-1))
