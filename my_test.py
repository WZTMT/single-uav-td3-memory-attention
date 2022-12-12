import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import rnn
from torch.utils.tensorboard import SummaryWriter
from model.actor import Actor


if __name__ == '__main__':
    action = torch.randn(3)
    for _ in range(100):
        print(np.random.normal(0, .3, size=3))
