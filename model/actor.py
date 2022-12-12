import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import rnn


class Actor(nn.Module):
    """
    input_dim: 输入维度，这里等于n_states
    output_dim: 输出维度，这里等于n_actions
    max_action: action的最大值
    """
    def __init__(self, n_states, n_actions, max_action, batch_size, device, init_w=3e-3):
        super(Actor, self).__init__()
        self.input_size = n_states + n_actions
        self.hidden_size = 128
        self.num_layers = 2
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.device = device

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)  # 处理历史轨迹(input_size, hidden_size, num_layers)
        self.l1 = nn.Linear(n_states, 128)  # 处理当前的状态数据
        self.l2 = nn.Linear(128, 128)  # 处理当前的状态数据
        self.l3 = nn.Linear(128, n_actions)
        self.la = nn.Linear(128, 256)  # 处理Attention，ht最外层有两层，合并之后为(256, 256, 1)
        self.max_action = max_action

        nn.init.uniform_(self.l3.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.l3.bias.detach(), a=-init_w, b=init_w)

        nn.init.uniform_(self.la.weight.detach(), a=-.1, b=.1)
        nn.init.uniform_(self.la.bias.detach(), a=-.1, b=.1)

    def forward(self, history, state):
        h0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)  # (num_layers, batch, output_size)
        c0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)  # (num_layers, batch, output_size)
        self.lstm.flatten_parameters()  # 提高显存的利用率和效率
        x1, (ht, ct) = self.lstm(history, (h0, c0))  # output(batch_size, time_step, hidden_size)
        x1, _ = rnn.pad_packed_sequence(x1, batch_first=True)  # 由packedSequence数据转换成tensor

        # Attention
        k = torch.cat((ht[0], ht[1]), dim=1).unsqueeze(-1)
        u = torch.tanh(self.la(x1))
        d = u.shape[0]
        att = torch.bmm(u, k) / math.sqrt(d)  # 每个时间步的数据都对应一个权重
        att_score = F.softmax(att, dim=1)
        scored_x1 = x1 * att_score

        state = state.unsqueeze(1)
        x2 = F.relu(self.l1(state))
        x2 = F.relu(self.l2(x2))

        x3 = torch.cat([scored_x1, x2], 1)
        action = torch.tanh(self.l3(x3))  # torch.tanh与F.tanh没有区别
        action = action[:, -1, :]

        return self.max_action * action


if __name__ == '__main__':
    actor = Actor(n_states=3 + 1 + 3 + 1 + 13, n_actions=3, max_action=1, batch_size=128, device=torch.device("cuda:0"))
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))
