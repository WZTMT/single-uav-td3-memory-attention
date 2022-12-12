import torch
import copy
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import rnn
from model.actor import Actor
from model.critic import Critic
from model.replay_buffer import ReplayBuffer


def to_packed_sequence(history):
    """
    由[[ndarray]]类型的数据转为PackedSequence类型的数据
    """
    # 将原本[ndarray]改为[tensor]
    his = []
    lens = []
    for h in history:
        his.append(torch.FloatTensor(np.array(h)))
        lens.append(his[-1].shape[0])
    lens = torch.Tensor(lens)
    # 用0补全，得到一个tensor类型的batch
    his = rnn.pad_sequence(his, batch_first=True, padding_value=0)
    # 改装成PackedSequence类型数据，去掉多余的0，可直接输入网络进行训练
    his = rnn.pack_padded_sequence(his, lens, batch_first=True, enforce_sorted=False)

    return his


class TD3(object):
    def __init__(self, cfg):
        self.max_action = cfg.max_action
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise  # 策略平滑正则化：加在目标策略上的噪声，用于防止critic过拟合
        self.noise_clip = cfg.noise_clip  # 噪声的最大值
        self.policy_freq = cfg.policy_freq  # 策略网络延迟更新，更新频率
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.n_states = cfg.n_states
        self.n_actions = cfg.n_actions
        self.total_iteration = 0  # 模型的更新次数

        self.actor = Actor(cfg.n_states, cfg.n_actions, self.max_action, cfg.batch_size, self.device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)  # 将一个网络赋值给另一个网络，且不相互影响
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(cfg.n_states, cfg.n_actions, cfg.batch_size, self.device).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.memory = ReplayBuffer(capacity=cfg.memory_capacity)

    '''
    根据一次的状态选出一次动作
    '''

    def choose_action(self, history, state):
        # 如果history中没有sa_pair，则装入一个全零的向量
        if len(history) == 0:
            mask = np.zeros(self.n_states + self.n_actions)
            history.append(mask)
        batch = [history]
        his = to_packed_sequence(batch).to(self.device)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action = self.actor(his, state)
        action = action.cpu().data.numpy().flatten()  # flatten()将高维数组展成一维向量
        return action

    def update(self):
        critic_loss = 0
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return critic_loss

        self.total_iteration += 1

        # Sample replay buffer 取一个batch_size的数据
        history, state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        # 由tuple转为list
        history = list(history)
        state = list(state)
        action = list(action)

        # 得到一个batch的next_history
        # next_history = history.copy()
        # 使用copy()复制只有第一层是相互独立的，剩余的几层都是对地址的复制，不是深度复制
        next_history = [[i for i in j] for j in history]
        sa_pair_matrix = np.append(state, action, axis=1)
        sa_pair = []
        for i in np.vsplit(sa_pair_matrix, self.batch_size):
            sa_pair.append(i.flatten())
        for i in range(len(next_history)):
            next_history[i].append(sa_pair[i])

        his = to_packed_sequence(history).to(self.device)
        next_his = to_packed_sequence(next_history).to(self.device)
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)
        not_done = torch.FloatTensor(1. - np.float32(done)).unsqueeze(-1).to(self.device)

        # 所有在该模块下计算出的tensor的required_grad都为false，都不会被求导
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # torch.randn_like()返回一个均值为0，方差为1的高斯分布的tensor
            # noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            noise = torch.FloatTensor(np.random.normal(0, self.policy_noise, size=self.n_actions).clip(-self.noise_clip, self.noise_clip)).to(self.device)

            next_action = self.actor_target(next_his, next_state)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_his, next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.gamma * target_q  # 时序差分目标函数

        # Get current Q estimates
        current_q1, current_q2 = self.critic(his, state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_iteration % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.q1(his, state, self.actor(his, state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss

    def save(self, path):
        torch.save(self.critic.state_dict(), path + "td3_critic")
        torch.save(self.critic_optimizer.state_dict(), path + "td3_critic_optimizer")

        torch.save(self.actor.state_dict(), path + "td3_actor")
        torch.save(self.actor_optimizer.state_dict(), path + "td3_actor_optimizer")

    def load(self, path):
        self.critic.load_state_dict(torch.load(path + "td3_critic"))
        self.critic_optimizer.load_state_dict(torch.load(path + "td3_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(path + "td3_actor"))
        self.actor_optimizer.load_state_dict(torch.load(path + "td3_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


if __name__ == '__main__':
    action = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    noise = torch.randn_like(action).clamp(-.5, .5)
    print(noise)
