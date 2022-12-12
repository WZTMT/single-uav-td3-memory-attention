import os
import sys

import torch
import numpy as np
import datetime
import airsim
import math

from model.td3 import TD3
from utils import save_results, make_dir, plot_rewards
from torch.utils.tensorboard import SummaryWriter
from env.multirotor import Multirotor

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

# curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
curr_time = "20221003-211610"


class TD3Config:
    def __init__(self) -> None:
        self.algo_name = 'TD3 with LSTM'  # 算法名称
        self.env_name = 'UE4 and Airsim'  # 环境名称
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 4000  # 训练的回合数
        self.test_eps = 150
        self.epsilon_start = 50  # Episodes initial random policy is used 增大样本的多样性
        self.max_step = 1000  # Max time steps to run environment
        self.expl_noise = 0.15  # Std of Gaussian exploration noise
        self.batch_size = 256  # Batch size for both actor and critic
        self.gamma = 0.98  # gamma factor
        self.tau = 0.0005  # soft update
        self.policy_noise = 0.2  # Std of Gaussian policy noise
        self.noise_clip = 0.3  # Range to clip target policy noise
        self.policy_freq = 3  # Frequency of delayed policy updates
        self.memory_capacity = 2**17
        self.update_times = 1
        self.n_states = 3+1+3+1+13
        self.n_actions = 3
        self.max_action = 1.0
        self.memory_window_length = 10  # 记忆窗口的长度
        self.seed = 1
        self.result_path = curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/models/'  # 保存模型的路径
        self.eval_model_path = curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/eval_models/'  # 保存验证得到的模型的路径
        self.save_fig = True  # 是否保存图片


def eval(cfg, client, agent):
    env = Multirotor(client, True)
    state = env.get_state()
    finish_step = 0
    ep_reward = 0
    final_distance = state[3] * env.max_distance
    history = []  # 记录每一个episode的历史轨迹
    for i_step in range(cfg.max_step):
        finish_step = finish_step + 1
        action = agent.choose_action(history, state)
        next_state, reward, done = env.step(action)

        sa_pair = np.append(state, action)
        history.append(sa_pair)  # 保存的历史轨迹为当前step之前的20步
        if len(history) > cfg.memory_window_length + 1:  # 仅保留窗口大小的数据
            history = history[1:cfg.memory_window_length + 1]
        state = next_state
        ep_reward += reward
        print('\r----Eval: Step: {}\tReward: {:.2f}\tDistance: {:.2f}'.format(i_step + 1, ep_reward,
                                                                                 state[3] * env.max_distance), end="")
        final_distance = state[3] * env.max_distance
        if done:
            break
    print('\r----Eval: Finish step: {}\tReward: {:.2f}\tFinal distance: {:.2f}'.format(finish_step,
                                                                                          ep_reward, final_distance))
    return ep_reward


def train(cfg, client, agent):
    print('Start Training!')
    print(f'Env：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    writer = SummaryWriter('./train_image')
    best_eval_reward = -math.inf
    critic_loss = 0
    for i_ep in range(int(cfg.train_eps)):
        is_full = True
        if i_ep > 5000 and np.random.rand() < 0.5:  # 逐渐增加障碍物较多的样本的数量，大概在2300episode左右达到一半
            is_full = False
        env = Multirotor(client, is_full)
        state = env.get_state()
        ep_reward = 0
        finish_step = 0
        final_distance = state[3] * env.max_distance
        history = []  # 记录每一个episode的历史轨迹
        for i_step in range(cfg.max_step):
            finish_step = finish_step + 1
            # Select action randomly or according to policy
            if i_ep < cfg.epsilon_start:
                action = np.random.uniform(-cfg.max_action, cfg.max_action, size=cfg.n_actions)
            else:
                action = (
                    agent.choose_action(history, state) +
                    np.random.normal(0, cfg.expl_noise, size=cfg.n_actions)
                ).clip(-cfg.max_action, cfg.max_action)
            next_state, reward, done = env.step(action)
            done = np.float32(done)
            agent.memory.push(history, state, action, next_state, reward, done)
            sa_pair = np.append(state, action)
            history.append(sa_pair)  # 保存的历史轨迹为当前step之前的20步
            if len(history) > cfg.memory_window_length:  # 仅保留窗口大小的数据
                history = history[1:cfg.memory_window_length+1]
            state = next_state
            ep_reward += reward

            # 样本量足够大时，可以在一个step内更新两次
            # replay_len = len(agent.memory)
            # k = 1 + replay_len / cfg.memory_capacity
            # update_times = int(k * cfg.update_times)
            # for _ in range(update_times):
            #     if i_ep + 1 >= cfg.epsilon_start:
            #         agent.update()

            critic_loss = agent.update()
            print('\rEpisode: {}\tStep: {}\tReward: {:.2f}\tDistance: {:.2f}'.format(i_ep + 1, i_step + 1, ep_reward, state[3] * env.max_distance), end="")
            final_distance = state[3] * env.max_distance
            if done:
                break
        print('\rEpisode: {}\tFinish step: {}\tAverage Reward: {:.2f}\tFinal distance: {:.2f}'.format(i_ep + 1, finish_step, ep_reward, final_distance))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * rewards[-1])
        else:
            ma_rewards.append(rewards[-1])
        writer.add_scalars(main_tag='train',
                           tag_scalar_dict={
                               'reward': rewards[-1],
                               'ma_reward': ma_rewards[-1]
                           },
                           global_step=i_ep)
        writer.add_scalar(tag='critic_loss', scalar_value=critic_loss, global_step=i_ep)
        if (i_ep + 1) % 10 == 0:
            agent.save(path=cfg.model_path)
            eval_reward = eval(cfg, client, agent)  # 验证模型
            if eval_reward > best_eval_reward:
                agent.save(path=cfg.eval_model_path)
                best_eval_reward = eval_reward
        if (i_ep + 1) == cfg.train_eps:
            env.land()
    print('Finish Training!')
    writer.close()
    return rewards, ma_rewards


def set_seed(seed):
    """
    全局生效
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    cfg = TD3Config()
    set_seed(cfg.seed)
    make_dir(cfg.result_path, cfg.model_path, cfg.eval_model_path)
    client = airsim.MultirotorClient()  # connect to the AirSim simulator
    agent = TD3(cfg)
    rewards, ma_rewards = train(cfg, client, agent)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, cfg, tag="train")
