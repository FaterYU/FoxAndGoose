from time import sleep
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, n_state, n_hidden, n_action):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x


class ValueNet(nn.Module):
    def __init__(self, n_state, n_hidden):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class PPO():
    def __init__(self, n_state, n_hidden, n_action, epochs, actor_lr=1e-4, critic_lr=1e-4, lmbda=0.95, eps=0.2, gamma=0.99, epsilon=0.2, device="cuda"):

        self.actor = PolicyNet(n_state, n_hidden, n_action)
        self.actor.to(device)
        self.critic = ValueNet(n_state, n_hidden)
        self.critic.to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)

        self.epochs = epochs
        self.eps = eps
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

        self.record = {
            "actor_loss": [],
            "critic_loss": []
        }

    def get_action(self, state, action_space, mask):
        state = torch.tensor(
            state, dtype=torch.float32, device=self.device)
        action_prob = self.actor(state)
        action_prob_mix = action_prob * \
            torch.tensor(mask, dtype=torch.float32, device=self.device)
        action = torch.argmax(action_prob_mix).item()
        if mask[action] == 0:
            action = self.get_action_random(action_space, mask)
        return action

    def get_action_random(self, action_space, mask):
        action = action_space.sample(mask)
        return action

    def train(self, transition_dict):
        self.record = {
            "actor_loss": [],
            "critic_loss": []
        }
        # 提取数据集
        states = torch.tensor(
            transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(
            self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(
            self.device).view(-1, 1)
        next_states = torch.tensor(
            transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(
            self.device).view(-1, 1)

        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1-dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value

        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()

        # 将 advantage_list 转换为 NumPy 数组
        advantage_array = np.array(advantage_list, dtype=np.float32)

        # 将 NumPy 数组转换为 PyTorch 张量
        advantage = torch.tensor(advantage_array).to(self.device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(
            states).gather(1, actions)).detach()

        # 一组数据训练 epochs 轮
        # for _ in tqdm(range(self.epochs)):
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(
                self.critic(states), td_target.detach()))

            # 梯度清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # 记录损失
            actor_loss = actor_loss.item()
            critic_loss = critic_loss.item()
            self.record["actor_loss"].append(actor_loss)
            self.record["critic_loss"].append(critic_loss)

    def save(self, path):
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")
        np.save(f"{path}/loss.npy", self.record)

        # save loss figure
        plt.figure()
        plt.plot(self.record["actor_loss"], label="actor_loss")
        plt.legend()
        plt.savefig(f"{path}/loss_actor.png")
        plt.close()

        plt.figure()
        plt.plot(self.record["critic_loss"], label="critic_loss")
        plt.legend()
        plt.savefig(f"{path}/loss_critic.png")
        plt.close()

    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))
        self.record = np.load(f"{path}/loss.npy", allow_pickle=True).item()
        return self.record
