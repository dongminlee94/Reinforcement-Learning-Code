# from https://github.com/reinforcement-learning-kr/reinforcement-learning-pytorch/blob/master/2-cartpole/1-dqn/train.py

import os
import sys
import gym
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import DQN
from memory import ReplayMemory
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--gamma', default=0.99, help='')
parser.add_argument('--batch_size', default=32, help='')
parser.add_argument('--initial_exploration', default=10000, help='')
parser.add_argument('--update_target', default=100, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--goal_score', default=400, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
def train_model(net, target_net, optimizer, batch, batch_size):
    # 메모리에서 배치 크기만큼 무작위로 샘플 추출
    states = torch.stack(batch.state).to(device)
    next_states = torch.stack(batch.next_state).to(device)
    actions = torch.Tensor(batch.action).long().to(device)
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)

    # 현재 상태에 대한 모델의 큐함수
    pred = net(states).squeeze(1) # squeeze - 불필요한 차원 없애기
    # 다음 상태에 대한 타깃 모델의 큐함수
    next_pred = target_net(next_states).squeeze(1)

    one_hot_action = torch.zeros(batch_size, pred.size(-1))
    # action을 one_hot_action에 뿌려주기 위해서 unsqueeze를 한다.
    # dim = 1, actions.unsqueeze(1)가 
    # 0이면 one_hot_action의 index가 0인 곳에 "1"을 뿌리고, 
    # 1이면 one_hot_action의 index가 1인 곳에 "1"을 뿌려준다.
    # 어떻게 보면 actions.unsqueeze(1)가 index라고 봐도 될 듯
    one_hot_action.scatter_(1, actions.unsqueeze(1), 1) 
    pred = torch.sum(pred.mul(one_hot_action), dim=1)

    # 벨만 최적 방정식을 이용한 업데이트 타깃
    target = rewards + masks * args.gamma * next_pred.max(1)[0]
    
    loss = F.mse_loss(pred, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 입실론 탐욕 정책으로 행동 선택
def get_action(epsilon, qvalue, num_actions):
    if np.random.rand() <= epsilon:
        return random.randrange(num_actions)
    else:
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]


# 타깃 모델을 모델의 가중치로 업데이트
def update_target_model(net, target_net):
    target_net.load_state_dict(net.state_dict())


if __name__ == "__main__":
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0] # 4
    num_actions = env.action_space.n # 2

    net = DQN(num_inputs, num_actions)
    target_net = DQN(num_inputs, num_actions)
    update_target_model(net, target_net)
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter('logs')
