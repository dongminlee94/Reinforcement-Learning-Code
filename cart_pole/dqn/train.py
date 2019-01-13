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


def main():
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

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    net.to(device)
    target_net.to(device)
    net.train()
    target_net.train()
    memory = ReplayMemory(10000)
    running_score = 0
    epsilon = 1.0
    steps = 0
    
    for e in range(3000):
        done = False
        
        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            if args.render:
                env.render()

            steps += 1
            qvalue = net(state)
            action = get_action(epsilon, qvalue, num_actions)
            next_state, reward, done, _ = env.step(action)
            
            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)
            
            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1
            memory.push(state, next_state, action, reward, mask)

            score += reward
            state = next_state

            if steps > args.initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(args.batch_size)
                train_model(net, target_net, optimizer, batch, args.batch_size)

                if steps % args.update_target:
                    update_target_model(net, target_net)

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % args.log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))
            writer.add_scalar('log/score', float(score), running_score)

        if running_score > args.goal_score:
            ckpt_path = args.save_path + 'model.pth'
            torch.save(net.state_dict(), ckpt_path)
            print('running score exceeds 400 so end')
            break  

if __name__ == '__main__':
    main()