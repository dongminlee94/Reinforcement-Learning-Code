import os
import gym
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

from model import Actor
from tnpg import train_model
from utils.utils import get_action
from utils.running_state import ZFilter

parser = argparse.ArgumentParser(description='PyTorch NPG')
parser.add_argument('--env_name', type=str, default="Hopper-v2")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    print('state size:', state_size) 
    print('action size:', action_size) 

    actor = Actor(state_size, action_size, args)
    # writer = SummaryWriter(args.logdir)
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    running_state = ZFilter((state_size,), clip=5)
    episodes = 0    

    for iter in range(2000):
        memory = deque()
        scores = []
        steps = 0

        while steps < 2048: 
            score = 0
            episodes += 1
            
            state = env.reset()
            state = running_state(state)
            
            for _ in range(10000): 
                if args.render:
                    env.render()

                steps += 1

                mu, std = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                next_state = running_state(next_state)
                state = next_state
                score += reward

                if done:
                    break
            
            scores.append(score)
        
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        # writer.add_scalar('log/score', float(score_avg), iter)

        actor.train()
        train_model(actor, memory, args)

        if iter % 100:
            ckpt_path = args.save_path + str(score_avg) + 'model.pth'
            torch.save(actor.state_dict(), ckpt_path)