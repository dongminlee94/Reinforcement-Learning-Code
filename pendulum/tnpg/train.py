import os
import gym
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from utils import *
from model import Actor
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Pendulum-v0")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--max_iter_num', type=int, default=1000)
parser.add_argument('--total_sample_size', type=int, default=2048)
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--goal_score', type=int, default=-300)
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

def train_model(actor, memory, args):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    actions = torch.Tensor(actions).squeeze(1)
    rewards = torch.Tensor(rewards).squeeze(1)
    masks = torch.Tensor(masks)

    # ----------------------------
    # step 1: get returns
    returns = get_returns(rewards, masks, args.gamma)

    # ----------------------------
    # step 2: get gradient of loss and hessian of kl and search direction
    loss = get_loss(actor, returns, states, actions)
    loss_grad = torch.autograd.grad(loss, actor.parameters())
    loss_grad = flat_grad(loss_grad)

    search_dir = conjugate_gradient(actor, states, loss_grad.data, nsteps=10)
    
    # ----------------------------
    # step 3: update actor
    params = flat_params(actor)
    new_params = params + 0.5 * search_dir
    update_model(actor, new_params)
    

def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    print('state size:', state_size)
    print('action size:', action_size)
    
    actor = Actor(state_size, action_size, args)

    # writer = SummaryWriter(args.logdir)

    recent_rewards = deque(maxlen=100)
    episodes = 0

    for iter in range(args.max_iter_num):
        memory = deque()
        steps = 0

        while steps < args.total_sample_size:
            score = 0
            episodes += 1

            state = env.reset()
            state = np.reshape(state, [1, state_size])

            for _ in range(200):
                if args.render:
                    env.render()

                steps += 1

                mu, std = actor(torch.Tensor(state))
                action = get_action(mu, std)

                next_state, reward, done, _ = env.step(action)
                mask = 0 if done else 1

                memory.append([state, action, reward, mask])
                
                next_state = np.reshape(next_state, [1, state_size])
                state = next_state
                score += reward

                if done:
                    recent_rewards.append(score)

        if iter % args.log_interval == 0:
            print('{} iter | {} episode | score_avg: {:.2f}'.format(iter, episodes, np.mean(recent_rewards)))
            # writer.add_scalar('log/score', float(score), iter)
        
        actor.train()
        train_model(actor, memory, args)

        if np.mean(recent_rewards) > args.goal_score:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            
            ckpt_path = args.save_path + 'model.pth'
            torch.save(actor.state_dict(), ckpt_path)
            print('Recent rewards exceed -300. So end')
            break  

if __name__ == '__main__':
    main()