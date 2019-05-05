import os
import gym
import random
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from utils import *
from model import Actor, Critic
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Pendulum-v0")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--theta', type=float, default=0.15)
parser.add_argument('--mu', type=float, default=0.0)
parser.add_argument('--sigma', type=float, default=0.2)
parser.add_argument('--max_iter_num', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--goal_score', type=int, default=-300)
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(actor, critic, actor_target, critic_target, 
                actor_optimizer, critic_optimizer, minibatch):
    minibatch = np.array(minibatch).transpose()
    states = np.vstack(minibatch[0])
    actions = np.vstack(minibatch[1])
    rewards = np.vstack(minibatch[2])
    next_states = np.vstack(minibatch[3])
    dones = np.vstack(minibatch[4].astype(int))

    states = torch.Tensor(states).to(device)
    actions = torch.Tensor(actions).to(device)
    rewards = torch.Tensor(rewards).to(device)
    next_states = torch.Tensor(next_states).to(device)
    dones = torch.Tensor(dones).to(device)

    # critic update
    criterion = torch.nn.MSELoss()

    pred = critic(states, actions)
    
    next_actions = actor_target(next_states)
    next_pred = critic_target(next_states, next_actions)
    target = rewards + (1 - dones) * args.gamma * next_pred
    
    critic_loss = criterion(pred, target)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # actor update
    pred_actions = actor(states)
    actor_loss = critic(states, pred_actions).mean()

    actor_loss = -actor_loss
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    
def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    print('state size:', state_size)
    print('action size:', action_size)
    
    actor = Actor(state_size, action_size, args).to(device)
    actor_target = Actor(state_size, action_size, args).to(device)
    critic = Critic(state_size, action_size, args).to(device)
    critic_target = Critic(state_size, action_size, args).to(device)
    
    actor.train(), critic.train()
    actor_target.train(), critic_target.train()

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
    writer = SummaryWriter(args.logdir)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # initialize target model
    init_target_model(actor, critic, actor_target, critic_target)

    ou_noise = OUNoise(action_size, args.theta, args.mu, args.sigma)
    memory = deque(maxlen=10000)
    recent_rewards = deque(maxlen=100)
    steps = 0

    for episode in range(args.max_iter_num):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if args.render:
                env.render()

            steps += 1

            action = get_action(actor, state, ou_noise)
            next_state, reward, done, _ = env.step([action])
            next_state = np.reshape(next_state, [1, state_size])
            reward = float(reward[0, 0])

            memory.append((state, action, reward, next_state, done))

            state = next_state
            score += reward

            if steps > args.batch_size:
                minibatch = random.sample(memory, args.batch_size)
                train_model(actor, critic, actor_target, critic_target, 
                            actor_optimizer, critic_optimizer, minibatch)
                soft_target_update(actor, critic, actor_target, critic_target, args.tau)

            if done:
                recent_rewards.append(score)

        if episode % args.log_interval == 0:
            print('{} episode | score_avg: {:.2f}'.format(episode, np.mean(recent_rewards)))
            # writer.add_scalar('log/score', float(score), score_avg)

        if np.mean(recent_rewards) > args.goal_score:
            ckpt_path = args.save_path + 'model.pth'
            torch.save({
                'actor': actor.state_dict(), 
                'critic': critic.state_dict()}, ckpt_path)
            print('Recent rewards exceed -300. So end')
            break  

if __name__ == '__main__':
    main()