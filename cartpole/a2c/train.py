import os
import gym
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.distributions import Categorical

from model import ActorCritic
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--max_iter_num', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--goal_score', type=int, default=400)
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

def train_model(actor_critic, optimizer, transition, policies, value):
    state, action, next_state, reward, mask = transition
    
    criterion = torch.nn.MSELoss()

    log_policy = torch.log(policies[0])[action]

    _, next_value = actor_critic(torch.Tensor(next_state))
    q_value = reward + mask * args.gamma * next_value[0]
    advantage = q_value - value[0]
    
    actor_loss = -log_policy * advantage.item() 
    critic_loss = criterion(q_value.detach(), value[0])
    entropy = policies[0] * torch.log(policies[0])

    loss = actor_loss + critic_loss - 0.1 * entropy.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_action(policies):
    m = Categorical(policies)
    action = m.sample()
    action = action.data.numpy()[0]
    return action


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('state size:', state_size)
    print('action size:', action_size)

    actor_critic = ActorCritic(state_size, action_size, args)
    optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)

    writer = SummaryWriter(args.logdir)

    running_score = 0

    for episode in range(args.max_iter_num):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if args.render:
                env.render()

            policies, value = actor_critic(torch.Tensor(state))
            action = get_action(policies)

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -1
            
            if done:
                mask = 0
            else:
                mask = 1
            
            transition = [state, action, next_state, reward, mask]

            actor_critic.train()
            train_model(actor_critic, optimizer, transition, policies, value)

            state = next_state
            score += reward

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score

        if episode % args.log_interval == 0:
            print('{} episode | running_score: {:.2f}'.format(episode, running_score))
            writer.add_scalar('log/score', float(score), episode)

        if running_score > args.goal_score:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
    
            ckpt_path = args.save_path + 'model.pth'
            torch.save(actor_critic.state_dict(), ckpt_path)
            print('Running score exceeds 400. So end')
            break  

if __name__=="__main__":
    main()
    