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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(actor_critic, optimizer, transition, policy, value):
    state, next_state, action, reward, mask = transition
    state = torch.Tensor(state).to(device)
    next_state = torch.Tensor(next_state).to(device)
    
    criterion = torch.nn.MSELoss()

    log_policy = torch.log(policy[0])[action]

    _, next_value = actor_critic(next_state)
    q_value = reward + mask * args.gamma * next_value[0]
    advantage = q_value - value[0]
    
    actor_loss = -log_policy * advantage.item()
    critic_loss = criterion(value[0], q_value.detach())

    loss = actor_loss + critic_loss
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

    actor_critic = ActorCritic(state_size, action_size, args).to(device)
    actor_critic.train()
    
    optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)
    writer = SummaryWriter(args.logdir)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

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
            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1
            
            transition = [state, next_state, action, reward, mask]
            train_model(actor_critic, optimizer, transition, policies, value)

            state = next_state
            score += reward

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score

        if episode % args.log_interval == 0:
            print('{} episode | running_score: {:.2f}'.format(episode, running_score))
            writer.add_scalar('log/score', float(score), running_score)

        if running_score > args.goal_score:
            ckpt_path = args.save_path + 'model.pth'
            torch.save(actor_critic.state_dict(), ckpt_path)
            print('Running score exceeds 400. So end')
            break  


if __name__=="__main__":
    main()
    