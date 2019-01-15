import os
import gym
import argparse
import numpy as np

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

from memory import Memory
from ppo import train_model
from utils import get_action
from model import Actor, Critic

parser = argparse.ArgumentParser(description='PyTorch PPO')
parser.add_argument('--env_name', type=str, default="MountainCar-v0", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', type=str, default='./save_model/', help='')
parser.add_argument('--render', action="store_true", default=False, help='')
parser.add_argument('--gamma', type=float, default=0.99, help='')
parser.add_argument('--lambda', type=float, default=0.98, help='')
parser.add_argument('--critic_lr', type=float, default=3e-4, help='')
parser.add_argument('--actor_lr', type=float, default=3e-4, help='')
parser.add_argument('--l2_rate', type=float, default=1e-3, help='')
parser.add_argument('--clip_param', type=float, default=0.2, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr) 
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr, 
                              weight_decay=args.l2_rate) 

    writer = SummaryWriter(args.logdir)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])


    episodes = 0    

    for iter in range(10000):
        actor.eval(), critic.eval()
        memory = Memory()

        scores = []
        steps = 0

        while steps < 2048:
            episodes += 1

            state = env.reset()
            score = 0

            while True: 
                if args.render:
                    env.render()

                steps += 1

                state = torch.Tensor(state)
                policies = actor(state.unsqueeze(0))
                action = get_action(policies)
                next_state, reward, done, _ = env.step(action)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.push(state, action, reward, mask)
                
                score += reward
                state = next_state

                if done:
                    break
            
            # The end of the first while-loop
            scores.append(score)
        
        # The end of the second while-loop
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        writer.add_scalar('log/score', float(score_avg), iter)
        
        transitions = memory.sample()
        actor.train(), critic.train()
        train_model(actor, critic, transitions, actor_optim, critic_optim, args)


        # if iter % 100:
        #     score_avg = int(score_avg)
        #     ckpt_path = os.path.join(args.save_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

        #     torch.save({
        #         'actor': actor.state_dict(),
        #         'critic': critic.state_dict(),
        #         'args': args,
        #         'score': score_avg
        #     }, filename=ckpt_path)


if __name__ == '__main__':
    main()