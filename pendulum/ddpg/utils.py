import torch
import numpy as np

class OUNoise:
    def __init__(self, action_size, theta, mu, sigma):
        self.action_size = action_size
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.X = np.zeros(self.action_size) 

    def reset(self):
        self.X = np.zeros(self.action_size)

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

def init_target_model(actor, critic, actor_target, critic_target):
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

def get_action(actor, state, ou_noise):
    state = torch.from_numpy(state).float()
    model_action = actor(state).detach().numpy() 
    action = model_action + ou_noise.sample() 
    return action

def soft_target_update(actor, critic, actor_target, critic_target, tau):
    soft_update(actor, actor_target, tau)
    soft_update(critic, critic_target, tau)

def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)