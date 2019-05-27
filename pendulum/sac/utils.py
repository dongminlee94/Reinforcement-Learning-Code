import torch
import numpy as np
from torch.distributions import Normal

def init_target_model(critic, critic_target):
    critic_target.load_state_dict(critic.state_dict())

def get_action(mu, std): 
    m = Normal(mu, std)
    action = m.rsample() # reparameterization trick (mean + std * N(0,1))
    action = torch.tanh(action)
    action = action.data.numpy()
    return action

# Enforcing Action Bound
# def eval_action(mu, logstd): 
#     action = plicies.detach().numpy() + ou_noise.sample() 
#     return action

# def soft_target_update(actor, critic, actor_target, critic_target, tau):
#     soft_update(actor, actor_target, tau)
#     soft_update(critic, critic_target, tau)

# def soft_update(net, target_net, tau):
#     for param, target_param in zip(net.parameters(), target_net.parameters()):
#         target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)