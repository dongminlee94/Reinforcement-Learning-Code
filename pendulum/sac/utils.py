import torch
import numpy as np
from torch.distributions import Normal

def get_action(mu, std): 
    m = Normal(mu, std)
    z = m.rsample() # reparameterization trick (mean + std * N(0,1))
    action = torch.tanh(z)

    return action.data.numpy()

def eval_action(mu, std, epsilon=1e-6):
    m = Normal(mu, std)
    z = m.rsample() # reparameterization trick (mean + std * N(0,1))
    action = torch.tanh(z)

    # Enforcing Action Bound
    log_prob = m.log_prob(z)
    log_prob -= torch.log(1 - action.pow(2) + epsilon)
    log_prob = log_prob.sum(1, keepdim=True)

    return action, log_prob

def hard_target_update(net, target_net):
    target_net.load_state_dict(net.state_dict())

def soft_target_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)