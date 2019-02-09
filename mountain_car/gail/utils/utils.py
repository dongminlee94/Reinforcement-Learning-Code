import math
import torch
from torch.distributions import Categorical

def get_action(policies):
    m = Categorical(policies)
    action = m.sample()
    action = action.data.numpy()[0]
    return action

def get_entropy(policies):
    entropy = policies * torch.log(policies)
    return entropy.sum().mean()

def get_reward(discrim, state, action):
    state = torch.Tensor(state)
    action = torch.Tensor([action])
    state_action = torch.cat([state, action])
    with torch.no_grad():
        return -math.log(discrim(state_action).data.numpy())

def save_checkpoint(state, filename):
    torch.save(state, filename)