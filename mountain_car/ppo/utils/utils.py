import math
import torch
from torch.distributions import Categorical

def get_action(policies):
    m = Categorical(policies)
    action = m.sample()
    action = action.data.numpy()[0]
    return action

def save_checkpoint(state, filename):
    torch.save(state, filename)