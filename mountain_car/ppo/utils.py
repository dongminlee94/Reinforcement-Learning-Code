import math
import torch
from torch.distributions import Categorical


def get_action(policies):
    m = Categorical(policies)
    action = m.sample()
    action = action.data.numpy()[0]
    return action


def log_prob_density(x, policies):
    log_prob_density = -(x - policies).pow(2) / 2 \
                        - 0.5 * math.log(2 * math.pi)
    return log_prob_density