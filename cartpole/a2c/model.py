import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, args):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_actor = nn.Linear(args.hidden_size, action_size)
        self.fc_critic = nn.Linear(args.hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        policies = torch.softmax(self.fc_actor(x), dim=1)
        value = self.fc_critic(x)
        return policies, value