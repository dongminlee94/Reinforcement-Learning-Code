import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_size, action_size, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policies = self.fc3(x)
        return policies

class Critic(nn.Module):
    def __init__(self, state_size, action_size, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size + action_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        
    def forward(self, x, actions):
        x = torch.relu(self.fc1(x))
        x = torch.cat([x, actions], dim=1)
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value