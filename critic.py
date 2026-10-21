import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        input_dim = state_dim + action_dim

        self.l1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.ln1(self.l1(x)))
        x = F.leaky_relu(self.ln2(self.l2(x)))
        return self.l3(x)