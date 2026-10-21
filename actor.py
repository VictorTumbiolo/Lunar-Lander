import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action

        #  256 wiht Layer Norm
        self.l1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        # outputs [action main engine, action side thruster]
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.leaky_relu(self.ln1(self.l1(state)))
        x = F.leaky_relu(self.ln2(self.l2(x)))
        x = torch.tanh(self.l3(x))
        return x * self.max_action