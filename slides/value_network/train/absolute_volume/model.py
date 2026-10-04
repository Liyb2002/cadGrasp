"""Fixed-pose state/action network for witnessed final occupied-volume costs."""
import torch
from torch import nn


class AbsoluteValueNetwork(nn.Module):
    def __init__(self, heads, poses, width=256):
        super().__init__()
        self.heads, self.poses = heads, poses
        # Explicit selected-head bits preserve the state, including head identity.
        self.encoder = nn.Sequential(nn.Linear(heads + poses, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(poses, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.evidence = nn.Linear(width, heads)
        self.values = nn.Linear(width, heads)
        self.terminal_value = nn.Linear(width, 1)
        self.tie_order = nn.Linear(width, heads)
        self.seating = nn.Linear(width, poses * 2)

    def forward(self, state, membership):
        z = self.encoder(torch.cat([state, membership], -1))
        context = self.context(membership)
        return (self.evidence(context) - 1000 * state, self.terminal_value(context) + .05 * self.values(z),
                self.tie_order(context), self.seating(context))
