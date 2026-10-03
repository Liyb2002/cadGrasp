"""Lossless selected-head mask input, multi-action value and bank-coverage outputs."""
import torch
from torch import nn

class FixedValueNetwork(nn.Module):
    def __init__(self,candidates=3000,poses=15,width=512,depth=2):
        super().__init__();self.candidates=candidates;self.poses=poses
        layers=[nn.Linear(candidates+2*poses,width),nn.GELU()]
        for _ in range(depth-1):layers.extend([nn.Linear(width,width),nn.GELU()])
        self.encoder=nn.Sequential(*layers)
        self.value=nn.Linear(width,candidates)
        self.coverage=nn.Linear(width,candidates)
    def forward(self,states,membership,counts):
        hidden=self.encoder(torch.cat([states,membership,counts/6],dim=-1))
        return self.value(hidden),self.coverage(hidden)
