"""Small conditional regressor: count prior plus shared head-set embedding."""
import torch
from torch import nn

class ValueNetwork(nn.Module):
    def __init__(self,candidates,poses):
        super().__init__()
        self.action_embedding=nn.Embedding(candidates,16)
        self.base=nn.Linear(2*poses,1,bias=False)
        self.base.weight.requires_grad_(False)
        self.residual=nn.Sequential(nn.Linear(32+2*poses,128),nn.SiLU(),nn.Dropout(.1),
                                    nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,states,membership,counts,actions):
        # The same learned head embeddings describe the selected set and action.
        selected=states@self.action_embedding.weight
        selected=selected/counts.sum(dim=-1,keepdim=True).clamp_min(1)
        features=torch.cat([selected,self.action_embedding(actions),membership,counts/6],dim=-1)
        prior=self.base(torch.cat([membership,counts],dim=-1)).squeeze(-1)
        return (prior+self.residual(features).squeeze(-1)).clamp_min(0)
