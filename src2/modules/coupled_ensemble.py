import torch.nn as nn
import torch.nn.functional as F
import torch

from . import serialmodule

#based on"coupled ensembled of neural networks" by Dutt et al.
class CoupledEnsemble(serialmodule.SerializableModule):
    def __init__(self, model_iterable):
        super().__init__()
        self.model_list=nn.ModuleList(model_iterable)

    def forward(self,x):
       out=0
       for model in self.model_list:
           out+=F.log_softmax(model(x), dim=1)
       out/=len(out)
       return out
