import torch
import torch.nn as nn

import torch.nn.functional as F
from . import serialmodule

class KimCNN(serialmodule.SerializableModule):
    def __init__(self,embedding):
        embedding_dim=embedding.weights.shape[1]
        self.embedding=embedding
        self.
