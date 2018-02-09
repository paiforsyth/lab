import torch
import torch.nn as nn
import torch.nn.functional as F

from . import serialmodule


VERY_NEGATIVE=-1000000
class MaxPoolLSTM(serialmodule.SerializableModule):
    def __init__(self,embedding, lstm):
        super().__init__()
        self.embedding=embedding
        self.lstm=lstm

    def forward(self,x,pad_mat):
        out=self.embedding(x) #batchsize by max sequence length by embedding dim
        out,_=self.lstm(out) #batchsize by max_sequence length by hidden dim
        #import pdb; pdb.set_trace()
        out=out+VERY_NEGATIVE*pad_mat #broadcast
        out, _=torch.max(out,dim=1) #batchsize by hidden_dim
        return out

        
class MaxPoolLSTMFC(serialmodule.SerializableModule): 
    '''
        FC = fully connected
    '''
    def __init__(self,embedding, lstm, lin):
        super().__init__()
        self.max_pool_lstm=MaxPoolLSTM(embedding,lstm)
        self.lin=lin
    
    def forward(self,x,pad_mat):
        out=self.max_pool_lstm(x,pad_mat)
        out=self.lin(out)
        out=F.leaky_relu(out)
        return out
