from torch.autograd import Function
from torch.autograd import Variable
import torch
import random
class ShakeDrop(Function):
    @staticmethod
    def forward(ctx, F, alpha_min, alpha_max, beta_min, beta_max, b_prob ):
        alpha=F.new(1,F.shape[1],1,1)
        alpha.uniform_(alpha_min,alpha_max)
        #alpha=random.uniform(alpha_min,alpha_max)
        #beta=random.uniform(beta_min, beta_max)
        beta=F.new(1,F.shape[1],1,1)
        beta.uniform_(beta_min,beta_max)
        beta=Variable(beta)
        b= 1 if random.uniform(0,1) <b_prob else 0 
        ctx.beta=beta
        ctx.b=b
        return (b+alpha-b*alpha)*F 
    @staticmethod
    def backward(ctx,grad_output):
        b=ctx.b
        beta=ctx.beta
        return (b+beta-b*beta)*grad_output, None, None, None, None, None
