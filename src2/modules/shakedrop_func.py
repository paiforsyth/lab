from torch.autograd import Function
class ShakeDrop(Function):
    @staticmethod
    def forward(ctx, F, alpha_min, alpha_max, beta_min, beta_max,b_prob ):
        
