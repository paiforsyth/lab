#based on https://github.com/ibadami/pytorch-semseg/blob/4f9fa49efa4114f91afe02a9958efc1c05f44d97/train.py
class PolyLrDecayer:
    def __init__(self,max_iter, init_lr, power):
        self.max_iter=max_iter
        self.iter=0
        self.init_lr=init_lr
        self.power=power
    def set_lr_poly(self, optimizer ):
        self.iter+=1
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.init_lr*(1 - self.iter/max_iter)**self.power
