import math
import logging
class MyAnneal:
    def __init__(self, optimizer, init_lr, Tmax, cur_step=-1):
        self.optimizer=optimizer
        self.init_lr=init_lr
        self.Tmax=Tmax
        self.cur_step=cur_step 
    def cur_lr(self):
        return 0.5*self.init_lr*(1+math.cos(self.cur_step/self.Tmax*math.pi) )
    def step(self):
        self.cur_step+=1
        lr=self.cur_lr()
        logging.info("Learning rate is now "+str(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr']=lr


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
            param_group['lr'] = self.init_lr*(1 - self.iter/self.max_iter)**self.power
