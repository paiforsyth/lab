#from https://github.com/daemon/vdpwi-nn-pytorch/blob/master/vdpwi/utils/tb.py
import datetime
import sys
import os.path

from tensorboardX import SummaryWriter

class TBWriter(object):
    def __init__(self, run_name_fmt="run_{}"):
        self.writer = SummaryWriter("../lb_logs/")
        self.run_name = run_name_fmt.format(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"))
        self.writer = SummaryWriter(os.path.join("../tb_logs/",self.run_name))
        self.train_idx = 0
        self.acc_idx = 0

    def write_hyperparams(self):
        self.writer.add_text("{}/hyperparams".format(self.run_name), " ".join(sys.argv))

    def write_train_loss(self, loss):
        self.writer.add_scalar("{}/train_loss".format(self.run_name), loss, self.train_idx)
        self.train_idx += 1

    def write_accuracy(self, acc):
        self.writer.add_scalar("{}/accuracy".format(self.run_name), acc , self.acc_idx)
        self.acc_idx += 1

    # def next(self):
        # self.i += 1
