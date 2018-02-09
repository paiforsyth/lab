import torch.nn as nn
#from https://github.com/daemon/vdpwi-nn-pytorch/blob/master/vdpwi/model.py
class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
