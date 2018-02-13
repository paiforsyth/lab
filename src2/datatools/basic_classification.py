from enum import Enum
import torch.nn.functional as F
import torch
import torch.utils.data.dataloader
from torch.autograd import Variable
class DataType(Enum):
    SEQUENCE=1
    IMAGE=2

def evaluate(context, loader): 
   correct=0
   total=0
   context.model.eval()
   for batch, *other in loader:
        categories=other[0]
        if context.data_type==DataType.SEQUENCE:
            pad_mat = other[1]
        total+=batch.shape[0]
        scores= context.model(batch,pad_mat) if context.data_type == DataType.SEQUENCE else context.model(batch)  #should have dimension batchsize
        scores=F.softmax(scores,dim=1)
        _,predictions=torch.max(scores,dim=1)
        correct+= torch.sum(predictions==categories).cpu().data[0]
   context.model.train()
   return correct / total 


def make_var_wrap_collater(args):
    def collater(batch_in):
       batch_in, categories, *rest=torch.utils.data.dataloader.default_collate(batch_in)
       batch_in = Variable(batch_in)
       categories =Variable(categories)
       if args.cuda:
           batch_in = batch_in.cuda()
           categories = categories.cuda()
       return tuple([batch_in, categories, *rest])
    return collater
