import argparse
import logging
import time
from tqdm import tqdm
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import basic_simp_class
import datatools.simplification_data
import datatools.datasets
import datatools.word_vectors
import modules.maxpool_lstm

#general rule: all used modules should be able to created just by passing args
#todo: could create a trainingtools module.  add evaluate binary classifier that takes a context and a loader.
def default_parser(parser=None):
    if parser is None:
        parser= argparse.ArgumentParser()
    parser.add_argument("--cuda",type=bool,default=False)
    parser.add_argument("--num_epochs",type=int,default=4)
    parser.add_argument("--validation_set_size",type=int,default=1000)
    return parser







def main():
   logging.basicConfig(level=logging.INFO)
   parser=default_parser()



   parser=basic_simp_class.add_args(parser)
   args=parser.parse_known_args()[0]
   basic_simp_class.run(args)


if __name__ == "__main__":
    main()



