import argparse
import logging
import time
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import basic_classify
import datatools.word_vectors
import modules.maxpool_lstm

#general rule: all used modules should be able to created just by passing args
#todo: 
#-implement an evaluation module for setence classifcation that outputs the setences in a validation or test set, indicating whether the model predicted correctly or incorrectly
#-implement tensorboard
#-do some testing
def default_parser(parser=None):
    if parser is None:
        parser= argparse.ArgumentParser()
    parser.add_argument("--cuda",type=bool,default=False)
    parser.add_argument("--num_epochs",type=int,default=4)
    parser.add_argument("--validation_set_size",type=int,default=1000)
    parser.add_argument("--model_save_path",type=str, default= "../saved_models/") 
    parser.add_argument("--resume", type=bool, default= False )
    parser.add_argument("--res_file",type=str, default="recent_model") 
    parser.add_argument("--mode", type=str, choices=["evaluate", "train"], default="train")
    parser.add_argument("--use_saved_processed_data",type=bool,default=True)
    parser.add_argument("--processed_data_path",type=str,default="../saved_processed_data")
    parser.add_argument("--report_path",type=str,default="../reports")
    parser.add_argument("--batch_size", type= int, default=32)

    return parser







def main():
   logging.basicConfig(level=logging.INFO)
   parser=default_parser()



   parser=basic_classify.add_args(parser)
   args=parser.parse_args()
   basic_classify.run(args)



def show_params():
   logging.basicConfig(level=logging.INFO)
   parser=default_parser()
   parser=basic_classify.add_args(parser)
   args=parser.parse_known_args()[0]
   context=basic_classify.make_context(args)
   for name, param in context.model.named_parameters():
       print(name)
       print(param.requires_grad)



if __name__ == "__main__":
    main()
    #show_params()
