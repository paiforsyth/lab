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
import genutil.modules

#general rule: all used modules should be able to created just by passing args
#todo: 
#do some basic tests, like verifying that paramters change.  Consider using rmsprop
def default_parser(parser=None):
    if parser is None:
        parser= argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--num_epochs",type=int,default=4)
    parser.add_argument("--validation_set_size",type=int,default=1000)
    parser.add_argument("--model_save_path",type=str, default= "../saved_models/") 
    parser.add_argument("--resume_mode", type=str, choices=["none", "standard", "checkpoint_ensemble"], default= "none" )
    parser.add_argument("--res_file",type=str, default="recent_model") 
    parser.add_argument("--mode", type=str, choices=["test", "train"], default="train")
    parser.add_argument("--test_report_filename", type=str)
    parser.add_argument("--use_saved_processed_data", action="store_true")
    parser.add_argument("--processed_data_path",type=str,default="../saved_processed_data")
    parser.add_argument("--report_path",type=str,default="../reports")
    parser.add_argument("--batch_size", type= int, default=32)
    parser.add_argument("--param_report", action="store_true")
    parser.add_argument("--mod_report",action="store_true")
    parser.add_argument("--param_difs", action="store_true" )
    parser.add_argument("--optimizer", type=str, choices=["sgd", "rmsprop", "adam"], default="sgd")
    parser.add_argument("--init_lr",type=float, default=0.1)
    parser.add_argument("--sgd_momentum",type=float, default=0)
    parser.add_argument("--sgd_weight_decay", type=float, default=0)
    parser.add_argument("--plateau_lr_scheduler_patience",type=int, default=10)
    parser.add_argument("--lr_scheduler",type=str, choices=[None, "exponential", "plateau", "linear", "multistep", "epoch_anneal"], default="exponential")
    parser.add_argument("--lr_gamma",type=float, default=0.99)
    parser.add_argument("--linear_scheduler_max_epoch", type=int, default=300)
    parser.add_argument("--linear_scheduler_subtract_factor", type=float, default=0.99)
    parser.add_argument("--multistep_scheduler_milestone1", type=int, default=150)
    parser.add_argument("--multistep_scheduler_milestone2", type=int, default=225)
    parser.add_argument("--epoch_anneal_numcycles", type=int, default=6)


    parser.add_argument("--grad_norm_clip",type=float, default=None)
    parser.add_argument("--output_level", type=str, choices=["info", "debug"], default="info") 
    parser.add_argument("--first_checkpoints_in_ensemble", type=int, default=2)
    parser.add_argument("--boundary_checkpoints_in_ensemble", type=int, default=6) #one beyond the last checkpoint

    return parser







def main():
   logging.basicConfig(level=logging.INFO)
   parser=default_parser()



   parser=basic_classify.add_args(parser)
   args=parser.parse_args()
   if args.param_report:
       show_params()
       return
   if args.mod_report:
       show_submods()
       return
   if args.output_level == "debug":
        logging.getLogger().setLevel(logging.DEBUG)
   basic_classify.run(args)



def show_params():
   logging.basicConfig(level=logging.DEBUG)
   parser=default_parser()
   parser=basic_classify.add_args(parser)
   args=parser.parse_known_args()[0]
   context=basic_classify.make_context(args)
   for name, param in context.model.named_parameters():
       print(name)
       print(param.shape)
       print(param.requires_grad)
   param_count = genutil.modules.count_trainable_params(context.model)
   print("total trainable params:{}".format(param_count))
    

def show_submods():
   logging.basicConfig(level=logging.DEBUG)
   parser=default_parser()
   parser=basic_classify.add_args(parser)
   args=parser.parse_known_args()[0]
   context=basic_classify.make_context(args)
   print("layers:")
   print(list(context.model.children()))


if __name__ == "__main__":
    main()
    #show_params()
