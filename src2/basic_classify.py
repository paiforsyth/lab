import argparse
import logging
import time
import collections
import os.path
import pickle
import torch.utils.data as data
import torch.nn as nn
import torch.nn.utils.clip_grad
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler
import numpy as np


import datatools.set_simp
import datatools.set_polarity
import datatools.set_cifar_challenge
import datatools.sequence_classification
import datatools.basic_classification
from datatools.basic_classification import DataType
import datatools.word_vectors
import modules.maxpool_lstm
import modules.squeezenet
import modules.kim_cnn
import monitoring.reporting
import monitoring.tb_log
import genutil.modules
from torchvision import transforms
import torchvision.datasets as tvds
def add_args(parser):
    if parser is None:
        parser= argparse.ArgumentParser() 
    parser.add_argument("--dataset_for_classification",type=str,choices=["simple","moviepol", "mnist", "cifar_challenge"],default="simple")

    parser.add_argument("--ds_path", type=str,default=None)
    parser.add_argument("--fasttext_path", type=str,default="../data/fastText_word_vectors/" )
    parser.add_argument("--data_trim", type=int, default=30000)
    parser.add_argument("--lstm_hidden_dim", type = int, default =300)
    parser.add_argument("--maxlstm_dropout_rate", type = int, default = 0.5)
    parser.add_argument("--reports_per_epoch", type=int,default=10)
    parser.add_argument("--save_prefix", type=str,default=None)
    parser.add_argument("--model_type", type=str, choices=["maxpool_lstm_fc", "kimcnn", "squeezenet"],default="maxpool_lstm_fc")
    modules.kim_cnn.add_args(parser)
    modules.squeezenet.add_args(parser)
    
    return parser

Context=collections.namedtuple("Context","model, train_loader, val_loader, optimizer, indexer, category_names, tb_writer, train_size, data_type, scheduler, test_loader")




def make_context(args):
   if args.dataset_for_classification == "simple":
        if args.save_prefix is None:
            args.save_prefix="simplification_classification"
        if args.ds_path is None:
            args.ds_path= "../data/sentence-aligned.v2" 
        train_dataset, val_dataset, index2vec, indexer = datatools.set_simp.load(args)
        category_names={0:"normal",1:"simple"}
        data_type=DataType.SEQUENCE
   elif args.dataset_for_classification == "moviepol":
        if args.save_prefix is  None:
            args.save_prefix= "moviepol"
        if args.ds_path is None:
            args.dspath = "../data/rt-polaritydata"
        train_dataset, val_dataset, index2vec, indexer = datatools.set_polarity.load(args)
        category_names={0:"negative",1:"positive"}
        data_type=DataType.SEQUENCE
   elif args.dataset_for_classification == "mnist":
        train_dataset = tvds.MNIST('../data/mnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (1,))]))
        val_dataset = tvds.MNIST('../data/mnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (1,))]))
        category_names={0:"1",1:"2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
        data_type = DataType.IMAGE 
   elif args.dataset_for_classification == "cifar_challenge":
        if args.mode == "train":
            f=open("../data/cifar/train_data","rb")
            squashed_images=pickle.load(f)
            labels=pickle.load(f)
            f.close()
            train_dataset,val_dataset = datatools.set_cifar_challenge.make_train_val_datasets(squashed_images, labels, args.validation_set_size, transform=None) 
            train_dataset.transform = transforms.Compose([transforms.RandomCrop(size=32 ,padding= 4), transforms.RandomHorizontalFlip(), transforms.ToTensor() ])
            val_dataset.transform = transforms.ToTensor()
        elif args.mode == "test":
            f=open("../data/cifar/test_data","rb")
            squashed_images=pickle.load(f)
            test_dataset= datatools.set_cifar_challenge.Dataset(data=squashed_images, labels=[-1]*squashed_images.shape[0], transform=transforms.ToTensor())
            data_type = DataType.IMAGE
            f.close()
        data_type = DataType.IMAGE
        category_names= { k:v for k,v in enumerate(datatools.set_cifar_challenge.CIFAR100_LABELS_LIST)}

   else:
        raise Exception("Unknown dataset.")
   




   if data_type == DataType.SEQUENCE:
        embedding=datatools.word_vectors.embedding(index2vec, indexer.n_words,300)
        train_loader= data.DataLoader(train_dataset,batch_size = args.batch_size,shuffle = True,collate_fn = datatools.sequence_classification.make_collater(args))
        val_loader= data.DataLoader(val_dataset,batch_size = args.batch_size, shuffle = False, collate_fn = datatools.sequence_classification.make_collater(args))
   elif data_type == DataType.IMAGE:
       indexer= None
       if args.mode == "train":
            train_loader=data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True, collate_fn=datatools.basic_classification.make_var_wrap_collater(args))
            val_loader=data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=datatools.basic_classification.make_var_wrap_collater(args))
       elif  args.mode == "test":
            test_loader=data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=datatools.basic_classification.make_var_wrap_collater(args))
            assert(args.resume)
   else:
       raise Exception("Unknown data type.")
        
   
   if args.model_type == "maxpool_lstm_fc":
    model=modules.maxpool_lstm.MaxPoolLSTMFC.from_args(embedding, args) 
   elif args.model_type == "kimcnn":
       model=modules.kim_cnn.KimCNN.from_args(embedding,args) 
   elif args.model_type == "squeezenet":
       model=modules.squeezenet.SqueezeNet.from_args(args)
   else:
       raise Exception("Unknown model")

   if args.cuda:
       model=model.cuda()

   if args.optimizer == "sgd":
        optimizer=optim.SGD(model.parameters(),lr=args.init_lr, momentum=args.sgd_momentum, weight_decay=args.sgd_weight_decay )
       
   elif args.optimizer == "rmsprop":
       optimizer = optim.RMSprop(model.parameters(), lr=args.init_lr)
   elif args.optimizer == "adam":
       optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
   else:
       raise Exception("Unknown optimizer.") 

   if args.lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)
   elif args.lr_scheduler == "plateau":
       scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", verbose=True, patience=args.plateau_lr_scheduler_patience)
   elif args.lr_scheduler == "linear":
        lam = lambda epoch: 1-args.linear_scheduler_subtract_factor* min(epoch,args.linear_scheduler_max_epoch)/args.linear_scheduler_max_epoch 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lam )
   elif args.lr_scheduler == "multistep":
        milestones=[args.multistep_scheduler_milestone1, args.multistep_scheduler_milestone2]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma )
   elif args.lr_scheduler == None:
       scheduler = None
   else: 
       raise Exception("Unknown Scheduler")


   if args.mode =="train":
       train_size=len(train_dataset)
       test_loader = None
   elif args.mode=="test":
       train_size= None
       train_loader = None
       val_loader = None
   return Context(model, train_loader, val_loader, optimizer, indexer, category_names=category_names, tb_writer=monitoring.tb_log.TBWriter("{}_"+args.save_prefix), train_size=train_size, data_type=data_type, scheduler=scheduler, test_loader=test_loader)






def run(args):

   context=make_context(args) 

   if args.resume:
       logging.info("loading saved model from file: "+args.res_file)
       context.model.load(os.path.join(args.model_save_path, args.res_file))
   
   if args.mode == "test":
       datatools.basic_classification.make_prediction_report(context, context.test_loader,args.test_report_filename ) 
       return




   context.tb_writer.write_hyperparams()
   timestamp=monitoring.reporting.timestamp()
   
   report_interval=max(len(context.train_loader) //  args.reports_per_epoch ,1)
   accumulated_loss=0 
   param_count=genutil.modules.count_trainable_params(context.model)
   logging.info("Number of parameters: "+ str(param_count))
   context.tb_writer.write_num_trainable_params(param_count)



   
   best_eval_score=-float("inf")
   for epoch_count in range(args.num_epochs):
        logging.info("Starting epoch "+str(epoch_count) +".")
        if args.param_difs:
           param_tensors=genutil.modules.get_named_trainable_param_tensors(context.model)
        step=0
        epoch_start_time=time.time()
        for batch_in, *other in context.train_loader: 
            categories = other[0]
            if context.data_type == DataType.SEQUENCE:
                pad_mat = other[1]  
            step+=1

            context.optimizer.zero_grad()
            scores= context.model(batch_in,pad_mat) if context.data_type == DataType.SEQUENCE else context.model(batch_in)  #should have dimension batchsize
            loss=  F.cross_entropy(scores,categories) 
            loss.backward()
            if args.grad_norm_clip is not None:
                torch.nn.utils.clip_grad.clip_grad_norm(context.model.parameters(), args.grad_norm_clip)
            context.optimizer.step()
            
            accumulated_loss+=loss.data[0]
            context.tb_writer.write_train_loss(loss.data[0])
            if step % report_interval == 0:
                monitoring.reporting.report(epoch_start_time,step,len(context.train_loader), accumulated_loss / report_interval)
                accumulated_loss = 0
        epoch_duration = time.time() - epoch_start_time
        context.tb_writer.write_data_per_second( context.train_size/epoch_duration)
        if args.param_difs:
            new_param_tensors=genutil.modules.get_named_trainable_param_tensors(context.model)
            context.tb_writer.write_param_change(new_param_tensors, param_tensors)
            param_tensors=new_param_tensors
        eval_score=datatools.basic_classification.evaluate(context, context.val_loader)
        context.tb_writer.write_accuracy(eval_score)
        logging.info("Finished epoch number "+ str(epoch_count+1) +  " of " +str(args.num_epochs)+".  Accuracy is "+ str(eval_score) +".")
        if context.scheduler is not None:
            if args.lr_scheduler == "exponential" or args.lr_scheduler == "linear" or args.lr_scheduler == "multistep":
                context.tb_writer.write_lr(context.scheduler.get_lr()[0] )
                context.scheduler.step()
            elif args.lr_scheduler == "plateau":
               # context.tb_writer.write_lr(next(context.optimizer.param_groups)['lr'] )
                context.scheduler.step(eval_score)
            else:
                raise Exception("Unknown Scheduler")

        if eval_score > best_eval_score:
            best_eval_score=eval_score
            logging.info("Saving model")
            context.model.save(os.path.join(args.model_save_path,timestamp+args.save_prefix +"_best_model" )  )
            context.model.save(os.path.join(args.model_save_path,timestamp+"recent_model" )  )

   logging.info("Loading best model")
   context.model.load(os.path.join(timestamp + args.model_save_path,args.save_prefix +"_best_model"))
   if context.data_type == DataType.SEQUENCE:
        datatools.sequence_classification.write_evaulation_report(context, context.val_loader,os.path.join(args.timestamp + report_path,args.save_prefix +".txt") , category_names=context.category_names) 
