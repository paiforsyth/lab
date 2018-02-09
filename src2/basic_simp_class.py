import argparse
import logging
import time
import collections
from tqdm import tqdm
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


import basic_simp_class
import datatools.simplification_data
import datatools.datasets
import datatools.word_vectors
import modules.maxpool_lstm
import monitoring.reporting


def add_args(parser):
    if parser is None:
        parser= argparse.ArgumentParser()
    parser.add_argument("--simplify_ds_path",type=str,default="../data/sentence-aligned.v2/")
    parser.add_argument("--fasttext_path",type=str,default="../data/fastText_word_vectors/" )
    parser.add_argument("--processed_data_path",type=str,default="../saved_processed_data")
    parser.add_argument("--simplification_data_trim",type=int, default=30000)
    parser.add_argument("--lstm_hidden_dim",type = int, default =300)
    parser.add_argument("--use_saved_processed_data",type=bool,default=True)
    parser.add_argument("--reports_per_epoch",type=int,default=10)
    return parser

Context=collections.namedtuple("Context","model, train_loader, val_loader, optimizer")

def make_context(args):
   train_dataset, val_dataset, index2vec, indexer = datatools.simplification_data.load_merged_simplification_classification(args)
   train_loader= data.DataLoader(train_dataset,batch_size = 32,shuffle = True,collate_fn = datatools.datasets.make_sequence_classification_collater(args))
   val_loader= data.DataLoader(val_dataset,batch_size = 32,shuffle = True,collate_fn = datatools.datasets.make_sequence_classification_collater(args))
   embedding=datatools.word_vectors.embedding(index2vec, indexer.n_words,300)
   model=modules.maxpool_lstm.MaxPoolLSTMFC.from_embed(embedding, args.lstm_hidden_dim) 
   if args.cuda:
       model=model.cuda()
   optimizer=optim.SGD(model.parameters(),lr=0.01)
   return Context(model, train_loader, val_loader, optimizer)



def evaluate(context): 
   correct=0
   total=0
   context.model.eval()
   for seqs, categories, pad_mat in context.val_loader:
        total+=seqs.shape[0]
        scores=context.model(seqs,pad_mat)
        scores=F.softmax(scores,dim=1).cpu()
        _,predictions=torch.max(scores,dim=1)
        correct+= torch.sum(predictions==categories).data[0]
   context.model.train()
   return correct / total 


def run(args):

   context=make_context(args) 

   starttime=time.time()
   report_interval=max(len(context.train_loader) //  args.reports_per_epoch ,1)
   for epoch_count in range(args.num_epochs):
    eval_score=evaluate(context)
    print("starting epoch number "+ str(epoch_count+1) +  " of " +str(args.num_epochs)+".  Accuracy is "+ str(eval_score) +".")
    step=1
    for seqs, categories, pad_mat in context.train_loader:
       step+=1
       context.optimizer.zero_grad()
       scores=context.model(seqs,pad_mat) #should have dimension batchsize
       loss=F.cross_entropy(scores,categories) 
       loss.backward()
       context.optimizer.step()
       if step % report_interval == 0:
            monitoring.reporting.report(starttime,step,len(context.train_loader),loss)
            


