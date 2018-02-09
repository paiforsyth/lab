import argparse
import logging
import time
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


def add_args(parser):
    if parser is None:
        parser= argparse.ArgumentParser()
    parser.add_argument("--simplify_ds_path",type=str,default="../data/sentence-aligned.v2/")
    parser.add_argument("--fasttext_path",type=str,default="../data/fastText_word_vectors/" )
    parser.add_argument("--processed_data_path",type=str,default="../saved_processed_data")
    parser.add_argument("--simplification_data_trim",type=int, default=30000)
    parser.add_argument("--use_saved_processed_data",type=bool,default=True)
    #parser.add_argument("--report_interval",type=int,default=1000):
    return parser

def evaluate(loader, model): 
   correct=0
   total=0
   model.eval()
   for seqs, categories, pad_mat in loader:
        total+=seqs.shape[0]
        scores=model(seqs,pad_mat)
        scores=F.softmax(scores,dim=1).cpu()
        _,predictions=torch.max(scores,dim=1)
        correct+= torch.sum(predictions==categories).data[0]
   model.train()
   return correct / total 


def run(args):
   train_dataset,val_dataset, index2vec, indexer = datatools.simplification_data.load_merged_simplification_classification(args)
   train_loader= data.DataLoader(train_dataset,batch_size = 32,shuffle = True,collate_fn = datatools.datasets.make_sequence_classification_collater(args))
   val_loader= data.DataLoader(val_dataset,batch_size = 32,shuffle = True,collate_fn = datatools.datasets.make_sequence_classification_collater(args))
   embedding=datatools.word_vectors.embedding(index2vec, indexer.n_words,300)

   lstm=nn.LSTM(300,300,1, bidirectional=True, batch_first=True)
   linear=nn.Linear(600,2, bias=False)
   model=modules.maxpool_lstm.MaxPoolLSTMFC(embedding,lstm,linear) 

   if args.cuda:
       model=model.cuda()

   optimizer=optim.SGD(model.parameters(),lr=0.01)
   starttime=time.time()
   step=0
   loader_wrapper= tqdm(train_loader, total=len(train_loader), desc="Loss")
   for epoch_count in range(args.num_epochs):
    print("starting epoch number:"+ str(epoch_count+1))
    eval_score=evaluate(val_loader, model)
    loader_wrapper.set_postfix({"acc":eval_score})
    for seqs, categories, pad_mat in loader_wrapper:
       step+=1
       optimizer.zero_grad()
       scores=model(seqs,pad_mat) #should have dimension batchsize
       loss=F.cross_entropy(scores,categories) 
       loss.backward()
       optimizer.step()
        
       loader_wrapper.set_description("Loss: {:<8}".format(round(loss.cpu().data[0],5), eval_score ) )


