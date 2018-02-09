import random
import torch.utils.data as data
import torch
from torch.autograd import Variable
import numpy as np


class SequenceClassificationDataset(data.Dataset):
    def __init__(self, sequences, categories):
        assert len(sequences) == len(categories)
        self.sequences = sequences
        self.categories = categories
        
    def __getitem__(self,idx):
        return (self.sequences[idx], self.categories[idx])

    def __len__(self):
        return len(self.sequences)
    def shuffle(self):
        c = list(zip(self.sequences, self.categories))
        random.shuffle(c)
        self.sequences, self.categories = zip(*c)

    def split(self, index):
        return SequenceClassificationDataset(self.sequences[:index],self.categories[:index]), SequenceClassificationDataset(self.sequences[index:],self.categories[index:] )


def make_sequence_classification_collater(args):
    def collater(batch):
        seqs=[entry[0] for entry in batch ]
        categories=[entry[1] for entry in batch]
        lengths=[len(entry[0]) for entry in  batch  ]
        max_len=max(lengths)
        pt_seqs=torch.LongTensor(len(seqs),max_len).zero_()
        pad_mat=torch.ones(len(seqs),max_len)
        for i in range(len(seqs)):
            pt_seqs[i,:lengths[i]]= torch.LongTensor(seqs[i])
            pad_mat[i,:lengths[i]]=0
        pt_seqs=Variable(pt_seqs)
        pt_categories= Variable(torch.LongTensor(categories))
        pad_mat=Variable(pad_mat)
        pad_mat=pad_mat.view(-1,max_len,1)
        if args.cuda:
            pt_seqs=pt_seqs.cuda()
            pt_categories=pt_categories.cuda()
            pad_mat=pad_mat.cuda()
        return pt_seqs, pt_categories, pad_mat
    return collater
