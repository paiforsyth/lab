import torch.utils.data as data
import os
import logging
import torch

from . import wordindexer
from  . import word_vectors
from  . import text_tool
from  . import datasets



sfname="merged_simplification_classification"
def load_merged_simplification_classification(args):
    if args.use_saved_processed_data and os.path.isfile(os.path.join(args.processed_data_path,sfname+'_traindataset')) and os.path.isfile(os.path.join(args.processed_data_path,sfname+'_valdataset'))   and os.path.isfile(os.path.join(args.processed_data_path,sfname+'_indexer')) and os.path.isfile(os.path.join(args.processed_data_path,sfname+'_vects')):
        logging.info("Found cached "+sfname+" data.  Loading it.")
        train_dataset = torch.load( os.path.join(args.processed_data_path,sfname+'_traindataset')  )
        val_dataset = torch.load( os.path.join(args.processed_data_path,sfname+'_valdataset')  )
        vects = torch.load( os.path.join(args.processed_data_path,sfname+'_vects')  )
        indexer = torch.load( os.path.join(args.processed_data_path,sfname+'_indexer')  )
    else:
        logging.info("Could not find cached "+sfname  +" data. Indexing words.")
        indexer=wordindexer.WordIndexer()

        def index_words(filename):
            with open( os.path.join(args.simplify_ds_path,filename) ) as f:
                for line in f:
                    sentence=line.split('\t')[2]
                    sentence = text_tool.normalize_string(sentence) 
                    indexer.add_sentence(sentence)
    
        index_words("normal.aligned")
        index_words("simple.aligned")
        indexer= indexer.trimmed(args.simplification_data_trim)
        vects, missing_words = word_vectors.fasttext_from_file(args, indexer) 
        sequences=[]
        categories=[]
    
        def get_sentences_and_classes(filename,category):
            with open( os.path.join(args.simplify_ds_path,filename) ) as f:
                for line in f:
                    sentence=line.split('\t')[2]
                    sentence=text_tool.normalize_string(sentence)
                    seq=indexer.sentence2seq(sentence,include_sos_eos=True) 
                    sequences.append(seq)
                    categories.append(category)

        get_sentences_and_classes("normal.aligned",0)
        get_sentences_and_classes("simple.aligned",1)
       
        combined_dataset=datasets.SequenceClassificationDataset(sequences,categories)
        combined_dataset.shuffle()
        val_dataset, train_dataset= combined_dataset.split(args.validation_set_size)
        
        torch.save(train_dataset, os.path.join(args.processed_data_path,sfname+'_traindataset') )
        torch.save(val_dataset, os.path.join(args.processed_data_path,sfname+'_valdataset') )
        torch.save(vects, os.path.join(args.processed_data_path,sfname+'_vects') )
        torch.save(indexer, os.path.join(args.processed_data_path,sfname+'_indexer') )
    return train_dataset, val_dataset, vects, indexer



