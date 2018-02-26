import collections

Context=collections.NamedTuple("Context","encoder","decoder")
def add_args(parser):
    pass
def make_context(args)
   if args.dataset_seq == "multi30k_auto":
        if args.save_prefix is None:
            args.save_prefix="multi30kauto"
        if args.ds_path is None:
            args.ds_path= "../data/sentence-aligned.v2" 
        train_dataset, val_dataset, index2vec, indexer = datatools.set_simp.load(args)
        category_names={0:"normal",1:"simple"}
        data_type=DataType.SEQUENCE
 
