from  . import text_tool
def load(args):
    indexer=wordindexer.WordIndexer()
    def index_words(filename):
        f= open( os.path.join(args.ds_path,filename) )
        for line in f:
                sentence=line
                sentence = text_tool.normalize_string(sentence) 
                indexer.add_sentence(sentence)
        f.close()
        index_words(train.en)    
