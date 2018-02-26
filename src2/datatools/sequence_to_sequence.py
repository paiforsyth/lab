class Dataset(data.Dataset):
    '''
    raw_sequences is an optional item that stores an unprocessed sequences. used mainly in evaluation
    '''
    def __init__(self, src_sequences, tgt_sequences, raw_src_sequences=None, raw_tgt_sequences=None):
        assert len(src_sequences) == len(tgt_sequences)
        assert len(src_sequences) > 0
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences
        if raw_src_sequences is None:
            self.raw_src_sequences= [None]*len(sequences)
        else:
            self.raw_src_sequences=raw_src_sequences
        if raw_tgt_sequences is None:
            self.raw_tgt_sequences= [None]*len(sequences)
        else:
            self.raw_tgt_sequences=raw_tgt_sequences
        
        
    def __getitem__(self,idx):
            return (self.src_sequences[idx], self.tgt_sequences[idx], self.raw_src_sequences[idx],self.raw_tgt_sequences[idx] )

    def __len__(self):
        return len(self.src_sequences)

    def shuffle(self):
        c = list(zip(self.src_sequences, self.tgt_sequences, self.raw_src_sequences,self.raw_tgt_sequences))
        random.shuffle(c)
        self.src_sequences, self.tgt_sequences, self.raw_src_sequences, self.raw_tgt_sequences = zip(*c)

    def split(self, index):
        return Dataset(self.src_sequences[:index],self.tgt_sequences[:index], self.raw_src_sequences[:index], self.raw_tgt_sequences[:index]), Dataset(self.self_sequences[index:],self.tgt_sequences[index:], self.raw_src_sequences[index:],self.raw_tgt_sequences[index:]  )

    def remove_raw(self):
        self.raw_src_sequences=[None]*len(self.src_sequences)
        self.raw_tgt_sequences=[None]*len(self.tgt_sequences)
