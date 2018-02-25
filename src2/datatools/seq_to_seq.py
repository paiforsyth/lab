class Dataset(data.Dataset):
    '''
    raw_sequences is an optional item that stores an unprocessed sequences. used mainly in evaluation
    '''
    def __init__(self, src_sequences, tgt_sequences, raw_sequences=None):
        assert len(src_sequences) == len(tgt_sequences)
        assert len(src_sequences) > 0
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences
        if raw_sequences is None:
            self.raw_sequences= [None]*len(sequences)
        else:
            self.raw_sequences=raw_sequences
        
    def __getitem__(self,idx):
            return (self.sequences[idx], self.categories[idx], self.raw_sequences[idx] )

    def __len__(self):
        return len(self.sequences)

    def shuffle(self):
        c = list(zip(self.sequences, self.categories, self.raw_sequences))
        random.shuffle(c)
        self.sequences, self.categories, self.raw_sequences = zip(*c)

    def split(self, index):
        return Dataset(self.sequences[:index],self.categories[:index], self.raw_sequences[:index]), Dataset(self.sequences[index:],self.categories[index:], self.raw_sequences[index:] )

    def remove_raw(self):
        self.raw_sequences=[None]*len(self.sequences)
