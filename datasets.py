from itertools import cycle
import pandas as pd
import torch
import torch.utils.data
from bpe import BPE
import numpy as np
import codecs

def rawcount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines

class TaggingDataSet(torch.utils.data.IterableDataset):
    
    def __init__(self, file_path, vocab, model_dir = ".", apply_bpe = False):
        self._file_path = file_path
        self._vocab = vocab
        self._size = rawcount(file_path) // 2
        self.bpe = None
        if apply_bpe:
            # setup BPE
            atoms = {x.strip():True for x in codecs.open(model_dir+"/atoms", encoding='utf-8')}
            self.bpe = BPE(codecs.open(model_dir+"/bpe.model", encoding='utf-8'), -1, "@@", None, None, atoms)

    def __iter__(self):
        yield from self.parse()

    def parse(self):
        chunksize=200000
        for chunk in pd.read_csv(self._file_path, chunksize=chunksize, header=None):
            for i in range(0,len(chunk),2):
                if self.bpe:
                    tokens = np.array([self._vocab.get(x, 1) for x in self.bpe.segment(chunk.iloc[i,0]).split()])
                else:
                    tokens = np.array([self._vocab.get(x, 1) for x in chunk.iloc[i,0].split()])
                tags = np.array([2 if x=="D" else 1 for x in chunk.iloc[i+1,0].split()])
                if len(tokens) > 128: # or len(tokens) < 3:
                    continue
                yield tokens, tags, len(tokens)

    def __len__(self):
        return self._size

class TaggingInferenceDataSet(TaggingDataSet):
    
    def __init__(self, file_path, vocab, model_dir = ".", apply_bpe = True):
        self._file_path = file_path
        self._vocab = vocab
        self._size = rawcount(file_path)
        self.bpe = None
        if apply_bpe:
            # setup BPE
            atoms = {x.strip():True for x in codecs.open(model_dir+"/atoms", encoding='utf-8')}
            self.bpe = BPE(codecs.open(model_dir+"/bpe.model", encoding='utf-8'), -1, "@@", None, None, atoms)

    def parse(self):
        chunksize=200000
        for chunk in pd.read_csv(self._file_path, chunksize=chunksize, header=None):
            for i in range(0,len(chunk)):
                if self.bpe:
                    import sys
                    sys.stderr.write(" ".join(self.bpe.segment(chunk.iloc[i,0]).split()))
                    tokens = np.array([self._vocab.get(x, 1) for x in self.bpe.segment(chunk.iloc[i,0]).split()])
                else:
                    tokens = np.array([self._vocab.get(x, 1) for x in chunk.iloc[i,0].split()])
                if len(tokens) > 128: # or len(tokens) < 3:
                    continue
                yield tokens, len(tokens)


class CycledTaggingDataSet(TaggingDataSet):
    def __iter__(self):
       return cycle(self.parse())

class ClsDataSet(torch.utils.data.IterableDataset):
    
    def __init__(self, file_path, vocab):
        self._file_path = file_path
        self._vocab = vocab
        self._size = rawcount(file_path)

    def __iter__(self):
        yield from self.parse()

    def parse(self):
        chunksize=200000
        for chunk in pd.read_csv(self._file_path, chunksize=chunksize, header=None):
            for i in range(0,len(chunk)):
                line = chunk.iloc[i,0].split()
                tokens = np.array([self._vocab.get(x,1) for x in line[1:]])
                cls = np.array([int(line[0])-1])
                if len(tokens) > 192 or len(tokens) < 3:
                    continue
                yield tokens, cls, len(tokens)

    def __len__(self):
        return self._size

class CycledClsDataSet(ClsDataSet):
    def __iter__(self):
       return cycle(self.parse())
