# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerModel
from torch.utils.data import DataLoader
from datasets import TaggingInferenceDataSet
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description='Use disfluency detector to tag data')
parser.add_argument('model_dir', help='model to test')
parser.add_argument('data', help='text data')
parser.add_argument('--iter', default="model.mdl", help='model iteration')

args = parser.parse_args()

model_dir = args.model_dir
test_data = args.data




######################################################################
# Load and batch data
# -------------------
#

input_vocab={x.strip():i for i,x in enumerate(open(model_dir+"/vocab","r",encoding="utf8"), 2)}
input_vocab[""] = 0 # empty word
input_vocab["<unk>"] = 1 # unknown word

reverse_vocab={i:x.strip() for i,x in enumerate(open(model_dir+"/vocab","r",encoding="utf8"), 2)}
reverse_vocab[0] = "" # empty word
reverse_vocab[1] = "<unk>" # empty word

def pad_and_sort_tag_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest, 
    """
    batch_size = len(DataLoaderBatch)
    batch_split = list(zip(*DataLoaderBatch))

    seqs, lengths = batch_split[0], batch_split[1]
    max_length = max(lengths)

    padded_seqs = np.zeros((batch_size, max_length))
    for i, l in enumerate(lengths):
        padded_seqs[i, 0:l] = seqs[i][0:l]

    return torch.tensor(padded_seqs), torch.tensor(lengths)


tag_batch_size = 32

######################################################################
# Initiate an instance
# --------------------
#


######################################################################
# The model is set up with the hyperparameter below. The vocab size is
# equal to the length of the vocab object.
#

ntokens = len(input_vocab) # the size of vocabulary
nclstokens = 4 # D0, D1, S0, S1 + PAD
ntagtokens = 1 # binary O or D
emsize = 512 # embedding dimension
nhid = 512 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.1 # the dropout value
device = torch.device("cpu")
model = TransformerModel(ntokens, nclstokens, ntagtokens, emsize, nhead, nhid, nlayers, dropout).to(device)
state_dict = torch.load("%s/%s" % (model_dir, args.iter),map_location='cpu')

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("module."):
        name = k[7:] # remove 'module.' of dataparallel
    else:
        name = k
    new_state_dict[name]=v

model.load_state_dict(new_state_dict)

import time

def tag(eval_model, tag_data):
    tag_loader = DataLoader(tag_data, batch_size=tag_batch_size, collate_fn=pad_and_sort_tag_batch)
    eval_model.eval() # Turn on the evaluation mode
    with torch.no_grad():
        target_true = 0.
        predicted_true = 0.
        correct_true = 0.
        unks_count = 0.
        for batch, sample in enumerate(tag_loader):

            data = sample[0].to(torch.int64).to(device)
            tag_output = eval_model(data)[0]

            for j in range(0, tag_output.size(0)):
                d_vote = 0
                o_vote = 0
                delim = ""
                for i,x in enumerate(data[j]):
                    if x == 1:
                        sys.stderr.write("<unk> detected at pos %d %d\n" % (batch*tag_batch_size+j, i))
                    if x == 0 or int(x) == input_vocab["[CLS]"]:
                        break
                    if tag_output[j][i]>0.5:
                        d_vote += 1
                    else:
                        o_vote += 1
                    if not reverse_vocab[int(x)].endswith("@@"):
                       sys.stdout.write(delim)
#                       sys.stdout.write("%s:"%reverse_vocab[int(x)].encode("utf-8"))
                       if d_vote >= o_vote:
                           sys.stdout.write("D")
                       else:
                           sys.stdout.write("O")
                       d_vote = 0
                       o_vote = 0
                       delim = " "
                sys.stdout.write("\n")

 
tag(model, TaggingInferenceDataSet(test_data, input_vocab, model_dir))

