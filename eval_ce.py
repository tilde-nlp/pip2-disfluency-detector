
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerModel
from torch.utils.data import DataLoader
from datasets import CycledTaggingDataSet, TaggingDataSet, ClsDataSet
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test disfluency detector on labelled data')
parser.add_argument('model_dir', help='model to test')
parser.add_argument('data', help='test data in tagging format')

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

def sort_batch(batch, targets, lengths):
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]
    return seq_tensor, target_tensor, seq_lengths

def pad_and_sort_tag_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest, 
    """
    batch_size = len(DataLoaderBatch)
    batch_split = list(zip(*DataLoaderBatch))

    seqs, targs, lengths = batch_split[0], batch_split[1], batch_split[2]
    max_length = max(lengths)

    padded_seqs = np.zeros((batch_size, max_length))
    padded_targs = np.zeros((batch_size, max_length - 1))
    for i, l in enumerate(lengths):
        padded_seqs[i, 0:l] = seqs[i][0:l]
        padded_targs[i, 0:l-1] = targs[i][0:l-1]

    return sort_batch(torch.tensor(padded_seqs), torch.tensor(padded_targs), torch.tensor(lengths))

def pad_and_sort_cls_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest, 
    """
    batch_size = len(DataLoaderBatch)
    batch_split = list(zip(*DataLoaderBatch))

    seqs, targs, lengths = batch_split[0], batch_split[1], batch_split[2]
    max_length = max(lengths)

    padded_seqs = np.zeros((batch_size, max_length))
    for i, l in enumerate(lengths):
        padded_seqs[i, 0:l] = seqs[i][0:l]

    return sort_batch(torch.tensor(padded_seqs), torch.tensor(targs).view(-1,1), torch.tensor(lengths))

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
ntagtokens = 2 + 1 # O, D + PAD
emsize = 512 # embedding dimension
nhid = 512 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.1 # the dropout value
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = TransformerModel(ntokens, nclstokens, ntagtokens, emsize, nhead, nhid, nlayers, dropout).to(device)
state_dict = torch.load(model_dir+"/model.mdl",map_location='cpu')

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


def validate(eval_model):
    tag_criterion = nn.CrossEntropyLoss(ignore_index=0)
    cls_criterion = nn.CrossEntropyLoss(ignore_index=0)
    cls_data = ClsDataSet("dev.cls", input_vocab)
    tag_data = CycledTaggingDataSet("dev.tag", input_vocab) 
    tag_loader = DataLoader(tag_data, batch_size=84, collate_fn=pad_and_sort_tag_batch)
    cls_loader = DataLoader(cls_data, batch_size=172, collate_fn=pad_and_sort_cls_batch)
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        tag_batch = iter(tag_loader)
        for batch, sample in enumerate(cls_loader):
            # cls loss
            data = sample[0].to(torch.int64).to(device)
            targets = sample[1].to(torch.int64).to(device)
            cls_output = model(data)[1]
            loss = cls_criterion(cls_output.view(-1, nclstokens), targets.view(-1))

            # tag loss
            sample = next(tag_batch)
            data = sample[0].to(torch.int64).to(device)
            targets = sample[1].to(torch.int64).to(device)
            tag_output = model(data)[0]
            loss += tag_criterion(tag_output.view(-1, ntagtokens), targets.view(-1))

            loss = torch.mean(loss)

            total_loss += loss.item()
    loss = total_loss
       
    return loss

def evaluate(eval_model, tag_data):
    tag_loader = DataLoader(tag_data, batch_size=tag_batch_size, collate_fn=pad_and_sort_tag_batch)
    eval_model.eval() # Turn on the evaluation mode
    with torch.no_grad():
        target_true = 0.
        predicted_true = 0.
        correct_true = 0.
        for batch, sample in enumerate(tag_loader):
            data = sample[0].to(torch.int64).to(device)
            targets = sample[1].to(torch.int64).to(device)
            tag_output = eval_model(data)[0]
            predicted_disfluencies = torch.argmax(tag_output, dim=2) == 2
            target_disfluencies = targets == 2
            target_true += torch.sum(target_disfluencies).float()
            predicted_true += torch.sum(predicted_disfluencies).float()
            correct_true += torch.sum(target_disfluencies * predicted_disfluencies).float()

    recall = correct_true / target_true
    precision = correct_true / predicted_true
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score


print(validate(model))
print(evaluate(model, TaggingDataSet(test_data, input_vocab)))

