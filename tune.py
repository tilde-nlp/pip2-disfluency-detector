
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerModel
from torch.utils.data import DataLoader
from datasets import CycledTaggingDataSet, TaggingDataSet, ClsDataSet
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Fine-tune disfluency detector on labelled data')
parser.add_argument('model_dir', help='model to fine tune')
parser.add_argument('data', help='fine-tune data in tagging format')

args = parser.parse_args()

model_dir = args.model_dir
tune_data = args.data

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
    padded_targs = np.zeros((batch_size, max_length))
    for i, l in enumerate(lengths):
        padded_seqs[i, 0:l] = seqs[i][0:l]
        padded_targs[i, 0:l] = targs[i][0:l]

    return sort_batch(torch.tensor(padded_seqs), torch.tensor(padded_targs), torch.tensor(lengths))

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
nclstokens = 4 # D0, D1, S0, S1
ntagtokens = 2 + 1 # O, D + PAD
emsize = 512 # embedding dimension
nhid = 512 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.1 # the dropout value
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(TransformerModel(ntokens, nclstokens, ntagtokens, emsize, nhead, nhid, nlayers, dropout)).to(device)

######################################################################
# Run the model
# -------------
#

tag_criterion = nn.CrossEntropyLoss(ignore_index=0)
lr = 0.00001 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.995)

import time
def train(tag_data, sched_interval):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    tag_loader = DataLoader(tag_data, batch_size=tag_batch_size, collate_fn=pad_and_sort_tag_batch)
    for batch, sample in enumerate(tag_loader):

        data = sample[0].to(torch.int64).to(device)
        targets = sample[1].to(torch.int64).to(device)
        tag_output = model(data)[0]
        loss = tag_criterion(tag_output.view(-1, ntagtokens), targets.view(-1))
        
        loss = torch.mean(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        if batch % sched_interval == 0 and batch > 0:
            scheduler.step()

        log_interval = 100
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.8f} | ms/batch {:5.2f} | '
                  'loss {:5.5f} '.format(
                    epoch, batch, len(cls_data) // cls_batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()


######################################################################
# fine tune

epochs = 20 # The number of epochs

for epoch in range(1, epochs + 1):
    data = TaggingDataSet(tune_data, input_vocab)
    sched_step = len(data) // tag_batch_size // 4
    epoch_start_time = time.time()
    train(tag_data, sched_step)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s  '
          .format(epoch, (time.time() - epoch_start_time)))
    print('-' * 89)

torch.save(model.state_dict(), "%s/tune.mdl" % model_dir)



