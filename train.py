
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerModel
from torch.utils.data import DataLoader
from datasets import CycledTaggingDataSet, TaggingDataSet, ClsDataSet

######################################################################
# Load and batch data
# -------------------
#

input_vocab={x.strip():i for i,x in enumerate(open("vocab","r",encoding="utf8"))}
input_vocab[""] = 0 # empty word

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

tag_batch_size = 84
cls_batch_size = 172


######################################################################
# Initiate an instance
# --------------------
#


######################################################################
# The model is set up with the hyperparameter below. The vocab size is
# equal to the length of the vocab object.
#

ntokens = len(input_vocab) # the size of vocabulary
nclstokens = 4 + 1 # D0, D1, S0, S1 + PAD
ntagtokens = 2 + 1 # O, D + PAD
emsize = 200 # embedding dimension
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
cls_criterion = nn.CrossEntropyLoss(ignore_index=0)
lr = 0.0001 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.995)

import time
def train(tag_data, cls_data, sched_interval):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    tag_loader = DataLoader(tag_data, batch_size=tag_batch_size, collate_fn=pad_and_sort_tag_batch)
    cls_loader = DataLoader(cls_data, batch_size=cls_batch_size, collate_fn=pad_and_sort_cls_batch)
    tag_batch = iter(tag_loader)
    for batch, sample in enumerate(cls_loader):
        # cls loss
        data = sample[0].to(device)
        targets = sample[1].to(device)
        optimizer.zero_grad()
        cls_output = model(data)[1]
        loss = cls_criterion(cls_output.view(-1, nclstokens), targets)

        # tag loss
        sample = next(tag_loader)
        data = sample[0].to(device)
        targets = sample[1].to(device)
        tag_output = model(data)[0]
        loss = tag_criterion(tag_output.view(-1, ntagtokens), targets)
        
        loss = torch.mean(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        if batch % sched_interval and batch > 0:
            validate(model)
            model.train()
            scheduler.step()

        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} '.format(
                    epoch, batch, len(cls_data) // cls_batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()


def validate(eval_model):
    cls_data = ClsDataSet("dev.cls", input_vocab)
    tag_data = CycledTaggingDataSet("dev.tag", input_vocab)
    loss = evaluate(eval_model, tag_data, cls_data)
    with open("valid.log","a") as f:
        f.write("%s\n" % loss)
    return loss
    
def evaluate(eval_model, tag_data, cls_data):
    tag_loader = DataLoader(tag_data, batch_size=tag_batch_size, collate_fn=pad_and_sort_tag_batch)
    cls_loader = DataLoader(cls_data, batch_size=cls_batch_size, collate_fn=pad_and_sort_cls_batch)
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        tag_batch = iter(tag_loader)
        for batch, sample in enumerate(cls_loader):
            # cls loss
            data = sample[0].to(device)
            targets = sample[1].to(device)
            cls_output = model(data)[1]
            loss = cls_criterion(cls_output.view(-1, nclstokens), targets)

            # tag loss
            sample = next(tag_loader)
            data = sample[0].to(device)
            targets = sample[1].to(device)
            tag_output = model(data)[0]
            loss += tag_criterion(tag_output.view(-1, ntagtokens), targets)

            loss = torch.mean(loss)

            total_loss += loss.item()
    return total_loss / (len(tag_data) + len(cls_data))

######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

best_val_loss = float("inf")
epochs = 30 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    cls_data = ClsDataSet("train.cls", input_vocab)
    tag_data = CycledDataSet("train.tag", input_vocab)
    sched_step = len(cls_data) // cls_batch_size // 4
    epoch_start_time = time.time()
    train(tag_data, cls_data, sched_step)
    val_loss = validate(model)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
          .format(epoch, (time.time() - epoch_start_time), val_loss))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        torch.save(model.state_dict(), "model.mdl")



