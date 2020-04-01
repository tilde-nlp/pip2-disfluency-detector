#!/usr/bin/env python3
import argparse
import sys
import os
import random


def random_mgram(mgrams):
    order = random.randint(0,len(mgrams)-1)
    mgram = get_random_line(mgrams[order])   
    return mgram.split()[:-1]

def get_random_line(filepath: str) -> str:
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        while True:
            pos = random.randint(0, file_size)
            if not pos:  # the first line is chosen
                f.seek(0,0)
                return f.readline().decode()  # return str
            f.seek(pos)  # seek to random position
            f.readline()  # skip possibly incomplete line
            line = f.readline()  # read next (full) line
            if line:
                return line.decode()
            # else: line is empty -> EOF -> try another position in next iteration


def s_add(line, mgrams):
    line = line.copy()
    sample_size = min(random.randint(1,3), len(line))
    positions = random.sample(range(0, len(line)+1), sample_size)
    positions.sort()
    labels = ["O"] * len(line)
    for i, pos in enumerate(positions):
        pert = random.choice([1,2])
        if i+1 < len(positions):
           max_count = positions[i+1] - pos
        else:
           max_count = len(line) - pos

        if pert == 1:
           # repeat
           count = min(random.randint(1,6), max_count)
           # count is 0 when pos is beyond last word in the sentence
           if count == 0:
               if len(positions) == 1:
                   # no perturbations were made, can not skip
                   count = 1
                   pos = pos - 1
               else:
                   # skip
                   continue
           line[pos+count-1] += " " + " ".join(line[pos:pos+count])
           labels[pos+count-1] += " " + " ".join(["D"]*count)
        else:
           # add mgram
           noise = random_mgram(mgrams)           
           count = len(noise)
           if pos == 0:
               line[0] = " ".join(noise) + " " + line[0]
               labels[0] = " ".join(["D"]*count) + " " + labels[0]
           else:
               line[pos-1] += " " + " ".join(noise)
               labels[pos-1] += " " + " ".join(["D"]*count)

    return line, labels
           

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Created synthetic data for classification task')
    parser.add_argument('order', help='model to test')
    args = parser.parse_args()

    max_order = int(args.order)
    
    mgrams = ["%dgrams" % (i+1) for i in range(0,max_order)]

    for line in sys.stdin:
        line = line.strip().split()
        if not line:
            continue
        if random.choice([1,2]) == 2:
            line, labels = s_add(line, mgrams)
        else:
            labels = ["O"] * len(line)         
        line = " ".join(line)
        labels = " ".join(labels)
        print("[CLS] %s" % line)
        print("%s" % labels)

    
