#!/usr/bin/env python3
import argparse
import sys
import random

from create_tag import s_add

def s_del(line):
    line = line.copy()
    sample_size = min(random.randint(1,3), len(line)-1)
    positions = random.sample(range(0, len(line)), sample_size)
    positions.sort() 
    deleted = 0   
    for i, pos in enumerate(positions):
        pert = random.choice([1,2])
        if i+1 < len(positions):
           max_count = positions[i+1] - pos
        else:
           max_count = len(line) - pos

        count = min(random.randint(1,6), max_count)

        if deleted + count >= len(line):
            continue

        for i in range(pos, pos+count):
           line[i] = "" 
           deleted+=1

    return [x for x in line if x != ""]


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Created synthetic data for classification task')
    parser.add_argument('order', help='model to test')
    args = parser.parse_args()

    max_order = int(args.order)
    
    mgrams = ["%dgrams" % (i+1) for i in range(0,max_order)]

    for line in sys.stdin:
        line = line.split()
        pert = random.choice([1,2])
        if pert == 1:
            line2, labels = s_add(line, mgrams)
        else:
            line2 = s_del(line)
        line = " ".join(line)
        line2 = " ".join(line2)
        if pert == 1:
            if random.choice([1,2]) == 1:
                print("1 [CLS] %s [SEP] %s" % (line2, line))
            else:
                print("2 [CLS] %s [SEP] %s" % (line, line2))
        else:
            if random.choice([1,2]) == 1:
                print("3 [CLS] %s [SEP] %s" % (line2, line))
            else:
                print("4 [CLS] %s [SEP] %s" % (line, line2))

