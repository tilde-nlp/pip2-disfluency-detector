#!/usr/bin/env python3
import argparse
import sys

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Fix labels after BPE split')
    args = parser.parse_args()

    words = [] 

    for line in sys.stdin:        
        if not words:
            words = line.split()
        else:
            labels = line.split()
            i = 0
            newlabels = []
            # for each word except [CLS] at the end
            for w in words[:-1]: 
                newlabels.append(labels[i])
                if not w.endswith("@@"):
                    i += 1
            print (" ".join(words))
            print (" ".join(newlabels))
            words = []
                    
        
    
