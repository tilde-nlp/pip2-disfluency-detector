#!/usr/bin/env python3
import sys

ln = 0
for line in sys.stdin:
    ln += 1
    tokens = line.strip().split()

    if len(tokens) < 2:
        continue

    words = []
    labels = []
    for token in tokens:
        token = token.split(":")
        if len(token) != 2:
            sys.stderr.write("Error at line %d, token %s\n" % (ln,token))
            sys.exit(1)
        word = token[0]
        label = token[1]
        words.append(word)
        labels.append("D" if label == "D" else "O")
    words.append("[CLS]")

    print (" ".join(words))
    print (" ".join(labels))
