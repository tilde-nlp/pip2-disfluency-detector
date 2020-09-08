# Multi-task self-supervised disfluency detection with Transformer

## WHAT IT IS
Disfluency detection model prototype

## DEPENDENCIES
Training script depends on the following tools that need to be acquired separately:

- [SRILM language modelling toolkit](http://www.speech.sri.com/projects/srilm/download.html). After installation, provide correct path for SRILM in Makefile.

## HOW TO 
### Training

Before training you need to obtain a monolingual text corpus. Such corpus for 
Latvian and some other languages can be obtained on [WMT2017 webpage](http://www.statmt.org/wmt17/translation-task.html#download).

Then, corpus should be tokenized, lowercased, filtered from punctuation and 
other non-word tokens and split into files train.txt, dev.txt, test.txt

You will also need some amount of labelled data that should stored in the file tune.txt in following format:
```
each:O utterance:O in:O separate:O line:O lowercased:O without:O punctuation:O with:O disfluent:D words:O tagged:O with:O D:O
```

Run following command to parse tune.txt file and prepare data for fine-tunning:
```bash
make tune.tag
```

Then, to train a model run:
```bash
make tune.mdl
```

### Inference
To tag a .txt file:
```bash
python3 tag.py --iter tune.mdl < input.txt > output.tags
```

## REFERENCES
Code is based on idea described in:

Wang S, Che W, Liu Q, Qin P, Liu T, Wang WY. Multi-Task Self-Supervised Learning for Disfluency Detection. In: The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI} 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA, February 7-12, 2020 [Internet]. {AAAI} Press; 2020. p. 9193200. 

BPE code is from [subword-nmt](https://github.com/rsennrich/subword-nmt) and licensed under:
The MIT License (MIT)

Copyright (c) 2015 University of Edinburgh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ACKNOWLEDGEMENTS
The research has been supported by the European Regional Development Fund within the research project ”Multilingual Artificial Intelligence Based Human Computer Interaction” No. 1.1.1.1/18/A/148

