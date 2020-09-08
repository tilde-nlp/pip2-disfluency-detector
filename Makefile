TOOLS=/data/Projekti/tools
SHELL=/bin/bash
SRILM=/opt/srilm/bin/i686-m64
BPESIZE=90000

1grams 2grams 3grams 4grams 5grams 6grams: train.txt
	$(SRILM)/ngram-count -text train.txt -no-eos -no-sos -order 6 -write1 1grams -write2 2grams -write3 3grams -write4 4grams -write5 5grams -write6 6grams

atoms:
	echo "[CLS]" > atoms
	echo "[SEP]" >> atoms

bpe.model: train.txt
	./learn_bpe.py -s $(BPESIZE) < $< > $@

%.tag: %.txt bpe.model atoms 1grams
	./create_tag.py 6 < $< |\
	./bpe.py -c bpe.model --atoms atoms |\
	./fix_bpe_tags.py > $@
	
%.cls: %.txt bpe.model atoms
	./create_cls.py 6 < $< |\
	./bpe.py -c bpe.model --atoms atoms > $@

vocab: bpe.model atoms chars
	tail -n +2 bpe.model | sed "s/ //;s/$$/@@/" | sed "s;</w>@@;;" > vocab
	cat chars >> vocab
	cat atoms >> vocab
	sort -u -o vocab vocab

final.mdl: train.cls train.tag dev.cls dev.tag vocab
	python3 train.py

tune.mdl: final.mdl tune.tag
	python3 tune.py --iter final.mdl . tune.tag
