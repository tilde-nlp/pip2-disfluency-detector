TOOLS=/data/Projekti/tools

1grams 2grams 3grams 4grams 5grams 6grams: train.txt
	$(TOOLS)/text/parallel/run.sh train.txt 4 "$(TOOLS)/text/normalizer.py -n -c -l -r lv" |\
	/opt/srilm/bin/i686-m64/ngram-count -text - -no-eos -no-sos -order 6 -write1 1grams -write2 2grams -write3 3grams -write4 4grams -write5 5grams -write6 6grams


atoms:
	echo "[CLS]" > atoms
	echo "[SEP]" >> atoms

%.tag: %.txt bpe.model atoms 1grams
	# line=`wc -l $< | cut -f1` ; \
	# line=((line / 3)) ; \
        # head -n $line $< > tmp2 
	$(TOOLS)/text/parallel/run.sh $< 4 "$(TOOLS)/text/normalizer.py -n -c -l -r lv" > tmp
	$(TOOLS)/text/parallel/run.sh tmp 4 "./create_tag.py 6" > tmp2
	$(TOOLS)/text/parallel/run.sh tmp2 4 "$(TOOLS)/text/segmentation/subword-nmt/apply_bpe.py -c bpe.model --atoms atoms" |\
	./fix_bpe_tags.py > $@
	rm tmp tmp2	
	
%.cls: %.txt bpe.model atoms
	# line=`wc -l $< | cut -f1` ; \
	# line=((line / 3)) ; \
        # tail -n +$line $< > tmp2 
	$(TOOLS)/text/parallel/run.sh $< 4 "$(TOOLS)/text/normalizer.py -n -c -l -r lv" > tmp
	$(TOOLS)/text/parallel/run.sh tmp 4 "./create_cls.py 6" > tmp2
	$(TOOLS)/text/parallel/run.sh tmp2 4 "$(TOOLS)/text/segmentation/subword-nmt/apply_bpe.py -c bpe.model --atoms atoms" > $@
	rm tmp tmp2

vocab: dev.tag dev.cls test.tag train.tag train.cls
	cat dev.tag dev.cls test.tag train.tag train.cls | $(TOOLS)/text/dict.py |\
	$(TOOLS)/text/dict_truncate.sh 95 | cut -f1 | sort > vocab

%.testtag: %.txt bpe.model atoms
	./prepare_tag.py < $< |\
	$(TOOLS)/text/segmentation/subword-nmt/apply_bpe.py -c bpe.model --atoms atoms |\
	./fix_bpe_tags.py > $@
