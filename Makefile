.DEFAULT_GOAL := info
CORPUS=/tmp/enwiki-latest-pages-articles.xml.bz2
BZ2=/tmp/AA/wiki_00.bz2
WIKIEXTRACTOR=../wikiextractor/WikiExtractor.py

info :
	@echo "To prepare training dataset try:  make prep"
	@echo "To prepare training dataset AND train try:  make train"

prep : ./data/counts.npy ./data/pairs.npy ./data/testSet.npy
train : ./data/counts.npy ./data/pairs.npy ./data/testSet.npy
	./5-glove.py

$(BZ2) : $(CORPUS)
	python $(WIKIEXTRACTOR) -b 150M -c -o /tmp $(CORPUS) 

./data/vocab.pickle : ./1-buildVocab.py $(BZ2)
	./1-buildVocab.py

./data/tokenizedCorpus.npy : ./2-tokenizeCorpus.py ./data/vocab.pickle
	./2-tokenizeCorpus.py

./data/testSet.npy : ./3-tokenizeTestSet.py ./data/tokenizedCorpus.npy ./data/vocab.pickle word-test.v1.txt
	./3-tokenizeTestSet.py

buildHashMatrix.so : buildHashMatrix.pyx
	python setup.py build_ext --inplace  # IGNORE NUMPY WARNINGS

./data/counts.npy : ./4-createCoMatrix.py buildHashMatrix.so ./data/tokenizedCorpus.npy
	./4-createCoMatrix.py

clean :
	rm -rf data
	rm -rf build
	rm -f buildHashMatrix.c buildHashMatrix.so *log
	rm -f *pyc
