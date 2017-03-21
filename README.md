tf-glove
========

`tf-glove` is inspired by [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) 
and utilizes Python, Cython and Tensorflow r1.0 on Ubuntu. The primary focus is
on clarity, simplicity and many tunable parameters suitable for exploration.
The co-occurrence matrix construction code relies on a hash, rather than a fixed
word-to-word array, and so can handle nearly unlimited vocabulary sizes.
Vocabulary is not constrained until the actual training phase. Code results
closely replicate those in the paper using the 1.6B token Wikipedia corpus.

`tf-glove` runs best with 32GB of DRAM. A GPU is not necessary for reasonable runtimes.


### Quick start

#### 1. Prerequisite: provide working corpus and clone wikiextractor code
`tf-glove` requires a Wikipedia article dump, test word analogies and the wikiextractor utility. For example:
~~~~
# Download latest ~13GB english Wikipedia dump
wget -O /tmp/enwiki-latest-pages-articles.xml.bz2 https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Download test set
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt

# Install wikiExtractor in a parallel directory
git clone https://github.com/attardi/wikiextractor.git ../wikiextractor
~~~~

#### 2. Run!
`tf-glove` extracts the Wikipedia article dump, builds a vocabulary, tokenizes
both the corpus and test set, and creates the co-occurence matrix for training.
The code may encounter pinch-points with less than 32GB of physical memory. The
Makefile only runs what is necessary to train. Have a look at the `make.log` file
to learn what to expect - at the end we see 63% top-1 accurracy for 100-D vectors.
~~~~
# Run! Sit back and relax...
make train

~~~~


### Code components

Behind the scenes, the Makefile will run the following code components in the
order listed below. The default parameters work great, so no overrides are needed.

* Scans the extracted corpus to derive vocab with counts

~~~~
./1-buildVocab.py --help

usage: 1-buildVocab.py [-h] [--corpDir CORPDIR] [--numCPU NUMCPU]
                       [--numFiles NUMFILES] [--vocabSz VOCABSZ]

optional arguments:
  -h, --help             show this help message and exit
  --corpDir CORPDIR      Source data directory
  --numCPU NUMCPU        Number of CPUs to run in parallel
  --numFiles NUMFILES    Number of files to process (-1 for all)
  --vocabSize VOCABSIZE  Vocab size limit (-1 for all
~~~~

* Using the corpus and its vocab, tokenize it into a large 1D numpy array

~~~~
./2-tokenizeCorpus.py --help

usage: 2-tokenizeCorpus.py [-h] [--corpDir CORPDIR] [--numCPU NUMCPU]
                           [--numFiles NUMFILES]

optional arguments:
  -h, --help           show this help message and exit
  --corpDir CORPDIR    Source data directory
  --numCPU NUMCPU      Number of processes to run in parallel
  --numFiles NUMFILES  Number of files to process (-1 is all)
~~~~

* Tokenize the testSet at word-test.v1.txt

~~~~
# This script takes no parameters
./3-tokenizeTestSet.py
~~~~

* Given our vocabulary sizes of interest, a vanilla co-occurence matrix
is far too big and the construction is far too slow. So we use a hash
structure in Cython. This must be compiled.

~~~~
# Build Cython worker for (4)
python setup.py build_ext --inplace  # Numpy warnings can be ignored
~~~~

* Build the co-occurrence hash matrix

~~~~
./4-createCoMatrix.py --help

usage: 4-createCoMatrix.py [-h] [--hashSize HASHSIZE] [--adjacent ADJACENT]
                           [--maxColl MAXCOLL] [--prop PROP]

optional arguments:
  -h, --help           show this help message and exit
  --hashSize HASHSIZE  2**Size of hash table
  --adjacent ADJACENT  Adjacent words to consider
  --maxColl MAXCOLL    Max collisions before abandoning storage
  --prop PROP          Proportion of corpus to parse
~~~~

* Finally, we can now train on the co-occurrence data. This is the best place
to constrain the vocabulary size to 400k if aiming to replicate paper results.

~~~~
./5-glove.py --help

usage: 5-glove.py [-h] [--batchSize BATCHSIZE] [--countMax COUNTMAX]
                  [--countMin COUNTMIN] [--embedSize EMBEDSIZE]
                  [--epochs EPOCHS] [--initRange INITRANGE]
                  [--learnRate LEARNRATE] [--logDir LOGDIR]
                  [--scalingF SCALINGF] [--topWords TOPWORDS]
                  [--vocabSize VOCABSIZE]
                  [{train,retrain,test}]

positional arguments:
  {train,retrain,test}   Cmd

optional arguments:
  -h, --help             show this help message and exit
  --batchSize BATCHSIZE  Batch size
  --countMax COUNTMAX    Max count
  --countMin COUNTMIN    Min count
  --embedSize EMBEDSIZE  Embedding size
  --epochs EPOCHS        Number of epochs to train
  --initRange INITRANGE  Init range
  --learnRate LEARNRATE  Learning rate
  --logDir LOGDIR        Logs directory
  --scalingF SCALINGF    Scaling factor
  --topWords TOPWORDS    Num top words to ignore
  --vocabSize VOCABSIZE  Vocabulary size
~~~~

### Misc notes

* The build directory holds intermediate Cython files
* The data directory holds intermediate files necessary to generate training data
* `make clean` generates a fresh start (but doesn't delete /tmp data)
* NB: Ubuntu clears the /tmp directory on reboot, potentially wiping the Wikipedia corpus stored there