#!/usr/bin/env python

#
# Scans extracted wikipedia articles alongside vocab, and generates numpy array of tokens
#

from __future__ import absolute_import, division, print_function
import argparse, bz2, datetime, glob, os, random, re, textwrap, time
from multiprocessing import Pool
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import utils


def getConfig(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpDir",  default="/tmp/AA", help="Source data directory",                  type=str)
    parser.add_argument("--numCPU",   default="4",       help="Number of processes to run in parallel", type=int)
    parser.add_argument("--numFiles", default="-1",      help="Number of files to process (-1 is all)", type=int)
    config = parser.parse_args()
    configString = "Configuration parameters: "
    configString += ", ".join(["{}={}".format(attr, value) for attr, value in sorted(config.__dict__.items())])
    now = datetime.datetime.now().strftime("%a %b %d %Y %H:%M")
    if verbose: print("# {} at {}".format(os.path.basename(__file__), now))
    if verbose: print(textwrap.fill(configString, 100, initial_indent="# ", subsequent_indent="# | "), end="\n\n")
    return config


def buildArray(filename):

    NON_CHAR = re.compile("[^a-z0-9]")
    MAP_TOKENS = {"n't": "not", "'d": "would", "'ll": "will", "'ve": "have", "'s": "is", "'re": "are", "'the": "the",
                  "'m": "am"}

    # This will not appear when stdout is redirected to a file
    now = time.strftime("%H:%M:%S")
    print("{}  Processing {}".format(now, filename))

    with bz2.BZ2File(filename, 'r') as fn:
        text = fn.read().decode('utf_8')
    text = re.sub(r'<doc[^>]*>', ' ', text)
    text = re.sub(r'</doc>', ' ', text)
    retList = []

    # Split into list of sentences
    sentences = sent_tokenize(text.lower())

    # Parse each sentence, word by word
    for sentence in sentences:
        wordTokens = word_tokenize(sentence)
        for token in wordTokens:
            if token in MAP_TOKENS:
                retList.append(wordDictionary[MAP_TOKENS[token]])
                continue
            if re.search(NON_CHAR, token):
                continue
            if token in wordDictionary:
                retList.append(wordDictionary[token])
            else:
                retList.append(wordDictionary['UNK'])
        retList.append(wordDictionary['-stop-'])
    return np.asanyarray(retList, dtype=np.int32)



if __name__ == "__main__":
    start = time.time()
    config = getConfig()
    vocab, wordDictionary, reverseDictionary = utils.buildDictionaries()

    bz2Files = sorted(glob.glob(config.corpDir + "/wiki*.bz2"))[0:config.numFiles]

    # Tokenize files in parallel
    pool = Pool(config.numCPU)
    results = pool.map(buildArray, bz2Files)

    tokenizedCorpus = np.concatenate(results)
    print("Shape of tokenized corpus: {}".format(tokenizedCorpus.shape))
    np.save("./data/tokenizedCorpus.npy", tokenizedCorpus)

    # Confirm counts for 20 most common word and 80 others selected at random
    commonVocab = vocab.most_common(20) + random.sample(vocab.most_common(10000), 80)
    for word, count in commonVocab:

        wordNum = wordDictionary[word]
        tokenCount = np.sum(tokenizedCorpus == wordNum)
        if tokenCount != count:
            print("Warning: word={} has vocab count of {} but token count of {}".format(
                word, count, tokenCount))

    print("\nFinished {} files in {:.0f} seconds".format(len(bz2Files), time.time() - start))
