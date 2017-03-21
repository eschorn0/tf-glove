#!/usr/bin/env python

#
# Scans through extracted wikipedia articles to build a vocabulary with usage counts
#

from __future__ import absolute_import, division, print_function
import argparse, bz2, collections, datetime, glob, os, pickle, re, textwrap, time
from multiprocessing import Pool
from nltk.tokenize import sent_tokenize, word_tokenize


def getConfig(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpDir",   default="/tmp/AA", help="Source data directory",                   type=str)
    parser.add_argument("--numCPU",    default="4",       help="Number of CPUs to run in parallel",       type=int)
    parser.add_argument("--numFiles",  default="-1",      help="Number of files to process (-1 for all)", type=int)
    parser.add_argument("--vocabSize", default="-1",      help="Vocab size limit (-1 for all",            type=int)
    config = parser.parse_args()
    configString = "Configuration parameters: "
    configString += ", ".join(["{}={}".format(attr, value) for attr, value in sorted(config.__dict__.items())])
    now = datetime.datetime.now().strftime("%a %b %d %Y %H:%M")
    if verbose: print("# {} at {}".format(os.path.basename(__file__), now))
    if verbose: print(textwrap.fill(configString, 100, initial_indent="# ", subsequent_indent="# | "), end="\n\n")
    return config


def extractVocab(filename):
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

    singleBaddies = collections.Counter()
    singleVocab = collections.Counter()

    # Split into list of sentences
    sentences = sent_tokenize(text.lower())

    # Parse each sentence, word by word
    for sentence in sentences:
        wordTokens = word_tokenize(sentence)
        newTokens = list()
        for token in wordTokens:
            if token in MAP_TOKENS:
                singleBaddies.update(token)
                newTokens.append(MAP_TOKENS[token])
                continue
            if re.search(NON_CHAR, token):  # Note: hyphenated words will be ignored
                singleBaddies.update(token)
                continue
            newTokens.append(token)
        singleVocab.update(newTokens)
    return singleVocab, singleBaddies



if __name__ == "__main__":
    start = time.time()
    config = getConfig()
    bz2Files = sorted(glob.glob(config.corpDir + "/wiki*.bz2"))[0:config.numFiles]

    # Extract files in parallel
    pool = Pool(config.numCPU)
    results = pool.map(extractVocab, bz2Files)

    vocab = collections.Counter()
    baddies = collections.Counter()

    # Combine parallel results
    for result in results:
        vocab.update(result[0])
        baddies.update(result[1])

    if config.vocabSize > 0:
        mostCommon = vocab.most_common(config.vocabSize)
        newVocab = collections.Counter()
        for item in mostCommon:
            newVocab[item[0]] = item[1]
        vocab = newVocab

    if not os.path.exists('./data'):
        os.makedirs('./data')

    print("{} unique words".format(len(vocab)))
    data = {'vocab': vocab}
    with open('./data/vocab.pickle', 'wb') as fn:
        pickle.dump(data, fn, protocol=pickle.HIGHEST_PROTOCOL)

    print("{} unique and ignored artifacts (baddies)".format(len(baddies)))
    data = {'baddies': baddies}
    with open('./data/baddies.pickle', 'wb') as fn:
        pickle.dump(data, fn, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nFinished processing {} files in {:.0f} seconds\n\n".format(len(bz2Files), time.time() - start))
