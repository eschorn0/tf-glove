from __future__ import absolute_import, division, print_function
import pickle, sys
import numpy as np

def buildDictionaries(vocabFileName='./data/vocab.pickle', vocabSize=0):

    # Load vocab
    with open(vocabFileName, 'rb') as fn:
        data = pickle.load(fn)
    vocab = data['vocab']

    if vocabSize == 0: vocabSize = len(vocab) + 2

    # Build wordDictionary
    common = vocab.most_common(vocabSize - 2)  # Less '-stop-' and 'UNK'
    wordDictionary = {'UNK' : 0, '-stop-': 1}
    for item in common:
        wordDictionary[item[0]] = len(wordDictionary)

    # Build reverseDictionary
    if sys.version_info[0] < 3:
        reverseDictionary = {v: k for k, v in wordDictionary.iteritems()}
    else:
        reverseDictionary = {v: k for k, v in wordDictionary.items()}

    return vocab, wordDictionary, reverseDictionary


def measureAnalogies(wordVecs, testSet):

    wNorm = np.zeros(wordVecs.shape)
    d = (np.sum(wordVecs ** 2, 1) ** (0.5))
    wNorm = (wordVecs.T / d).T

    numTest = testSet.shape[0]
    top1 = top2 = 0
    testVecs = wNorm[testSet[:, 1]] - wNorm[testSet[:, 0]] + wNorm[testSet[:, 2]]

    for i in xrange(numTest):
        dist = np.dot(wNorm, testVecs[i].T)
        dist[testSet[i, 0]] = -np.Inf
        dist[testSet[i, 1]] = -np.Inf
        dist[testSet[i, 2]] = -np.Inf
        prediction1 = np.argmax(dist, 0).flatten()
        if testSet[i, 3] == prediction1: top1 += 1
        else:
            dist[prediction1] = -np.Inf
            prediction2 = np.argmax(dist, 0).flatten()
            if testSet[i, 3] == prediction2: top2 += 1

    top2 += top1
    top1 /= numTest
    top2 /= numTest
    print("Word Analogies -- Top 1: {:.1%}    Top 1+2: {:.1%}".format(top1, top2))
    return
