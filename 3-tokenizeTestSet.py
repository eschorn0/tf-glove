#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import datetime, os
import numpy as np
import utils

vocab, wordDictionary, reverseDictionary = utils.buildDictionaries()

with open("word-test.v1.txt") as fn:
    testText = fn.readlines()

testSet = []
tooMany = 0; okLine = 0; okWords = 0
for line in testText:
    if line.startswith(':'):
        tooMany += 1
        continue
    words = line.lower().split()
    if len(words) > 4:
        tooMany += 1  # Header's and (a few) multi-word phrases will be skipped
        continue
    okLine += 1
    if words[0] not in wordDictionary or words[1] not in wordDictionary or \
       words[2] not in wordDictionary or words[3] not in wordDictionary: continue
    testSet.append([wordDictionary[words[0]], wordDictionary[words[1]], wordDictionary[words[2]],
                    wordDictionary[words[3]]])
    okWords += 1
now = datetime.datetime.now().strftime("%a %b %d %Y %H:%M")
print("# {} at {}\n".format(os.path.basename(__file__), now))
print("{} lines in original test set text".format(len(testText)))
print(" less {} lines that are headers or had more than four words".format(tooMany))
print("{} acceptable 4 word lines".format(okLine))
print("{} lines had everything required in the wordDictionary".format(okWords))

testArray = np.asarray(testSet)
testArray = testArray[np.logical_and(np.logical_and(testArray[:,0] < 50000, testArray[:,1] < 50000),
                                     np.logical_and(testArray[:,2] < 50000, testArray[:,3] < 50000)),:]

print("{} items in testSet with word index < 50k\n".format(len(testSet)))

np.save("./data/testSet.npy", testArray)

