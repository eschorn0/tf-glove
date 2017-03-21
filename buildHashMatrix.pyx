from __future__ import absolute_import, division, print_function
import time
import numpy as np
cimport numpy as np
try: xrange
except NameError: xrange = range  # Python 3


def buildHashMatrix(float prop, int adjacency, int maxCollisions, float hash, int verbose):

    cdef int hashSize = int(2**hash)
    print("Hash arrays size: {}".format(hashSize))

    def addToMatrix(int x, int y, float adder):
        cdef int lo, hi, index
        cdef long collisions = 0
        lo, hi = (x, y) if x < y else (y, x)
        while True:
            index = (lo  + hi * 987654 + collisions * 4321) % hashSize
            if cmLo[index] == 0 or (cmLo[index] == lo and cmHi[index] == hi):
                cmLo[index] = lo
                cmHi[index] = hi
                cmCt[index] += adder
                return 0
            else:
                if collisions == maxCollisions:
                    if verbose > 0: print("Hit max collisions of {}, abandoning storage of lo={} hi={}".format(maxCollisions, lo, hi))
                    return 1
                collisions += 1

    tokenizedCorpus = np.load("./data/tokenizedCorpus.npy")
    numTokens = int(tokenizedCorpus.shape[0]*prop)
    print("Total number of tokens to process {}".format(numTokens))

    Lo = np.zeros(hashSize, dtype=np.int32)
    cdef int [:] cmLo = Lo
    Hi = np.zeros(hashSize, dtype=np.int32)
    cdef int [:] cmHi = Hi
    Ct = np.zeros(hashSize, dtype=np.float32)
    cdef float [:] cmCt = Ct

    cdef long collisions = 0
    cdef int index, delta
    startTime = time.time()
    for index in xrange(adjacency, numTokens-adjacency):
        if index > adjacency and (index - adjacency) % (numTokens//25) == 0:
            sofar = time.time() - startTime
            now = time.strftime("%H:%M:%S")
            print("{}  Finished {:>2.0f}% at {:.0f} ips with {} total abandoned stores".format(now, 100.0*index/numTokens, index/sofar, collisions))
        for delta in xrange(-adjacency, adjacency+1):
            if delta == 0: continue
            collisions += addToMatrix(tokenizedCorpus[index], tokenizedCorpus[index + delta], abs(1. / delta))

    totalStores = numTokens * 2 * adjacency
    print("Percentage of abandoned stores {:.1%}".format(1.0*collisions/totalStores))

    return Lo, Hi, Ct, collisions
