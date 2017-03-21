#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import argparse, datetime, os, textwrap, time
import numpy as np

from buildHashMatrix import buildHashMatrix


def getConfig(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashSize", default=30.0, help="2**Size of hash table",                    type=float)
    parser.add_argument("--adjacent", default=10,   help="Adjacent words to consider",               type=int)
    parser.add_argument("--maxColl",  default=1000, help="Max collisions before abandoning storage", type=int)
    parser.add_argument("--prop",     default=1.00, help="Proportion of corpus to parse",            type=float)
    config = parser.parse_args()
    configString = "Configuration parameters: "
    configString += ", ".join(["{}={}".format(attr, value) for attr, value in sorted(config.__dict__.items())])
    now = datetime.datetime.now().strftime("%a %b %d %Y %H:%M")
    if verbose: print("# {} at {}".format(os.path.basename(__file__), now))
    if verbose: print(textwrap.fill(configString, 100, initial_indent="#", subsequent_indent="# | ") + '\n')
    return config



if __name__ == "__main__":
    startTime = time.time()
    config = getConfig()

    cmLo, cmHi, cmCt, collisions = buildHashMatrix(prop=config.prop, adjacency=config.adjacent,
                                                maxCollisions=config.maxColl, hash=config.hashSize, verbose=0)

    pairs = np.stack((cmHi[cmCt > 0], cmLo[cmCt > 0]))

    np.save("./data/pairs.npy", pairs)
    np.save("./data/counts.npy", cmCt[cmCt > 0])
    finishDelta = time.time() - startTime
    print("Finished in {:.0f} seconds with {} abandoned stores".format(finishDelta, collisions))
    print("Hash table density {}".format(pairs.shape[1] / (2 ** config.hashSize)))
    print("Co-occurrence matrix entries: {}".format(pairs.shape))
    print("cmCt sum {}".format(cmCt.sum()))




