#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import argparse, datetime, os, random, textwrap, time
import numpy as np
import tensorflow as tf
import utils


def getConfig(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize",    default=256,     help="Batch size", type=int)
    parser.add_argument("--countMax",     default=100,     help="Max count", type=int)
    parser.add_argument("--countMin",     default=0.11,    help="Min count", type=float)
    parser.add_argument("--embedSize",    default=100,     help="Embedding size", type=int)
    parser.add_argument("--epochs",       default=40,      help="Number of epochs to train", type=int)
    parser.add_argument("--initRange",    default=0.025,   help="Init range", type=float)
    parser.add_argument("--learnRate",    default=0.032,   help="Learning rate", type=float)
    parser.add_argument("--logDir",       default="---",   help="Logs directory", type=str)
    parser.add_argument("--scalingF",     default=0.75,    help="Scaling factor", type=float)
    parser.add_argument("--topWords",     default=10,      help="Num top words to ignore", type=int)
    parser.add_argument("--vocabSize",    default=400000,  help="Vocabulary size", type=int)
    parser.add_argument('cmd', nargs='?', default="train", help="Cmd", choices=["train", "retrain", "test"], type=str)
    config = parser.parse_args()
    if config.logDir == '---': config.logDir = '/tmp/tf-glove-' + "".join(random.sample('ABCDEFGHIJKLMNPQRSTUVWXYZ', 4)) + "/"
    configString = "Configuration parameters: tf version={}, np version={}, ".format(tf.__version__, np.__version__)
    configString += ", ".join(["{}={}".format(attr, value) for attr, value in sorted(config.__dict__.items())])
    now = datetime.datetime.now().strftime("%a %b %d %Y %H:%M")
    if verbose: print("# {} at {}".format(os.path.basename(__file__), now))
    if verbose: print(textwrap.fill(configString, 100, initial_indent="# ", subsequent_indent="# | "), end="\n\n")
    return config


class DataGen(object):
    def __init__(self, batchSize, countMin, topWords, vocabSize):
        wordPairs = np.load("./data/pairs.npy").astype(np.int32).T
        oCounts = np.load("./data/counts.npy").astype(np.float32).T
        assert wordPairs.shape[0] == oCounts.shape[0], "Length of pairs different than counts"
        print("Original data length {:,}".format(wordPairs.shape[0]))

        # Remove all UNK, -stop- and topWords
        saveIndices = np.logical_and(wordPairs[:,0] > (topWords+2), wordPairs[:,1] > (topWords+2))
        wordPairs = wordPairs[saveIndices, :]
        oCounts = oCounts[saveIndices]
        print("After removing UNK, -stop- and {} top words, data length is now {:,}".format(topWords, wordPairs.shape[0]))

        # Remove all smallish counts
        saveIndices = oCounts >= countMin
        wordPairs = wordPairs[saveIndices, :]
        oCounts = oCounts[saveIndices]
        print("After removing counts less than {}, data length is now {:,}".format(countMin, wordPairs.shape[0]))

        # Trim to correct vocab size
        if vocabSize > 0:
            saveIndices = np.logical_and(wordPairs[:,0] < vocabSize, wordPairs[:,1] < vocabSize)
            wordPairs = wordPairs[saveIndices,:]
            oCounts = oCounts[saveIndices]
            print("With limited vocab size of {}, data length is now {:,}".format(vocabSize, wordPairs.shape[0]))

        # Save data necessary for integer number of correctly sized batches
        self.batchSize = batchSize
        self.numBatches = wordPairs.shape[0] // self.batchSize
        wordPairs = wordPairs[0:self.batchSize*self.numBatches,:]
        oCounts = oCounts[0:self.batchSize*self.numBatches]

        # Shuffle and split into batches -> very fast to stuff into feed_dict
        indices = np.random.permutation(wordPairs.shape[0])
        self.wordI = np.split(wordPairs[indices, 0], self.numBatches)
        self.wordJ = np.split(wordPairs[indices, 1], self.numBatches)
        self.Xij = np.split(oCounts[indices], self.numBatches)

        print("Working dataset length is {:,} split into {:,} equal batches of {}".format(
            wordPairs.shape[0], self.numBatches, self.batchSize))

    def __iter__(self):
        for index in range(self.numBatches):
            yield self.wordI[index], self.wordJ[index], self.Xij[index]


class Model(object):
    def __init__(self, countMax, embedSize, initRange, learnRate, scalingFactor, vocabSize):

        self.Xij = tf.placeholder(tf.float32, shape=[None], name="Xij")

        self.wordI = tf.placeholder(tf.int32, shape=[None], name="wordI")
        self.wordIW = tf.Variable(tf.random_uniform([vocabSize, embedSize], initRange, -initRange), name="wordIW")
        wordIB = tf.Variable(tf.random_uniform([vocabSize], initRange, -initRange), name="wordIB")
        wi = tf.nn.embedding_lookup([self.wordIW], self.wordI, name="wi")
        bi = tf.nn.embedding_lookup([wordIB], self.wordI, name="bi")

        self.wordJ = tf.placeholder(tf.int32, shape=[None], name="wordJ")
        self.wordJW = tf.Variable(tf.random_uniform([vocabSize, embedSize], initRange, -initRange), name="wordJW")
        wordJB = tf.Variable(tf.random_uniform([vocabSize], initRange, -initRange), name="wordJB")
        wj = tf.nn.embedding_lookup([self.wordJW], self.wordJ, name="wj")
        bj = tf.nn.embedding_lookup([wordJB], self.wordJ, name="bj")

        wiwjProduct = tf.reduce_sum(tf.multiply(wi, wj), 1)
        logXij = tf.log(self.Xij)
        dist = tf.square(tf.add_n([wiwjProduct, bi, bj, tf.negative(logXij)]))

        self.scalingFactor = tf.constant([scalingFactor], name="scalingFactor")
        self.countMax = tf.constant([countMax], name="countMax")
        wFactor = tf.minimum(1.0, tf.pow(tf.div(self.Xij, countMax), self.scalingFactor))
        self.loss = tf.reduce_sum(tf.multiply(wFactor, dist))
        tf.summary.scalar("GloVe loss", self.loss)

        self.global_step = tf.Variable(0, trainable=False)
        self.learnRate = tf.Variable(learnRate, trainable=False)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learnRate)\
                                  .minimize(self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver()



if __name__ == "__main__":

    config = getConfig()

    dataGen = DataGen(batchSize=config.batchSize, countMin=config.countMin, topWords=config.topWords,
                      vocabSize=config.vocabSize)
    testSet = np.load("./data/testSet.npy")
    model = Model(countMax=config.countMax, embedSize=config.embedSize, initRange=config.initRange,
                  learnRate=config.learnRate, scalingFactor=config.scalingF, vocabSize=config.vocabSize)

    avgLoss = 50.0
    feed_dict = {model.wordI: [1], model.wordJ: [1], model.Xij: [1]}
    lr = 0  # Safety def

    session = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    session.run(init_op)

    if config.cmd == "retrain" or config.cmd == "test":
        print("Loading model...")
        model.saver.restore(session, config.logDir)

    if config.cmd == "test":
        fWeights, cWeights = session.run([model.wordIW, model.wordJW], feed_dict=feed_dict)
        wordVecs = fWeights + cWeights
        utils.measureAnalogies(wordVecs[0:50000, :], testSet)
        exit()

    if not os.path.exists(config.logDir):
        os.makedirs(config.logDir)

    start = time.time()
    for epoch in xrange(config.epochs):
        for data in dataGen:
            feed_dict = {model.wordI: data[0], model.wordJ: data[1], model.Xij: data[2]}
            _, loss, lr, gs = session.run([model.optimizer, model.loss, model.learnRate, model.global_step],
                                          feed_dict=feed_dict)
            avgLoss = 0.9999 * avgLoss + 0.0001 * loss
            if gs % (dataGen.numBatches // 10) == 0:
                now = time.strftime("%H:%M:%S")
                print("{} Epoch={:0>6.2f}  GS={:5.2e}  LR={:5.2e}  Loss={:4.3f}  Speed={:5.2e}s/sec".format(
                    now, gs / dataGen.numBatches, gs, lr, avgLoss, config.batchSize * gs / (time.time() - start)))

        # At the *END* of epoch XXX cut the learning rate in half
        if epoch == 6 or epoch == 12 or epoch == 18 or epoch == 24 or epoch == 32:
            assign_op = model.learnRate.assign(lr / 2)
            session.run(assign_op)

        if epoch > -1:
            fWeights, cWeights = session.run([model.wordIW, model.wordJW], feed_dict=feed_dict)
            wordVecs = fWeights + cWeights
            utils.measureAnalogies(wordVecs[0:50000, :], testSet)
            np.save("./data/wordVecs.npy", wordVecs)
            model.saver.save(session, save_path=config.logDir)
