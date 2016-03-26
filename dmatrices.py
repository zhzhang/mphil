import cPickle as pickle
import numpy as np
import time
from scipy import linalg

class DMatrices(object):
    def __init__(self, matrices, wordmap):
        print "Loading matrices at %s" % matrices
        t = time.time()
        matrices = open(matrices, 'r')
        self.matrices = pickle.load(matrices)
        matrices.close()
        print "Matrices loaded in %0.3f seconds" % (time.time() - t)
        print "Loading wordmap at %s" % wordmap 
        wordmap = open(wordmap, 'r')
        self.wordmap = pickle.load(wordmap)
        wordmap.close()
        self.dimension = len(self.wordmap)

    # Computes representativeness from word x to word y.
    def repres(self, x, y):
        try:
            x = self.wordmap[x]
            y = self.wordmap[y]
        except KeyError:
            return -1
        X = self._load_matrix(x)
        Y = self._load_matrix(y)
        r = self._compute_entropy(X, Y)
        return 1 / (1 + r)

    def _load_matrix(self, target):
        target = self.matrices[target]
        output = np.zeros([self.dimension, self.dimension])
        for pair in target:
            x,y = pair
            output[x,y] = target[pair]
            output[y,x] = target[pair]
        return output / linalg.norm(output)

    def _compute_entropy(self, X, Y):
        logX = linalg.logm(X)
        logY = linalg.logm(Y)
        a = np.trace(X * logX)
        b = np.trace(X * logY)
        return a - b

if __name__ == '__main__':
    pass

