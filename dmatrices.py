import cPickle as pickle
import numpy as np
from scipy import linalg

class Matrices(object):
    def __init__(self, matrices, wordmap):
        matrices = open(matrices, 'r')
        self.matrices = pickle.load(matrices)
        wordmap = open(wordmap, 'r')
        self.wordmap = pickle.load(wordmap)
        self.dimension = len(self.wordmap)

    # Computes representativeness from word x to word y.
    def repr(x, y):
        try:
            x = self.wordmap(x)
            y = self.wordmap(y)
        except KeyError:
            return -1
        X = self._load_matrix(x)
        Y = self._load_matrix(y)
        r = self._compute_entropy(X, Y)
        return 1 / (1 + r)

    def _load_matrix(target):
        target = self.matrices[target]
        output = np.empty([self.dimension, self.dimension])
        for pair in target:
            x,y = pair
            output[x,y] = target[pair]
            output[y,x] = target[pair]
        return output / linalg.norm(output)

    def _compute_entropy(X,Y):
        logX = linalg.logm(X)
        logY = linalg.logm(Y)
        a = linalg.trace(X * logX)
        b = linalg.trace(X * logY)
        return a - b

if __name__ == '__main__':
    pass

