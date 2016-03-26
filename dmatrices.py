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
        xmatrix, ymatrix = self._load_pair(x, y)
        r = self._compute_entropy(xmatrix, ymatrix)
        return 1 / (1 + r)

    def _load_pair(self, x, y):
        x = self.matrices[x]
        y = self.matrices[y]
        basis = set()
        for a,b in x.keys() + y.keys():
            basis.add(a)
            basis.add(b)
        basis = list(basis)
        basis.sort()
        basis_map = {}
        for i, b in enumerate(basis):
            basis_map[b] = i
        return self._get_matrix(x, basis_map),\
          self._get_matrix(y, basis_map)

    def _get_matrix(self, target, basis_map):
        output = np.zeros([len(basis_map), len(basis_map)])
        for pair in target:
            x,y = pair
            x_ind = basis_map[x]
            y_ind = basis_map[y]
            output[x_ind,y_ind] = target[pair]
            output[y_ind,x_ind] = target[pair]
        return output / linalg.norm(output)

    def _compute_entropy(self, X, Y):
        logX = linalg.logm(X)
        logY = linalg.logm(Y)
        a = np.trace(X * logX)
        b = np.trace(X * logY)
        return a - b

if __name__ == '__main__':
    pass

