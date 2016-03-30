import cPickle as pickle
import numpy as np
import time
from scipy import linalg

ZERO_THRESH = 1e-12

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
        if xmatrix == None:
            print "Word not found in matrices: %s" % x
        if ymatrix == None:
            print "Word not found in matrices: %s" % y
        if xmatrix == None or ymatrix == None:
            return None
        t = time.time()
        r = self._compute_entropy(xmatrix, ymatrix)
        print "Entropy computed in %0.3f seconds" % (time.time() - t)
        return 1 / (1 + r)

    def _get_basis(self, *args):
        basis = set()
        for target in args:
            for a,b in target.keys():
                basis.add(a)
                basis.add(b)
        basis = list(basis)
        basis.sort()
        basis_map = {}
        for i, b in enumerate(basis):
            basis_map[b] = i
        return basis_map

    def _load_single(self, x):
        x = self.matrices[x]
        basis_map = self._get_basis(x)
        return self._get_matrix(x, basis_map)

    def _load_pair(self, x, y):
        x = self.matrices[x]
        y = self.matrices[y]
        basis_map = self._get_basis(x,y)
        return self._get_matrix(x, basis_map),\
          self._get_matrix(y, basis_map)

    def _get_matrix(self, target, basis_map):
        if len(target) == 0:
            return None
        output = np.zeros([len(basis_map), len(basis_map)])
        for pair in target:
            x,y = pair
            x_ind = basis_map[x]
            y_ind = basis_map[y]
            output[x_ind,y_ind] = target[pair]
            output[y_ind,x_ind] = target[pair]
        return output / linalg.norm(output)

    def _compute_entropy(self, X, Y):
        eigx, vecx = np.linalg.eig(X)
        eigy, vecy = np.linalg.eig(Y)
        eigx = np.real(eigx)
        eigy = np.real(eigy)
        tracex = 0.0
        for lamx in eigx:
            if not lamx < ZERO_THRESH:
                tracex += lamx * np.log(lamx)
        XV = np.dot(X,vecy)
        tracey = 0.0
        for i,lamy in enumerate(eigy):
            tmp = np.real(np.dot(XV[:,i], vecy[:,i]))
            if tmp < ZERO_THRESH:
                continue
            elif lamy < ZERO_THRESH:
                return float('inf')
            else:
                tracey += tmp * np.log(lamy)
        return tracex - tracey

if __name__ == '__main__':
    pass

