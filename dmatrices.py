import cPickle as pickle
import numpy as np
import os
import time
from scipy import linalg
from multiprocessing import Pool

ZERO_THRESH = 1e-12

class DMatrices(object):
    def __init__(self, matrices_path, wordmap):
        print "Loading matrices at %s" % matrices_path
        t = time.time()
        self.matrices_path = matrices_path
        matrices = open(matrices_path, 'r')
        self.matrices = pickle.load(matrices)
        matrices.close()
        print "Matrices loaded in %0.3f seconds" % (time.time() - t)
        print "Loading wordmap at %s" % wordmap 
        wordmap = open(wordmap, 'r')
        self.wordmap = pickle.load(wordmap)
        wordmap.close()
        self.dimension = len(self.wordmap)

    def get_eigenvectors(self, words, num_cores=1):
        pool = Pool(processes=num_cores)
        args = []
        eigen_path = os.path.join(os.path.dirname(self.matrices_path), 'eigenvectors')
        if not os.path.exists(eigen_path):
            os.makedirs(eigen_path)
        for word in words:
            try:
                word_id = self.wordmap[word]
            except KeyError:
                continue
            matrix, basis_map = self._load_single(word_id)
            if matrix == None:
                continue
            args.append((word, matrix, basis_map, eigen_path))
        pool.map(get_eigenvectors_worker, args)
        pool.close()
        pool.join()

    def repres(self, pairs):
        words = set()
        for a,b in pairs:
            words.add(a)
            words.add(b)
        self.get_eigenvectors(words)
        """
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
        """

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
        return self._get_matrix(x, basis_map), basis_map

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
        VXV = np.dot(vecy.T, np.dot(X,vecy))
        tracey = 0.0
        for i, lamy in enumerate(eigy):
            tmp = VXV[i,i]
            if tmp < ZERO_THRESH:
                continue
            elif lamy < ZERO_THRESH:
                return float('inf')
            else:
                tracey += tmp * np.log(lamy)
        return tracex - tracey

def get_eigenvectors_worker(args):
    word, matrix, basis_map, eigen_path = args
    print "Computing eigenvectors for: %s" % word
    eigx, vecx = np.linalg.eig(matrix)
    with open(os.path.join(eigen_path, word + '.pkl'), 'w+') as f:
        pickle.dump((eigx, vecx, basis_map), f)

if __name__ == '__main__':
    pass

