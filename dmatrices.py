import cPickle as pickle
import density_matrix_pb2
import numpy as np
import os
import time
from scipy import linalg
from multiprocessing import Pool

ZERO_THRESH = 1e-12 # Convention is to take x nonzero if x >= ZERO_THRESH

class DMatrices(object):
    def __init__(self, matrices_path, dense=False):
        self._matrices_path = matrices_path
        self.dense = dense
        self._eigen_path = os.path.join(os.path.join(self._matrices_path, 'eigenvectors'))
        self._load_wordmap(os.path.join(matrices_path, "wordmap.txt"))
        self._get_words()

    def _load_wordmap(self, wordmap_path):
        if not os.path.exists(wordmap_path):
            self._wordmap = None
            return
        wordmap = []
        with open(wordmap_path, 'r') as f:
            for line in f:
                word = line.rstrip('\n').lower()
                wordmap.append(word)
        self._wordmap = wordmap

    def _get_words(self):
        self.words = map(lambda x: os.path.splitext(x)[0],\
                filter(lambda x: ".dat" in x, os.listdir(self._matrices_path)))
    
    def get_avg_density(self):
        total = 0
        for word in self.words:
            matrix = self.load_matrix(word)
            total += sum(sum(matrix >= ZERO_THRESH))
        dim = len(matrix)
        print total / float(len(self.words) * dim * dim)

    def get_eigenvectors(self, words, num_processes=1):
        pool = Pool(processes=num_processes)
        args = []
        eigen_path = self._eigen_path
        if not os.path.exists(eigen_path):
            os.makedirs(eigen_path)
        for word in words:
            if os.path.exists(os.path.join(eigen_path, word + ".pkl")):
                continue
            matrix = self.load_matrix(word, smoothed=False)
            if not matrix is None:
                args.append((word, matrix, eigen_path))
        if len(args) == 0:
            return
        results = pool.imap(_get_eigenvectors_worker, args)
        pool.close()
        pool.join()

    def repres(self, pairs, num_processes=1):
        # Compute missing eigenvectors.
        words = set()
        for a,b in pairs:
            if not (a in self.words and b in self.words):
                pass
            words.add(a)
            words.add(b)
        self.get_eigenvectors(words)
        pool = Pool(processes=num_processes)
        args = []
        eigen_path = self._eigen_path
        for i, (a,b) in enumerate(pairs):
            args.append((a, b, eigen_path))
        results = pool.map(_compute_repres_worker, args)
        pool.close()
        pool.join()
        return results

    def print_eigenvectors(self, word, n=1):
        word_eigen_path = os.path.join(self._eigen_path, word + ".pkl")
        if not os.path.exists(word_eigen_path):
            self.get_eigenvectors((word,))
            if not os.path.exists(word_eigen_path):
                print "Eigenvectors not found"
                return
        with open(word_eigen_path, 'rb') as f:
            eig, vec = pickle.load(f)
        for i in range(1,n+1):
            print "Eigenvalue %e:" % eig[-i]
            print " ; ".join(map(lambda x: "%s %e" % x, filter(lambda x: x[1] != 0.0,\
                    sorted(zip(self._wordmap,vec[:,-i]), key=lambda x: x[1], reverse=True))))

    @staticmethod
    def _get_basis(*args):
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

    def load_matrix(self, target, smoothed=False):
        matrix_path = os.path.join(self._matrices_path, target + '.dat')
        if not os.path.exists(matrix_path):
            return None
        with open(matrix_path, 'rb') as f:
            if self.dense:
                matrix = density_matrix_pb2.DMatrixDense()
            else:
                matrix = density_matrix_pb2.DMatrixSparse()
            matrix.ParseFromString(f.read())
        dim = matrix.dimension
        output = np.zeros([dim, dim])
        if self.dense:
            x = 0
            data_iter = iter(matrix.data)
            while x < dim:
                y = x
                while y < dim:
                    val = data_iter.next()
                    output[x,y] = val
                    output[y,x] = val
                    y += 1
                x += 1
        else:
            for entry in matrix.entries:
                x = entry.x
                y = entry.y
                output[x,y] = entry.value
                output[y,x] = entry.value
        if smoothed:
            DMatrices._smooth_matrix(output)
        if np.trace(output) == 0.0:
            return output
        return output / np.trace(output)

    @staticmethod
    def _smooth_matrix(matrix):
        dim = matrix.shape[0]
        for i in range(dim):
            matrix[i,i] = matrix[i,i] + 1e-8

def _load_eigen(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _compute_repres_worker(args):
    (word_x, word_y, eigen_path) = args
    pathx = os.path.join(eigen_path, word_x + '.pkl')
    pathy = os.path.join(eigen_path, word_y + '.pkl')
    if not (os.path.exists(pathx) and os.path.exists(pathy)):
        return None
    eigx, vecx = _load_eigen(pathx)
    eigy, vecy = _load_eigen(pathy)
    #vecx, vecy = _merge_basis(basis_map_x, basis_map_y, vecx, vecy)
    tmp = compute_rel_ent(eigx, vecx, eigy, vecy)
    return (1/(1+tmp[0]), 1/(1+tmp[1]))

def _merge_basis(basis_map_x, basis_map_y, vecx, vecy):
    new_basis_map = basis_map_x.copy()
    index = max(new_basis_map.values())
    for key in basis_map_y:
        if not key in new_basis_map:
            index += 1
            new_basis_map[key] = index
    new_vecx = np.zeros([len(new_basis_map), vecx.shape[0]])
    new_vecx[0:vecx.shape[0], 0:vecx.shape[0]] = vecx
    new_vecy = np.zeros([len(new_basis_map), vecy.shape[0]])
    for key in basis_map_y:
        new_ind = new_basis_map[key]
        new_vecy[new_ind,:] = vecy[basis_map_y[key],:]
    return new_vecx, new_vecy

def compute_rel_ent(eigx, vecx, eigy, vecy):
    trxx = sum([lam_x * np.log(lam_x) if lam_x >= ZERO_THRESH else 0.0 for lam_x in eigx])
    tryy = sum([lam_y * np.log(lam_y) if lam_y >= ZERO_THRESH else 0.0 for lam_y in eigy])
    trxy, tryx = _tr_log(vecx, eigx, vecy, eigy)
    return (trxx - trxy, tryy - tryx)

def _tr_log(A, eiga, B, eigb):
    AtB = np.dot(A.T, B)
    tmp_ab = np.einsum('ij,i,ji->j', AtB, eiga, AtB.T)
    tmp_ba = np.einsum('ij,i,ji->j', AtB.T, eigb, AtB)
    output_ab = 0.0
    for i,lam_b in enumerate(eigb):
        if np.absolute(tmp_ab[i]) < ZERO_THRESH:
            continue
        elif np.absolute(lam_b) < ZERO_THRESH:
            output_ab = -float('inf')
            break
        else:
            output_ab += tmp_ab[i] * np.log(lam_b)
    output_ba = 0.0
    for i,lam_a in enumerate(eiga):
        if np.absolute(tmp_ba[i]) < ZERO_THRESH:
            continue
        elif np.absolute(lam_a) < ZERO_THRESH:
            return output_ab, -float('inf')
        else:
            output_ba += tmp_ba[i] * np.log(lam_a)
    return output_ab, output_ba

def _get_eigenvectors_worker(args):
    word, matrix, eigen_path = args
    print "Computing eigenvectors for: %s" % word
    eig, vec = np.linalg.eigh(matrix)
    with open(os.path.join(eigen_path, word + '.pkl'), 'wb+') as f:
        pickle.dump((eig, vec), f)

if __name__ == '__main__':
    pass

