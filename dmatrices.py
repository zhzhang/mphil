import cPickle as pickle
import numpy as np
import os
import struct
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
        self._load_parameters(os.path.join(matrices_path, "parameters.txt"))
        self._get_words()

    def _load_wordmap(self, wordmap_path):
        if not os.path.exists(wordmap_path):
            self._wordmap = None
            return
        wordmap = []
        with open(wordmap_path, 'r') as f:
            for line in f.readlines():
                word = line.lower()
                wordmap.append(word)
        self._wordmap = wordmap

    def _load_parameters(self, parameters_path):
        with open(parameters_path, 'r') as f:
            self.dimension = int(f.readline().split(' ')[1])

    def _get_words(self):
        self.words = map(lambda x: os.path.splitext(x)[0],\
                filter(lambda x: ".bin" in x, os.listdir(self._matrices_path)))
    
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
            word = word.lower()
            if not os.path.exists(os.path.join(eigen_path, word + ".pkl")):
                matrix_path = os.path.join(self._matrices_path, word + '.bin')
                args.append((matrix_path, self.dimension, eigen_path, self.dense))
        if len(args) == 0:
            return
        results = pool.imap(_get_eigenvectors_worker, args)
        pool.close()
        pool.join()

    def _compute_measure(self, pairs, measure, num_processes):
        # Compute missing eigenvectors.
        words = set()
        for a,b in pairs:
            if not (a in self.words and b in self.words):
                pass
            words.add(a)
            words.add(b)
        t = time.time()
        self.get_eigenvectors(words)
        print "Eigenvectors computed in %d seconds" % (time.time() - t)
        pool = Pool(processes=num_processes)
        args = []
        for i, (a,b) in enumerate(pairs):
            args.append((a, b, measure, self._eigen_path, self.dense))
        results = pool.map(_compute_measure_worker, args)
        pool.close()
        pool.join()
        return results


    def repres(self, pairs, num_processes=1):
        return self._compute_measure(pairs, "repres", num_processes)

    def weeds_prec(self, pairs, num_processes=1):
        return self._compute_measure(pairs, "weedsprec", num_processes)

    def print_eigenvectors(self, word, n=1):
        word_eigen_path = os.path.join(self._eigen_path, word + ".pkl")
        if not os.path.exists(word_eigen_path):
            self.get_eigenvectors((word,))
            if not os.path.exists(word_eigen_path):
                print "Eigenvectors not found"
                return
        with open(word_eigen_path, 'rb') as f:
            eig, vec = pickle.load(f)
        for i in xrange(1,n+1):
            print "Eigenvalue %e:" % eig[-i]
            print " ; ".join(map(lambda x: "%s %e" % x, filter(lambda x: x[1] != 0.0,\
                    sorted(zip(self._wordmap,vec[:,-i]), key=lambda x: x[1], reverse=True))))

    def load_matrix(self, target, smoothed=False):
        matrix_path = os.path.join(self._matrices_path, target + '.bin')
        if not os.path.exists(matrix_path):
            return None
        if self.dense:
            return _load_matrix(matrix_path, self.dimension, smoothed)
        else:
            return _load_matrix_sparse(matrix_path, self.dimension, smoothed)

    @staticmethod
    def _smooth_matrix(matrix):
        dim = matrix.shape[0]
        for i in xrange(dim):
            matrix[i,i] = matrix[i,i] + 1e-8

####################
# HELPER FUNCTIONS #
####################

def _load_matrix_dense(matrix_path, dimension, smoothed):
    matrix_file = open(matrix_path, 'rb')
    if dense:
        output = np.zeros([dimension, dimension])
        x = 0
        while x < dimension:
            y = x
            while y < dimension:
                (value,) = struct.unpack('>f', matrix_file.read(4))
                output[x,y] = value
                output[y,x] = value
                y += 1
            x += 1
    if smoothed:
        DMatrices._smooth_matrix(output)
    return output / np.trace(output)

def _load_matrix_sparse(matrix_path, dimension, smoothed):
    matrix_data = {}
    matrix_file = open(matrix_path, 'rb')
    while True:
        data = matrix_file.read(12)
        if len(data) < 12:
            break
        x, y, value = struct.unpack('>iif', data)
        matrix_data[(x,y)] = value
    basis = _get_basis(matrix_data)
    output = np.zeros([len(basis), len(basis)])
    for x,y in matrix_data:
        xind = basis[x]
        yind = basis[y]
        output[xind,yind] = matrix_data[(x,y)]
        output[yind,xind] = matrix_data[(x,y)]
    matrix_file.close()
    if smoothed:
        DMatrices._smooth_matrix(output)
    return output / np.trace(output), basis

def _get_basis(matrix_data):
    basis_set = set()
    for x,y in matrix_data:
        basis_set.add(x)
        basis_set.add(y)
    return dict([(t[1], t[0]) for t in enumerate(sorted(list(basis_set)))])

def _load_eigen(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _merge_basis(basis_map_x, basis_map_y, vecx, vecy):
    new_basis_map = basis_map_x.copy()
    index = max(new_basis_map.values())
    for key in basis_map_y:
        if not key in new_basis_map:
            index += 1
            new_basis_map[key] = index
    new_vecx = np.zeros([len(new_basis_map), vecx.shape[1]])
    new_vecx[0:vecx.shape[0], :] = vecx
    new_vecy = np.zeros([len(new_basis_map), vecy.shape[1]])
    for key in basis_map_y:
        new_ind = new_basis_map[key]
        new_vecy[new_ind,:] = vecy[basis_map_y[key],:]
    return new_vecx, new_vecy

def _compute_measure_worker(args):
    (word_x, word_y, measure, eigen_path, dense) = args
    pathx = os.path.join(eigen_path, word_x + '.pkl')
    pathy = os.path.join(eigen_path, word_y + '.pkl')
    if not (os.path.exists(pathx) and os.path.exists(pathy)):
        return None
    if dense:
        eigx, vecx = _load_eigen(pathx)
        eigy, vecy = _load_eigen(pathy)
    else:
        eigx, vecx, basisx = _load_eigen(pathx)
        eigy, vecy, basisy  = _load_eigen(pathy)
        vecx, vecy = _merge_basis(basisx, basisy, vecx, vecy)
    if measure == "repres":
        tmp = compute_rel_ent(eigx, vecx, eigy, vecy)
        return (1/(1+tmp[0]), 1/(1+tmp[1]))
    elif measure == "weedsprec":
        return compute_weeds_prec(eigx, vecx, eigy, vecy)

def compute_rel_ent(eigx, vecx, eigy, vecy):
    trxx = sum([lam_x * np.log(lam_x) if lam_x >= ZERO_THRESH else 0.0 for lam_x in eigx])
    tryy = sum([lam_y * np.log(lam_y) if lam_y >= ZERO_THRESH else 0.0 for lam_y in eigy])
    trxy, tryx = _compute_cross_entropy(vecx, eigx, vecy, eigy)
    return (trxx - trxy, tryy - tryx)

def _compute_cross_entropy(A, eiga, B, eigb):
    # Project the eigenspaces onto each other.
    AtB = np.dot(A.T, B)
    # Check containment of one eigenspace inside the other.
    projBA = np.linalg.norm(AtB, axis = 0) # proj of B eigvecs onto A eig space
    spannedBA = np.all(np.absolute(projBA - np.ones(projBA.shape)) < ZERO_THRESH)
    projAB = np.linalg.norm(AtB, axis = 1)
    spannedAB = np.all(np.absolute(projAB - np.ones(projAB.shape)) < ZERO_THRESH)
    # Compute cross entropy A -> B
    if spannedAB:
        output_ab = 0.0
        tmp_ab = np.einsum('ij,i,ji->j', AtB, eiga, AtB.T)
        for i,lam_b in enumerate(eigb):
            if np.absolute(tmp_ab[i]) < ZERO_THRESH:
                continue
            else:
                output_ab += tmp_ab[i] * np.log(lam_b)
    else:
        output_ab = -float('inf')
    # Compute cross entropy B -> A
    if spannedBA:
        tmp_ba = np.einsum('ij,i,ji->j', AtB.T, eigb, AtB)
        output_ba = 0.0
        for i,lam_a in enumerate(eiga):
            if np.absolute(tmp_ba[i]) < ZERO_THRESH:
                continue
            else:
                output_ba += tmp_ba[i] * np.log(lam_a)
    else:
        output_ba = -float('inf')
    return output_ab, output_ba

def compute_weeds_prec(eiga, A, eigb, B):
    # Project the eigenspaces onto each other.
    AtB = np.dot(A.T, B)
    projAB = np.linalg.norm(AtB, axis = 1)
    numerator = sum([projAB[i] * eiga[i] for i in range(len(eiga))])
    # Denominator
    denominator = sum(eiga)
    return numerator / denominator

def _get_eigenvectors_worker(args):
    matrix_path, dimension, eigen_path, dense = args
    if not os.path.exists(matrix_path):
        return
    if dense:
        matrix = _load_matrix_dense(matrix_path, dimension, False)
    else:
        matrix, basis = _load_matrix_sparse(matrix_path, dimension, False)
    eig, vec = np.linalg.eigh(matrix)
    index = len(eig)
    total = 0.0
    while index >= 0 and total < (1.0 - ZERO_THRESH):
        index -= 1
        total += eig[index]
    output_eig = eig[index:]
    tmp = sum(np.absolute(eig[:index]))
    word = os.path.splitext(os.path.basename(matrix_path))[0]
    if tmp >= 0.01:
        # TODO: log a warning here if there is not a clean cutoff in eigenvalues.
        print "Warning: total eigenvalue error is greater than 1%"
    output_vec = vec[:,index:]
    with open(os.path.join(eigen_path, word + '.pkl'), 'wb+') as f:
        if dense:
            pickle.dump((output_eig, output_vec), f)
        else:
            pickle.dump((output_eig, output_vec, basis), f)

if __name__ == '__main__':
    pass

