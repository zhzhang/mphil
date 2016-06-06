import cPickle as pickle
import numpy as np
import os
import struct
import time
from scipy import linalg
from measures import *
from multiprocessing import Pool

ZERO_THRESH = 1e-12 # Convention is to take x nonzero if x >= ZERO_THRESH

class DMatrices(object):
    def __init__(self, matrices_path, n=None, mode=None):
        self._matrices_path = matrices_path
        self._load_parameters(os.path.join(matrices_path, "parameters.txt"))
        self._n = n
        self._mode = mode
        if self.dense and not n == None:
            raise RuntimeError("Parameter n not allowed when matrices are dense.")
        if not n == None:
            if mode == None:
                self._eigen_path = 'eigenvectors-%d' % n
            else:
                self._eigen_path = 'eigenvectors-%d-%s' % (n, mode)
        else:
            self._eigen_path = "eigenvectors"
        self._load_wordmap(os.path.join(matrices_path, "wordmap.txt"))
        self._get_words()

    # Setup methods for DMatrices object.
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
        self.dense = False
        with open(parameters_path, 'r') as f:
            for line in f.read().splitlines():
                line = line.split(" ")
                if line[0] == "dimension":
                    self.dimension = int(line[1])
                if line[0] == "dense":
                    self.dense = True

    def _get_words(self):
        self.words = map(lambda x: os.path.splitext(x)[0],\
                filter(lambda x: ".bin" in x, os.listdir(self._matrices_path)))
   
    # DMatrices methods
    def get_avg_density(self):
        total = 0
        for word in self.words:
            matrix = self.load_matrix(word)
            total += sum(sum(matrix >= ZERO_THRESH))
        dim = len(matrix)
        print total / float(len(self.words) * dim * dim)

    def get_stats(self):
        rank = []
        features = []
        self.get_eigenvectors(self.words)
        for word in self.words:
            path = os.path.join(self._eigen_path, word + '.pkl')
            if not os.path.exists(path):
                continue
            if self.dense:
                eig, vec, norm = _load_eigen(path)
            else:
                eig, vec, norm, basis = _load_eigen(path)
                features.append(len(basis))
            rank.append(len(eig))
        print "RANK avg %0.2f std %0.2f" % (np.mean(rank), np.std(rank))
        stats_path = os.path.join(self._matrices_path, self._eigen_path, "statistics")
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)
        with open(os.path.join(stats_path, "rank-data.pkl"), 'w') as f:
            pickle.dump(rank, f)
        if not self.dense:
            print "FEATURES avg %0.2f std %0.2f" % (np.mean(features), np.std(features))
            with open(os.path.join(stats_path, "features-data.pkl"), 'w') as f:
                pickle.dump(features, f)

    def get_eigenvectors(self, words, num_processes=1):
        pool = Pool(processes=num_processes)
        args = []
        eigen_path = os.path.join(self._matrices_path, self._eigen_path)
        if not os.path.exists(eigen_path):
            os.makedirs(eigen_path)
        for word in words:
            word = word.lower()
            if not os.path.exists(os.path.join(eigen_path, word + ".pkl")):
                args.append((word, self._matrices_path, eigen_path, self.dimension, self.dense, self._n, self._mode))
        if len(args) == 0:
            return
        results = pool.imap(_get_eigenvectors_worker, args)
        pool.close()
        pool.join()

    def _compute_measure(self, pairs, measure, num_processes, params=None):
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
            args.append((a, b, measure, self._matrices_path, self._eigen_path, self.dense, params))
        results = pool.map(_compute_measure_worker, args)
        pool.close()
        pool.join()
        return results

    def repres(self, pairs, num_processes=1):
        return self._compute_measure(pairs, "repres", num_processes)

    def skew_repres(self, pairs, num_processes=1, alpha=0.99):
        params = {}
        params["alpha"] = alpha
        params["n"] = self._n
        params["mode"] = self._mode
        return self._compute_measure(pairs, "skew", num_processes, params=params)

    def weeds_prec(self, pairs, num_processes=1):
        return self._compute_measure(pairs, "weedsprec", num_processes)

    def clarke_de(self, pairs, num_processes=1):
        return self._compute_measure(pairs, "clarkede", num_processes)

    def inv_cl(self, pairs, num_processes=1):
        return self._compute_measure(pairs, "invcl", num_processes)

    def print_eigenvectors(self, word, n=1):
        word_eigen_path = os.path.join(self._eigen_path, word + ".pkl")
        if not os.path.exists(word_eigen_path):
            self.get_eigenvectors((word,))
            if not os.path.exists(word_eigen_path):
                print "Eigenvectors not found"
                return
        with open(word_eigen_path, 'r') as f:
            eig, vec = pickle.load(f)
        for i in xrange(1,n+1):
            print "Eigenvalue %e:" % eig[-i]
            print " ; ".join(map(lambda x: "%s %e" % x, filter(lambda x: x[1] != 0.0,\
                    sorted(zip(self._wordmap,vec[:,-i]), key=lambda x: x[1], reverse=True))))

    def load_matrix(self, target):
        matrix_path = os.path.join(self._matrices_path, target + '.bin')
        if not os.path.exists(matrix_path):
            return None
        if self.dense:
            return _load_matrix(matrix_path, self.dimension)
        else:
            return _load_matrix_sparse(matrix_path)

    @staticmethod
    def _smooth_matrix(matrix):
        dim = matrix.shape[0]
        for i in xrange(dim):
            matrix[i,i] = matrix[i,i] + 1e-8

####################
# HELPER FUNCTIONS #
####################

def _load_matrix_dense(matrix_path, dimension):
    matrix_file = open(matrix_path, 'rb')
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
    return output / np.trace(output), np.trace(output)

def _load_matrix_sparse(matrix_path, n, mode):
    matrix_data = _load_matrix_data(matrix_path)
    basis = _get_basis(matrix_data, n, mode)
    output = _convert_to_numpy(matrix_data, basis)
    return output / np.trace(output), np.trace(output), basis

def _load_matrix_data(matrix_path):
    matrix_data = {}
    matrix_file = open(matrix_path, 'rb')
    while True:
        data = matrix_file.read(12)
        if len(data) < 12:
            break
        x, y, value = struct.unpack('>iif', data)
        matrix_data[(x,y)] = value
    matrix_file.close()
    return matrix_data

def _convert_to_numpy(matrix_data, basis):
    output = np.zeros([len(basis), len(basis)])
    for x,y in matrix_data:
        if x in basis and y in basis:
            xind = basis[x]
            yind = basis[y]
            output[xind,yind] = matrix_data[(x,y)]
            output[yind,xind] = matrix_data[(x,y)]
    return output

def _get_basis(matrix_data, n, mode):
    if n == None and mode == None:
        basis_set = set()
        for x,y in matrix_data:
            basis_set.add(x)
            basis_set.add(y)
        return dict([(t[1], t[0]) for t in enumerate(sorted(basis_set))])
    elif mode == None: # Cutoff by count.
        basis_set = set()
        for x,y in matrix_data:
            basis_set.add(x)
            basis_set.add(y)
        output = {}
        for i, b in enumerate(sorted(basis_set)):
            if n <= b:
                return output
            output[b] = i
        return output
    elif mode == "prob": # top n by probability
        diag = []
        for pair in matrix_data:
            if pair[0] == pair[1]:
                diag.append((matrix_data[pair], pair[0]))
        diag = sorted(diag, reverse=True)
        output = {}
        if n == None: # Sort and cutoff by probability.
            total = float(sum([x[0] for x in diag]))
            cummulative = 0
            for i, (v, b) in enumerate(diag):
                if cummulative / total >= 0.8:
                    return output
                output[b] = i
                cummulative += v
        else: # Sort by probability, cutoff by count.
            for i, (_, b) in enumerate(diag):
                if i == n:
                    return output
                output[b] = i
            return output

def _get_eigenvectors_worker(args):
    word, matrices_path, eigen_path, dimension, dense, n, mode = args
    word_path = os.path.join(matrices_path, word + ".bin")
    if dense:
        matrix, norm = _load_matrix_dense(word_path, dimension)
    else:
        matrix, norm, basis = _load_matrix_sparse(word_path, n, mode)
    output_eig, output_vec = _compute_eigenvectors(matrix)
    with open(os.path.join(eigen_path, word + '.pkl'), 'w') as f:
        if dense:
            pickle.dump((output_eig, output_vec, norm), f)
        else:
            pickle.dump((output_eig, output_vec, norm, basis), f)

def _compute_eigenvectors(matrix):
    eig, vec = np.linalg.eigh(matrix)
    index = len(eig)
    total = 0.0
    while index >= 0 and total < (1.0 - ZERO_THRESH):
        index -= 1
        total += eig[index]
    output_eig = eig[index:]
    tmp = sum(np.absolute(eig[:index]))
    # Print warnings
    if tmp >= 0.01:
        print "Warning: total eigenvalue error is greater than 1%"
    if index > 0 and eig[index-1] > 0.0 and np.absolute(output_eig[0] / eig[index-1]) < 100:
        print "Warning: cutoff %0.2f not sharp" % np.absolute(output_eig[0] / eig[index-1])
    if output_eig[0] < 0:
        print "Warning: negative eigenvalue included"
    output_vec = vec[:,index:]
    return output_eig, output_vec


def _load_eigen(path):
    with open(path, 'r') as f:
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
    word_x, word_y, measure, matrices_path, eigen_path, dense, params = args
    pathx = os.path.join(matrices_path, eigen_path, word_x + '.pkl')
    pathy = os.path.join(matrices_path, eigen_path, word_y + '.pkl')
    if not (os.path.exists(pathx) and os.path.exists(pathy)):
        return None
    if measure == "skew": # Cannot be computed from just eigenvectors.
        tmp = _compute_skew_divergence(word_x, word_y, matrices_path, eigen_path, dense, params)
        if tmp == None:
            return None
        return (1/(1+tmp[0]), 1/(1+tmp[1]))
    if dense:
        eigx, vecx, normx = _load_eigen(pathx)
        eigy, vecy, normy = _load_eigen(pathy)
    else:
        eigx, vecx, normx, basisx = _load_eigen(pathx)
        eigy, vecy, normy, basisy  = _load_eigen(pathy)
        vecx, vecy = _merge_basis(basisx, basisy, vecx, vecy)
    if measure == "repres":
        tmp = compute_rel_ent(eigx, vecx, eigy, vecy)
        return (1/(1+tmp[0]), 1/(1+tmp[1]))
    elif measure == "weedsprec":
        return compute_weeds_prec(eigx, vecx, eigy, vecy)
    elif measure == "clarkede":
        return compute_clarke_de(eigx, vecx, normx, eigy, vecy, normy)
    elif measure == "invcl":
        forward = compute_clarke_de(eigx, vecx, normx, eigy, vecy, normy)
        backward = compute_clarke_de(eigy, vecy, normy, eigx, vecx, normx)
        return np.sqrt(forward * (1 - backward))

def _compute_skew_divergence(word_x, word_y, matrices_path, eigen_path, dense, params):
    pathx = os.path.join(matrices_path, eigen_path, word_x + '.pkl')
    pathy = os.path.join(matrices_path, eigen_path, word_y + '.pkl')
    if not (os.path.exists(pathx) and os.path.exists(pathy)):
        return None
    if dense: # TODO
        eigx, vecx, normx = _load_eigen(pathx)
        eigy, vecy, normy = _load_eigen(pathy)
    else:
        eigx, vecx, normx, basisx = _load_eigen(pathx)
        eigy, vecy, normy, basisy  = _load_eigen(pathy)
        matrix_path_x = os.path.join(matrices_path, word_x + ".bin")
        matrix_path_y = os.path.join(matrices_path, word_y + ".bin")
        matrix_xy, matrix_yx, basis = _load_skew_sparse(matrix_path_x, matrix_path_y, params['alpha'], params['n'], params['mode'])
        eigxy, vecxy = _compute_eigenvectors(matrix_xy)
        eigyx, vecyx = _compute_eigenvectors(matrix_yx)
        vecx, vecxy = _merge_basis(basisx, basis, vecx, vecxy)
        vecy, vecyx = _merge_basis(basisy, basis, vecy, vecyx)
        relentxy = compute_single_rel_ent(eigx, vecx, eigxy, vecxy)
        relentyx = compute_single_rel_ent(eigy, vecy, eigyx, vecyx)
        return (relentxy, relentyx)

def _load_skew_sparse(matrix_path_x, matrix_path_y, alpha, n, mode):
    # Retrieve matrix data and basis.
    matrix_data_x = _load_matrix_data(matrix_path_x)
    basis_x = _get_basis(matrix_data_x, n, mode)
    matrix_data_y = _load_matrix_data(matrix_path_y)
    basis_y = _get_basis(matrix_data_y, n, mode)
    # Merge basis.
    basis = basis_x.copy()
    index = max(basis.values())
    for key in basis_y:
        if not key in basis:
            index += 1
            basis[key] = index
    # Convert to numpy.
    matrix_x = _convert_to_numpy_pruned(matrix_data_x, basis, basis_x)
    matrix_y = _convert_to_numpy_pruned(matrix_data_y, basis, basis_y)
    matrix_x = matrix_x / np.trace(matrix_x)
    matrix_y = matrix_y / np.trace(matrix_y)
    output_xy = (1-alpha) * matrix_x + alpha * matrix_y
    output_yx = (1-alpha) * matrix_y + alpha * matrix_x
    return output_xy, output_yx, basis

def _convert_to_numpy_pruned(matrix_data, basis, pruning_basis):
    output = np.zeros([len(basis), len(basis)])
    for x,y in matrix_data:
        if x in basis and y in basis\
                and x in pruning_basis and y in pruning_basis:
            xind = basis[x]
            yind = basis[y]
            output[xind,yind] = matrix_data[(x,y)]
            output[yind,xind] = matrix_data[(x,y)]
    return output

