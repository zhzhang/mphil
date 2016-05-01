import cPickle as pickle
import density_matrix_pb2
import numpy as np
import os
import time
from scipy import linalg
from multiprocessing import Pool

ZERO_THRESH = 1e-12

class DMatrices(object):
    def __init__(self, matrices_path, test=False, dense=False):
        self.matrices_path = matrices_path
        self.dense = dense
        self._eigen = {}

    def get_eigenvectors(self, words, num_processes=1):
        pool = Pool(processes=num_processes)
        args = []
        eigen_path = os.path.join(os.path.dirname(self.matrices_path), 'eigenvectors')
        if not os.path.exists(eigen_path):
            os.makedirs(eigen_path)
        exists = []
        for word in words:
            matrix = self.load_matrix(word, smoothed=False)
            if not matrix is None:
                exists.append(word)
                args.append(matrix)
        if len(args) == 0:
            return
        results = pool.map(np.linalg.eigh, args)
        pool.close()
        pool.join()
        for i, word in enumerate(exists):
            self._eigen[word] = results[i]

    def repres(self, pairs, num_processes=1):
        # Compute missing eigenvectors.
        words = set()
        for a,b in pairs:
            words.add(a)
            words.add(b)
        self.get_eigenvectors(words)
        pool = Pool(processes=num_processes)
        args = []
        eigen_path = os.path.join(os.path.dirname(self.matrices_path), 'eigenvectors')
        for i, (a,b) in enumerate(pairs):
            try:
                args.append((self._eigen[a], self._eigen[b]))
            except KeyError:
                args.append(None)
        results = pool.map(_compute_repres_worker, args)
        pool.close()
        pool.join()
        return results

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
        matrix_path = os.path.join(self.matrices_path, target + '.dat')
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
                output[x,y] = entry.val
                output[y,x] = entry.val
        if smoothed:
            DMatrices._smooth_matrix(output)
        return output / np.trace(output)

    @staticmethod
    def _smooth_matrix(matrix):
        dim = matrix.shape[0]
        for i in range(dim):
            matrix[i,i] = matrix[i,i] + 1e-8

def _load_eigen(path):
    with open(path, 'r') as f:
        return pickle.load(f)

def _compute_repres_worker(args):
    if args is None:
        return None
    ((eigx, vecx), (eigy, vecy)) = args
    """
    (word_x, word_y, eigen_path) = args
    pathx = os.path.join(eigen_path, word_x + '.pkl')
    pathy = os.path.join(eigen_path, word_y + '.pkl')
    if not (os.path.exists(pathx) and os.path.exists(pathy)):
        return None
    eigx, vecx = _load_eigen(pathx)
    eigy, vecy = _load_eigen(pathy)
    """
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
    trxx = sum([lam_x * np.log(lam_x) if lam_x > ZERO_THRESH else 0.0 for lam_x in eigx])
    tryy = sum([lam_y * np.log(lam_y) if lam_y > ZERO_THRESH else 0.0 for lam_y in eigy])
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
    eigx, vecx = np.linalg.eig(matrix)
    with open(os.path.join(eigen_path, word + '.pkl'), 'w+') as f:
        pickle.dump((eigx, vecx), f)

if __name__ == '__main__':
    pass

