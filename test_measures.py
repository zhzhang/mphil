import measures
import numpy as np
import struct
from dmatrices import _compute_eigenvectors, ZERO_THRESH
from scipy.linalg import logm

def run_tests():
    matrix1, norm1 = load_matrix("test_matrices/villa.bin")
    matrix2, norm2 = load_matrix("test_matrices/home.bin")
    eig1, vec1 = _compute_eigenvectors(matrix1)
    eig2, vec2 = _compute_eigenvectors(matrix2)
    full_eig1, full_vec1 = np.linalg.eigh(matrix1)
    full_eig2, full_vec2 = np.linalg.eigh(matrix2)

    relent1, relent2 = measures.compute_rel_ent(eig1, vec1, eig2, vec2)
    true_relent1, true_relent2 = compute_rel_ent(matrix1, matrix2)
    compare(relent1, true_relent1, "RELENT")

    weeds1 = measures.compute_weeds_prec(eig1, vec1, eig2, vec2)
    weeds2 = measures.compute_weeds_prec(eig2, vec2, eig1, vec1)
    true_weeds1 = np.trace(np.dot(vec2.T, np.dot(matrix1, vec2)))
    true_weeds2 = np.trace(np.dot(vec1.T, np.dot(matrix2, vec1)))
    compare(weeds1, true_weeds1, "WEEDS")
    compare(weeds2, true_weeds2, "WEEDS")

    clarke1 = measures.compute_clarke_de(eig1, vec1, norm1, eig2, vec2, norm2)
    clarke2 = measures.compute_clarke_de(eig2, vec2, norm2, eig1, vec1, norm1)
    true_clarke1 = compute_clarke_de(eig1, vec1, norm1, matrix2, norm2)
    true_clarke2 = compute_clarke_de(eig2, vec2, norm2, matrix1, norm1)
    compare(clarke1, true_clarke1, "CLARKE")
    compare(clarke2, true_clarke2, "CLARKE")

    print "Tests completed successfully."

def compare(x, y, test):
    if np.absolute(x - y) >= ZERO_THRESH:
        raise RuntimeError("Test failed %s" % test)

def compute_rel_ent(X,Y):
    a = np.trace(np.dot(X,logm(X))) - np.trace(np.dot(X,logm(Y)))
    b = np.trace(np.dot(Y,logm(Y))) - np.trace(np.dot(Y,logm(X)))
    return np.absolute(a), np.absolute(b)

def compute_clarke_de(eig, vec, norm, matrix2, norm2):
    total = 0.0
    eig = norm * eig
    matrix2 = norm2 * matrix2
    for i, lam in enumerate(eig):
        v = vec[:,i]
        total += min(lam, np.dot(v.T, np.dot(matrix2, v)))
    return total / sum(eig)

def load_matrix(matrix_path):
    matrix = np.zeros([200,200])
    matrix_file = open(matrix_path, 'rb')
    while True:
        data = matrix_file.read(12)
        if len(data) < 12:
            break
        x, y, value = struct.unpack('>iif', data)
        matrix[x,y] = matrix[y,x] = value
    matrix_file.close()
    norm = np.trace(matrix)
    return matrix / norm, norm

if __name__ == "__main__":
    run_tests()
