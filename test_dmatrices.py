import dmatrices
import numpy as np

def run_tests():
    # Test matrix loading.
    path = "test_matrices/home.bin"
    matrix, norm, basis = dmatrices._load_matrix_sparse(path, None, None)
    matrix_half, norm_half, basis_half = dmatrices._load_matrix_sparse(path, 100, None)
    matrix_prob, norm_prob, basis_prob = dmatrices._load_matrix_sparse(path, 100, "prob")
    matrix_dynam, norm_dynam, basis_dynam = dmatrices._load_matrix_sparse(path, None, "prob")
    evaluate(norm * matrix[:100,:100], norm_half * matrix_half)
    true_prob_matrix = get_prob_matrix(matrix, norm, 100)
    evaluate(true_prob_matrix, norm_prob * matrix_prob)
    true_dynam_matrix = get_dynam_matrix(matrix, norm)
    evaluate(true_dynam_matrix, norm_dynam * matrix_dynam)

    # Test basis merging.
    path2 = "test_matrices/villa.bin"
    matrix2, norm2, basis2 = dmatrices._load_matrix_sparse(path2, None, None)
    eig1, vec1 = dmatrices._compute_eigenvectors(matrix)
    eig2, vec2 = dmatrices._compute_eigenvectors(matrix2)
    new_vec1, new_vec2 = dmatrices._merge_basis(basis, basis2, vec1, vec2)
    for i in xrange(len(eig1)):
        if not np.absolute(sum(vec1[:,3]) - sum(new_vec1[:,3])) < dmatrices.ZERO_THRESH:
            raise RuntimeError("Test failed.")
    for i in xrange(len(eig2)):
        if not np.absolute(sum(vec2[:,3]) - sum(new_vec2[:,3])) < dmatrices.ZERO_THRESH:
            raise RuntimeError("Test failed.")

    # Test skew loading.
    skew1, skew2, skew_basis = dmatrices._load_skew_sparse(path, path2, 0.6, None, None)
    true_skew1, true_skew2 = get_skew_matrix(matrix, basis, matrix2, basis2, skew_basis)
    evaluate(skew1, true_skew1)
    evaluate(skew2, true_skew2)

    matrix, norm, basis = dmatrices._load_matrix_sparse(path, 100, None)
    matrix2, norm2, basis2 = dmatrices._load_matrix_sparse(path2, 100, None)
    skew1, skew2, skew_basis = dmatrices._load_skew_sparse(path, path2, 0.6, 100, None)
    true_skew1, true_skew2 = get_skew_matrix(matrix, basis, matrix2, basis2, skew_basis)
    evaluate(skew1, true_skew1)
    evaluate(skew2, true_skew2)

    matrix, norm, basis = dmatrices._load_matrix_sparse(path, 100, "prob")
    matrix2, norm2, basis2 = dmatrices._load_matrix_sparse(path2, 100, "prob")
    skew1, skew2, skew_basis = dmatrices._load_skew_sparse(path, path2, 0.6, 100, "prob")
    true_skew1, true_skew2 = get_skew_matrix(matrix, basis, matrix2, basis2, skew_basis)
    evaluate(skew1, true_skew1)
    evaluate(skew2, true_skew2)

    matrix, norm, basis = dmatrices._load_matrix_sparse(path, None, "prob")
    matrix2, norm2, basis2 = dmatrices._load_matrix_sparse(path2, None, "prob")
    skew1, skew2, skew_basis = dmatrices._load_skew_sparse(path, path2, 0.6, None, "prob")
    true_skew1, true_skew2 = get_skew_matrix(matrix, basis, matrix2, basis2, skew_basis)
    evaluate(skew1, true_skew1)
    evaluate(skew2, true_skew2)

    print "Tests completed successfully."

def evaluate(A,B):
    if not (np.absolute(A - B) < dmatrices.ZERO_THRESH).all():
        raise RuntimeError("Failed test")

def get_skew_matrix(matrix1, basis1, matrix2, basis2, skew_basis):
    tmp1 = np.zeros([len(skew_basis), len(skew_basis)])
    tmp2 = np.zeros([len(skew_basis), len(skew_basis)])
    for i in skew_basis.keys():
        for j in skew_basis.keys():
            new_x = skew_basis[i]
            new_y = skew_basis[j]
            try:
                x = basis1[i]
                y = basis1[j]
                tmp1[new_x, new_y] = tmp1[new_y, new_x] = matrix1[x,y]
            except KeyError:
                continue
    for i in skew_basis.keys():
        for j in skew_basis.keys():
            new_x = skew_basis[i]
            new_y = skew_basis[j]
            try:
                x = basis2[i]
                y = basis2[j]
                tmp2[new_x, new_y] = tmp2[new_y, new_x] = matrix2[x,y]
            except KeyError:
                continue
    true_skew1 = 0.4 * tmp1 + 0.6 * tmp2 
    true_skew2 = 0.6 * tmp1 + 0.4 * tmp2
    return true_skew1, true_skew2

def get_prob_matrix(matrix, norm, dim):
    matrix = norm * matrix
    diag = []
    for i in xrange(matrix.shape[0]):
        diag.append(matrix[i,i])
    new_basis = dict([(b, i) for i, (_, b) in enumerate(sorted([(val, index)\
        for index, val in enumerate(diag)], reverse=True)[:dim])])
    output = np.zeros([dim, dim])
    for i in xrange(matrix.shape[0]):
        for j in xrange(matrix.shape[0]):
            try:
                x = new_basis[i]
                y = new_basis[j]
            except KeyError:
                continue
            output[x,y] = output[y,x] = matrix[i,j]
    return output

def get_dynam_matrix(matrix, norm):
    matrix = norm * matrix
    diag = []
    for i in xrange(matrix.shape[0]):
        diag.append((matrix[i,i], i))
    diag = sorted(diag, reverse=True)
    cummulative = 0.0
    index = 0
    new_basis = {}
    while cummulative / norm < 0.85:
        cummulative += diag[index][0]
        new_basis[diag[index][1]] = index
        index += 1
    output = np.zeros([index, index])
    for i in xrange(matrix.shape[0]):
        for j in xrange(matrix.shape[0]):
            try:
                x = new_basis[i]
                y = new_basis[j]
            except KeyError:
                continue
            output[x,y] = output[y,x] = matrix[i,j]
    return output

if __name__ == "__main__":
    run_tests()
