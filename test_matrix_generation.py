import cPickle as pickle
import numpy as np
import os
import shutil
from dmatrices import DMatrices
from single import *

file1 = [[0,0,2,3],[0,1,1,3]]
file2 = [[0,1,1,3],[1,2,3,3]]

files = file1 + file2

truedm = {}
for line in files:
    counts = np.zeros(4)
    for i in range(4):
        counts[i] = float(line.count(i))
    for i in range(4):
        counti = counts[i]
        pure = np.copy(counts)
        if pure[i] > 0:
            pure[i] -= 1
        else:
            continue
        pure = pure / np.linalg.norm(pure)
        if i in truedm:
            truedm[i] += counti*np.outer(pure, pure)
        else:
            truedm[i] = counti*np.outer(pure, pure)
for i in range(4):
    truedm[i] = truedm[i] / np.linalg.norm(truedm[i])

try:
    os.mkdir('tmp_test_dir')
except OSError:
    pass
with open('tmp_test_dir/test_file1.pkl', 'w+') as f:
    pickle.dump(file1,f)
with open('tmp_test_dir/test_file2.pkl', 'w+') as f:
    pickle.dump(file2,f)

testdm = {}
process_file('tmp_test_dir', 'test_file1.pkl', testdm, [0,1,2,3], 4)
process_file('tmp_test_dir', 'test_file2.pkl', testdm, [0,1,2,3], 4)

with open('tmp_test_dir/matrices.pkl', 'w+') as f:
    pickle.dump(testdm, f)
with open('tmp_test_dir/dummy_wordmap.pkl', 'w+') as f:
    pickle.dump({}, f)

testdm = DMatrices('tmp_test_dir/matrices.pkl','tmp_test_dir/dummy_wordmap.pkl')
TOL = 1e-12
basis_map = {0:0, 1:1, 2:2, 3:3}
for i in range(4):
    diffmatrix = testdm._get_matrix(testdm.matrices[i], basis_map) - truedm[i]
    for (x,y), diff in np.ndenumerate(diffmatrix):
        if np.absolute(diff) > TOL:
            print "Failure for index %d" % i
print "Test matrix generation successful"
shutil.rmtree('tmp_test_dir')

