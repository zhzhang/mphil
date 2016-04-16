import dmatrices
from math import log
import numpy as np
from scipy.linalg import logm, sqrtm
import os
import shutil

# Define formulaic computation of the associated measures.
def compute_rel_ent(X,Y):
    a = np.trace(np.dot(X,logm(X))) - np.trace(np.dot(X,logm(Y)))
    b = np.trace(np.dot(Y,logm(Y))) - np.trace(np.dot(Y,logm(X)))
    return a,b

def compute_fidelity(X,Y):
    sX = sqrtm(X)
    return np.trace(sqrtm(sX * Y * sX))

# Test 1: matrices composed of linearly independent pure states.
A = np.zeros([3,3])
A[0,0] = 0.2
A[1,1] = 0.3
A[2,2] = 0.5
I = np.eye(3) / 3
tmp = 1.0 / 3
# compute entropies by hand
true_S_IA = log(tmp) - (log(0.2) + log(0.3) + log(0.5)) / 3
true_S_AI = 0.2 * (log(0.2) - log(tmp)) + 0.3 * (log(0.3) - log(tmp)) + 0.5 * (log(0.5) - log(tmp))
# via formulas
(S_IA, S_AI) = compute_rel_ent(I,A)
err_IA = np.absolute(S_IA - true_S_IA)
err_AI = np.absolute(S_AI - true_S_AI)
if err_IA > dmatrices.ZERO_THRESH:
    print "Failed Test 1, error I->A was %f" % err_IA
if err_AI > dmatrices.ZERO_THRESH:
    print "Failed Test 1, error A->I was %f" % err_AI

# Test 2: matrices with same span, different eigenvalues
A = np.zeros([2,2])
A[0,0] = 0.4
A[1,1] = 0.6
B = np.zeros([2,2])
B[0,0] = 0.7
B[1,1] = 0.3
true_S_AB = 0.4 * (log(0.4) - log(0.7)) + 0.6 * (log(0.6) - log(0.3))
true_S_BA = 0.7 * (log(0.7) - log(0.4)) + 0.3 * (log(0.3) - log(0.6))
(S_AB, S_BA) = compute_rel_ent(A,B)
err_AB = np.absolute(S_AB - true_S_AB)
err_BA = np.absolute(S_BA - true_S_BA)
if err_BA > dmatrices.ZERO_THRESH:
    print "Failed Test 2, error B->A was %f" % err_BA
if err_AB > dmatrices.ZERO_THRESH:
    print "Failed Test 2, error A->B was %f" % err_AB

# Test 3: example from E. Balkir's work.
a = np.array([6,5,0], dtype=float)
b = np.array([7,3,0], dtype=float)

c = np.array([1,0,0], dtype=float)
d = np.array([1,1,0], dtype=float)

"""
a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)
c = c / np.linalg.norm(c)
d = d / np.linalg.norm(d)
"""

lager = np.outer(a,a)
ale = np.outer(b,b)
beer = 6 * np.outer(c,c) + 7 * np.outer(d,d)

lager = lager / np.trace(lager)
ale = ale / np.trace(ale)
beer = beer / np.trace(beer)

eig_beer, vec_beer = np.linalg.eig(beer)
eig_lager, vec_lager = np.linalg.eig(lager)
print dmatrices.compute_rel_ent(eig_beer, vec_beer, eig_lager, vec_lager)
_, tmp = compute_rel_ent(beer, lager)
print tmp
print 1 / (1 + tmp)
print '---'

psy = np.zeros([3,3])
psy[0,0] = 2
psy[1,1] = 5
doc = np.zeros([3,3])
doc[0,0] = 5
doc[1,1] = 2
doc[2,2] = 3
psy = psy / np.trace(psy)
doc = doc / np.trace(doc)
a = psy[0,0]
b = psy[1,1]
c = doc[0,0]
d = doc[1,1]
print a * (log(a) - log(c)) + b * (log(b) - log(d))


eig_doc, vec_doc = np.linalg.eig(doc)
eig_psy, vec_psy= np.linalg.eig(psy)
print dmatrices.compute_rel_ent(eig_doc, vec_doc, eig_psy, vec_psy)

tmp, _ = compute_rel_ent(psy, doc)
print 1 / (1 + tmp)

print compute_fidelity(doc, psy)


