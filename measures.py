import numpy as np

ZERO_THRESH = 1e-12 # Convention is to take x nonzero if x >= ZERO_THRESH

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
    AtB = np.dot(A.T, B)
    projAB = np.linalg.norm(AtB, axis = 1)
    numerator = sum([projAB[i] * eiga[i] for i in range(len(eiga))])
    denominator = sum(eiga)
    return numerator / denominator

def compute_clarke_de(eiga, A, norma, eigb, B, normb):
    AtB = np.dot(A.T, B)
    eiga = norma * eiga
    eigb = normb * eigb
    tmpb = np.einsum('ij,i,ji->j', AtB, eiga, AtB.T)
    numerator = sum([min(pair) for pair in zip(tmpb, eigb)])
    denominator = sum(eiga)
    return min(numerator / denominator, 1.0)

def compute_fidelity(eiga, A, eigb, B):
    AtB = np.dot(A.T, B)
    eigsqrt = np.sqrt(np.outer(eiga, eigb))
    return np.einsum('ij,ij->', AtB, eigsqrt)

if __name__ == '__main__':
    pass

