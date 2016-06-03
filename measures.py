import numpy as np

ZERO_THRESH = 1e-12 # Convention is to take x nonzero if x >= ZERO_THRESH

def compute_rel_ent(eigx, vecx, eigy, vecy): # Avoids recomputing the projection matrix.
    trxx = sum([lam_x * np.log(lam_x) if lam_x >= ZERO_THRESH else 0.0 for lam_x in eigx])
    tryy = sum([lam_y * np.log(lam_y) if lam_y >= ZERO_THRESH else 0.0 for lam_y in eigy])
    projection_matrix = np.dot(vecx.T, vecy)
    trxy = _compute_cross_entropy(projection_matrix, eigx, eigy)
    tryx = _compute_cross_entropy(projection_matrix.T, eigy, eigx)
    return (trxx - trxy, tryy - tryx)

def compute_single_rel_ent(eigx, vecx, eigy, vecy):
    trxx = sum([lam_x * np.log(lam_x) if lam_x >= ZERO_THRESH else 0.0 for lam_x in eigx])
    projection_matrix = np.dot(vecx.T, vecy)
    trxy = _compute_cross_entropy(projection_matrix, eigx, eigy)
    return trxx - trxy

def _compute_cross_entropy(AtB, eiga, eigb):
    # Check containment of one eigenspace inside the other.
    projAB = np.linalg.norm(AtB, axis = 1)
    spannedAB = np.all(np.absolute(projAB - np.ones(projAB.shape)) < ZERO_THRESH)
    # Compute cross entropy A -> B
    if spannedAB:
        output = 0.0
        tmp_ab = np.einsum('ij,i,ji->j', AtB, eiga, AtB.T)
        for i,lam_b in enumerate(eigb):
            if np.absolute(tmp_ab[i]) < ZERO_THRESH:
                continue
            else:
                output += tmp_ab[i] * np.log(lam_b)
    else:
        output = -float('inf')
    return output

def compute_weeds_prec(eiga, A, eigb, B):
    AtB = np.dot(A.T, B)
    projAB = np.linalg.norm(AtB, axis = 1)
    numerator = sum([projAB[i] * eiga[i] for i in range(len(eiga))])
    numerator = np.einsum('ij,i,ji->', AtB, eiga, AtB.T)
    denominator = sum(eiga)
    return numerator / denominator

def compute_clarke_de(eiga, A, norma, eigb, B, normb):
    AtB = np.dot(A.T, B)
    eiga = norma * eiga
    eigb = normb * eigb
    tmpa = np.einsum('ij,i,ji->j', AtB.T, eigb, AtB)
    numerator = sum([min(pair) for pair in zip(tmpa, eiga)])
    denominator = sum(eiga)
    return min(numerator / denominator, 1.0)

def compute_fidelity(eiga, A, eigb, B):
    AtB = np.dot(A.T, B)
    eigsqrt = np.sqrt(np.outer(eiga, eigb))
    return np.einsum('ij,ij->', AtB, eigsqrt)

if __name__ == '__main__':
    pass

