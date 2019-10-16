from tools import *
from fock import *
import numpy as np
import time
import sys

def H_dif0(det1, ha, hb, Vaaaa, Vbbbb, Vabab, Vbaba):
    alphas = det1.alpha_list()
    betas = det1.beta_list()
    one = np.einsum('mm,m->', ha, alphas) + np.einsum('mm,m->', hb, betas)
    
    # Compute J for all combinations of m n being alpha or beta
    x1 = np.einsum('mnmn, m, n', Vaaaa, alphas, alphas, optimize = 'optimal')
    x2 = np.einsum('mnmn, m, n', Vbbbb, betas,   betas, optimize = 'optimal')
    x3 = np.einsum('mnmn, m, n', Vabab, alphas,  betas, optimize = 'optimal')
    x4 = np.einsum('mnmn, m, n', Vbaba, betas,  alphas, optimize = 'optimal')
    return 0.5 * (x1 + x2 + x3 + x4) + one
    
    
def H_dif4(det1, det2, Vaaaa, Vbbbb, Vabab, Vabba, Vbaba, Vbaab):
    p = det1.phase(det2)
    [[o1, s1], [o2, s2]] = det1.Exclusive(det2)
    [[o3, s3], [o4, s4]] = det2.Exclusive(det1)
    if   (s1, s2, s3, s4) == (0,0,0,0):
        return p*Vaaaa[o1, o2, o3, o4]

    elif (s1, s2, s3, s4) == (1,1,1,1):
        return p*Vbbbb[o1, o2, o3, o4]

    elif (s1, s2, s3, s4) == (0,1,0,1):
        return p*Vabab[o1, o2, o3, o4]

    elif (s1, s2, s3, s4) == (0,1,1,0):
        return p*Vabba[o1, o2, o3, o4]

    elif (s1, s2, s3, s4) == (1,0,1,0):
        return p*Vbaba[o1, o2, o3, o4]

    elif (s1, s2, s3, s4) == (1,0,0,1):
        return p*Vbaab[o1, o2, o3, o4]

def H_dif2(det1, det2, ha, hb, Vaaaa, Vbbbb, Vabab, Vbaba):
    # Use Exclusive to return a list of [orbital, spin] that are present in the first det, but not in the second
    [o1, s1] = det1.Exclusive(det2)[0]
    [o2, s2] = det2.Exclusive(det1)[0]
    alphas = det1.alpha_list()
    betas = det1.beta_list()
    if s1 != s2:  # Check if the different orbitals have same spin  # DONT THINK THIS IS NECESSARY
        return 0
    p = det1.phase(det2)
    if s1 == 0:
        return p * (ha[o1,o2] + np.einsum('nn, n->', Vaaaa[o1,:,o2,:], alphas, optimize='optimal') \
                         + np.einsum('nn,n->', Vabab[o1,:,o2,:], betas, optimize='optimal'))
    else:
        return p * (hb[o1,o2] + np.einsum('nn, n->', Vbaba[o1,:,o2,:], alphas, optimize='optimal') \
                         + np.einsum('nn,n->', Vbbbb[o1,:,o2,:], betas, optimize='optimal'))

# FUNCTION: Given a list of determinants, compute the Hamiltonian matrix

def get_H(dets, ha, hb, Vaaaa, Vbbbb, Vabab, Vabba, Vbaba, Vbaab, v = False, t = False):
        l = len(dets)
        H = np.zeros((l,l))
        t0 = time.time()
        file = sys.stdout
        for i,d1 in enumerate(dets):
            H[i,i] = H_dif0(d1, ha=ha, hb=hb, Vaaaa=Vaaaa, Vbbbb=Vbbbb, Vabab=Vabab, Vbaba=Vbaba)
            for j,d2 in enumerate(dets):
                if j >= i:
                    break
                dif = d1 - d2
                if dif == 4:
                    H[i,j] = H_dif4(d1, d2,Vaaaa=Vaaaa, Vbbbb=Vbbbb, Vabab=Vabab, Vabba=Vabba, Vbaba=Vbaba, Vbaab=Vbaab)
                    H[j,i] = H[i,j]
                elif dif == 2:
                    H[i,j] = H_dif2(d1, d2, ha=ha, hb=hb, Vaaaa=Vaaaa, Vbbbb=Vbbbb, Vabab=Vabab, Vbaba=Vbaba)
                    H[j,i] = H[i,j]
            if v: showout(i+1, l, 50, "Generating Hamiltonian Matrix: ", file)
        file.write('\n')
        file.flush()
        if t:
            print("Completed. Time needed: {}".format(time.time() - t0))
        return H
