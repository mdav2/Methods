import psi4
import numpy as np
import scipy.linalg as la
import os
import sys
import timeit
from itertools import permutations

file_dir = os.path.dirname('../Aux/')
sys.path.append(file_dir)

file_dir = os.path.dirname('../HF/RHF/')
sys.path.append(file_dir)

from tools import *
from fock import *
from rhf import RHF
from Hamiltonian import *

# Generate H matrix, but for two diff lists of dets

def det_int(det1, det2, molint1, molint2, v = False, t = False):
        l1 = len(det1)
        l2 = len(det2)
        H = np.zeros((l1,l2))
        t0 = timeit.default_timer()
        prog = 0
        for i,d1 in enumerate(det1):
            for j,d2 in enumerate(det2):
                if j > i:
                    break
                H[i,j] = Htot(d1, d2, molint1, molint2)
                H[j,i] = H[i,j]
            prog += 1
            if v:
                print("Progress: {:2.0f}%".format((prog/l)*100))
        tf = timeit.default_timer()
        if t:
            print("Completed. Time needed: {}".format(tf - t0))
        return H

def CASINT(cas_list):
    H = np.zeros((len(cas_list),len(cas_list)))
    S = np.zeros((len(cas_list),len(cas_list)))
    for i,psiA in enumerate(cas_list):
        for j,psiB in enumerate(cas_list):
            if j > i:
                break
            if psiA == psiB:
                H[i,j] = psiA.E
                S[i,j] = 1.0
            else:
                Hint = det_int(psiA.determinants, psiB.determinants, psiA.MIone, psiA.MItwo)
                hold = np.einsum('i,j,ij->',psiA.C0,psiB.C0,Hint)
                H[i,j] = hold
                H[j,i] = hold

                commonA = []
                for i1,x in enumerate(psiA.determinants):
                    if x in psiB.determinants:
                        commonA.append(i1)
                commonB = []
                for i2,x in enumerate(psiB.determinants):
                    if x in psiA.determinants:
                        commonB.append(i2)
                hold = 0
                for di,dj in zip(commonA, commonB):
                    hold += psiA.C0[di]*psiB.C0[dj]
                S[i,j] = hold
                S[j,i] = hold
    
    print(H)
    print(S)
    X = np.matrix(la.inv(la.sqrtm(S)))
    H = np.matrix(H)
    Ht = X * H * X
    Ecasint, Ct = la.eigh(Ht)
    print(Ecasint)
    print(X * Ct)
    return Ecasint[0] + psiA.V_nuc
        

                
    
    
