import psi4
import numpy as np
import scipy.linalg as la
import os
import sys

file_dir = os.path.dirname('../Aux/')
sys.path.append(file_dir)

file_dir = os.path.dirname('../HF/RHF/')
sys.path.append(file_dir)

from tools import *
from fock import *
from rhf import RHF

# Function to produce elements of the 1e Hamiltonian matrix a given pair of bras: BRUTE FORCE APPROACH

def BF_Hone(bra1, bra2, molint):
    N = range(len(bra1.occ[0]))
    out = 0
    for p in N:
        for q in N:
            h1 = overlap(bra1.an(p,0), bra2.an(q,0))
            h2 = overlap(bra1.an(p,1), bra2.an(q,1))
            out += (h1 + h2)*molint[p,q]
    return out

def BF_Htwo(bra1, bra2, molint):
    N = range(len(bra1.occ[0]))
    out = 0
    for p in N:
        for q in N:
            for r in N:
                for s in N:
                    h1 = 0.5*overlap(bra1.an(p,0).cr(q,0), bra2.an(s,0).cr(r,0))
                    h2 = 0.5*overlap(bra1.an(p,0).cr(q,0), bra2.an(s,1).cr(r,1))
                    h3 = 0.5*overlap(bra1.an(p,1).cr(q,1), bra2.an(s,0).cr(r,0))
                    h4 = 0.5*overlap(bra1.an(p,1).cr(q,1), bra2.an(s,1).cr(r,1))
                    if q == r:
                        h5 = 0.5*overlap(bra1.an(p,0), bra2.an(s,0))
                        h6 = 0.5*overlap(bra1.an(p,1), bra2.an(s,1))
                    else:
                        h5 = 0
                        h6 = 0
                    total = h1 + h2 + h3 + h4 - h5 - h6 
                    out += total*molint[p,q,r,s]
    return out

def SR_Hone(bra1, bra2, molint):
    out = 0
    dif = bra1 - bra2
    if int(dif) == 0:
        for orb in np.where(bra1.occ[0] == 1)[0]:
            out += molint[orb, orb]
        for orb in np.where(bra1.occ[1] == 1)[0]:
            out += molint[orb, orb]
        return out*bra1.p * bra2.p
    if int(dif) == 2:
        nalpha = dif.occ[0].sum()
        if nalpha == 0:
            orbs = np.where(dif.occ[1] == 1)[0]
            out += molint[orbs[0], orbs[1]]
            return out*bra1.p*bra2.p
        if nalpha == 2:
            orbs = np.where(dif.occ[0] == 1)[0]
            out += molint[orbs[0], orbs[1]]
            return out*bra1.p*bra2.p
        else:
            return 0
    else:
        return 0
            
class CI:
    
# Pull in Hartree-Fock data, including integrals

    def __init__(self, HF):
        self.orbitals = HF.orbitals
        self.ndocc = HF.ndocc
        self.nelec = HF.nelec
        self.nbf = HF.nbf
        self.V_nuc = HF.V_nuc
        self.h = HF.T + HF.V
        self.g = HF.g.swapaxes(1,2)
        self.S = HF.S

# Convert atomic integrals to MO integrals
        self.MIone = np.einsum('up,vq,uv->pq', self.orbitals, self.orbitals, self.h)
        self.MItwo = np.einsum('up,vq,hr,zs,uvhz->pqrs', self.orbitals, self.orbitals, self.orbitals, self.orbitals,self.g)

if __name__ == '__main__':
    alphas = np.array([1, 0, 1, 0, 0, 0])
    betas  = np.array([1, 1, 0, 0, 0, 0])
    occ = np.array([alphas, betas])
    ref = Bra(occ)
    ref2 = ref.an(1,1).cr(2,1)
    print(ref)
    print(BF_Hone(ref, ref))
    print(BF_Htwo(ref, ref))

