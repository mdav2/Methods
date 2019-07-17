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
# This function is very inneficient, but is totally based on second quantization, so it should produce the right answer.
# Since the function overlap is called, there is no need to multiply by phases at the end.

def BF_Hone(bra1, bra2, molint):
    N = range(len(bra1.occ[0]))
    out = 0
    # Loop thorugh the orbitals and multiply the molecular integral by the phase factor arising from the anh and cre operators
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
                    # Loop thorugh the orbitals and multiply the molecular integral by the phase factor arising from the anh and cre operators
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

    # Use slater rules case 1: Equal determinants. 
    # Loop thourgh orbitals computing (p|h|p) for alpha p and beta p.
    # Insert the phase factor that comes with each determinant

    if int(dif) == 0:
        for orb in np.where(bra1.occ[0] == 1)[0]:
            out += molint[orb, orb]
        for orb in np.where(bra1.occ[1] == 1)[0]:
            out += molint[orb, orb]
        return out*bra1.p * bra2.p

    # Second case of slater rules. Determinants differ by one pair of MOs
    # Test if the different MOs have the same spin. If not, return zero. That is if nalpha = 1
    # Use np.where to locate which are the occupied orbitals. It returns a list of such orbitals. In this case, [p, q]
    # Compute (p|h|q). Note that, if the determinants are not in maximum coincidence a sign problem may happen

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

    # Third case. Determinants differ in more than two MO. Return zero.

    else:
        return 0

# This version uses slater rules and second quantization to fix sign problems, but being more effective

def Hone(bra1, bra2, molint):
    out = 0
    dif = bra1 - bra2

    # First case. Just apply Slater Rules, this cannot go wrong.

    if int(dif) == 0:
        for orb in np.where(bra1.occ[0] == 1)[0]:
            out += molint[orb, orb]
        for orb in np.where(bra1.occ[1] == 1)[0]:
            out += molint[orb, orb]
        return out*bra1.p * bra2.p

    # Second case. We know from Slater rules that only the integral (p|h|q) needs to be elvaluted (p and q are the differente MOs)
    # We do so, but using second quantization to keep track of the sign. It is simular to the brute force, but in this case we just compute phases factors
    # For the specific p and q. Note that phase is compute for (p|h|q) and (q|h|p) (p1 and p2) as we sum over all pair within second quant.

    if int(dif) == 2:
        nalpha = dif.occ[0].sum()
        if nalpha == 0:
            orbs = np.where(dif.occ[1] == 1)[0]
            p1 = overlap(bra1.an(orbs[0],0), bra2.an(orbs[1],0)) + overlap(bra1.an(orbs[0],1), bra2.an(orbs[1],1))
            p2 = overlap(bra1.an(orbs[1],0), bra2.an(orbs[0],0)) + overlap(bra1.an(orbs[1],1), bra2.an(orbs[0],1))
            return molint[orbs[0], orbs[1]] * (p1 + p2)
        if nalpha == 2:
            orbs = np.where(dif.occ[0] == 1)[0]
            p1 = overlap(bra1.an(orbs[0],0), bra2.an(orbs[1],0)) + overlap(bra1.an(orbs[0],1), bra2.an(orbs[1],1))
            p2 = overlap(bra1.an(orbs[1],0), bra2.an(orbs[0],0)) + overlap(bra1.an(orbs[1],1), bra2.an(orbs[0],1))
            return molint[orbs[0], orbs[1]]* (p1 + p2)
        else:
            return 0

    # Third case. Again, just Slater rules, which says this is zero. 

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

