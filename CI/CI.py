import psi4
import numpy as np
import scipy.linalg as la
import os
import sys
import timeit

file_dir = os.path.dirname('../Aux/')
sys.path.append(file_dir)

file_dir = os.path.dirname('../HF/RHF/')
sys.path.append(file_dir)

from tools import *
from fock import *
from rhf import RHF

### USE A SECOND QUANTIZATION APPROACH TO GET ELEMENTS OF THE HAMILTONIAN MATRIX ###

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

### FUNCTION: USE SLATER RULES TO GET ELEMENTS OF THE ONE-ELECTRON HAMILTONIAN MATRIX ###

def Hone(bra1, bra2, molint):
    dif = bra1 - bra2

    # Use slater rules case 1: Equal determinants. 
    # Sum over the molecular integrals with equal index, but multiply by the occupancy. This way we do not sum unoccipied orbitals.
    if dif == 0:
        out = 0
        A = np.einsum('mm,m->', molint, bra1.occ[0])
        B = np.einsum('mm,m->', molint, bra1.occ[1])
        return (A + B) * bra1.p * bra2.p

    # Second case of slater rules. Determinants differ by one pair of MOs
    # Test if the different MOs have the same spin. If not, return zero. That is if nalpha = 1
    # occ_dif should have a -1 and 1. Located the position using np.where. To use Slater rules we would have to
    # Move the orbital that returned 1 to the position where it returned -1. Thus we need to count how many occupied orbitals are there
    # In between these two positions to compute the phase.

    elif dif == 2:
        # Use notin to return a list of [orbital, spin] that are present in the first bra, but not in the second
        [o1, s1] = bra1.notin(bra2)[0]
        [o2, s2] = bra2.notin(bra1)[0]
        if s1 != s2:  # Check if the different orbitals have same spin
            return 0
        phase = bra1.an(o1, s1).cr(o2, s2).p * bra2.p # Annihilating o1 and creating o2 generates the same phase as moving o1 to the o2 position
        return molint[o1, o2] * phase

    # Third case. Determinants differ in more than two MO. Return zero.

    else:
        return 0

### FUNCTION: USE SLATER RULES TO GET ELEMENTS OF THE TWO-ELECTRON HAMILTONIAN MATRIX ###
# The multiple einsums are used to account for the occupancy with all combinations of alpha and beta M and N orbitals

def Htwo(bra1, bra2, molint):
    dif = bra1 - bra2
    if dif == 0:
        alphas = bra1.occ[0]
        betas = bra1.occ[1]
        # Compute J for all combinations of m n being alpha or beta
        x1 = np.einsum('mmnn, m, n', molint, alphas, alphas)
        x2 = np.einsum('mmnn, m, n', molint, betas, betas)
        x3 = np.einsum('mmnn, m, n', molint, alphas, betas)
        x4 = np.einsum('mmnn, m, n', molint, betas, alphas)
        J = x1 + x2 + x3 + x4
        # For K m and n have to have the same spin, thus only two cases are considered
        x1 = np.einsum('mnnm, m, n', molint, alphas, alphas)
        x2 = np.einsum('mnnm, m, n', molint, betas, betas)
        K = x1 + x2
        return 0.5 * bra1.p * bra2.p * (J - K)

    elif dif == 2:
        # Use notin to return a list of [orbital, spin] that are present in the first bra, but not in the second
        [o1, s1] = bra1.notin(bra2)[0]
        [o2, s2] = bra2.notin(bra1)[0]
        if s1 != s2:  # Check if the different orbitals have same spin
            return 0
        phase = bra1.an(o1, s1).cr(o2, s2).p * bra2.p # Annihilating o1 and creating o2 generates the same phase as moving o1 to the o2 position
        # For J, (mp|nn), n can have any spin. Two cases are considered then. Obs: bra1.occ or bra2.occ would yield the same result. When n = m or p J - K = 0
        J = np.einsum('nn, n->', molint[o1,o2], bra1.occ[0]) + np.einsum('nn, n->', molint[o1,o2], bra1.occ[1]) 
        K = np.einsum('nn, n->', molint.swapaxes(1,3)[o1,o2], bra1.occ[s1])
        return phase * (J - K)

    elif dif == 4:
        [[o1, s1], [o2, s2]] = bra1.notin(bra2)
        [[o3, s3], [o4, s4]] = bra2.notin(bra1)
        phase = bra1.an(o1, s1).cr(o3, s3).an(o2, s2).cr(o4, s4).p * bra2.p
        if s1 == s3 and s2 == s4:
            J = molint[o1, o3, o2, o4] 
        else:
            J = 0
        if s1 == s4 and s2 == s3:
            K = molint[o1, o4, o2, o3]
        else:
            K = 0
        return phase * (J - K)
    else:
        return 0
            
def get_H(dets, molint1, molint2, v = False, t = False):
        H = []
        t0 = timeit.default_timer()
        prog_total = len(dets)
        prog = 0
        for d1 in dets:
            hold = []
            for d2 in dets:
                hold.append(Hone(d1, d2, molint1) + Htwo(d1, d2, molint2))
            H.append(hold)
            prog += 1
            if v:
                print("Progress: {:2.0f}%".format((prog/prog_total)*100))
        tf = timeit.default_timer()
        if t:
            print("Completed. Time needed: {}".format(tf - t0))
        return H

class CI:
    
# Pull in Hartree-Fock data, including integrals

    def __init__(self, HF):
        self.orbitals = HF.orbitals
        self.ndocc = HF.ndocc
        self.nelec = HF.nelec
        self.nbf = HF.nbf
        self.virtual = self.nbf - self.ndocc
        self.V_nuc = HF.V_nuc
        self.h = HF.T + HF.V
        self.g = HF.g.swapaxes(1,2)
        self.S = HF.S

# Convert atomic integrals to MO integrals
        print("Converting atomic integrals to MO integrals...")
        self.MIone = np.einsum('up,vq,uv->pq', self.orbitals, self.orbitals, self.h)
        self.MItwo = np.einsum('up,vq,hr,zs,uvhz->pqrs', self.orbitals, self.orbitals, self.orbitals, self.orbitals,self.g)
        print("Completed!")

    def compute_CIS(self):
        print("Starting CIS computation")
        oc = np.array([1]*self.ndocc + [0]*self.virtual)
        self.ref = Bra([oc, oc])
        determinants = [self.ref]

        # GENERATE EXCITATIONS

        print("Generating singly excited states")
        prog_total = self.virtual*self.ndocc*2
        prog = 0
        for i in range(self.ndocc, self.nbf):
            for a in range(self.ndocc):
                for s in [0, 1]:
                    new = self.ref.an(a, s).cr(i, s)
                    determinants.append(new)
                    prog += 1
                    print("Progress: {:2.0f}%".format(100*prog/prog_total))

        # COMPUTE HAMILTONIAN MATRIX

        print("Generating Hamiltonian Matrix")
        H = get_H(determinants, self.MIone, self.MItwo, v = True, t = True)

        # DIAGONALIZE HAMILTONIAN MATRIX

        print("Diagonalizing Hamiltonian Matrix")
        t0 = timeit.default_timer()
        E, C = la.eigh(H)
        tf = timeit.default_timer()
        print("Completed. Time needed: {}".format(tf - t0))
        print("Energies:")
        print(E + self.V_nuc)

    def compute_CISD(self):
        print("Starting CIS computation")
        oc = np.array([1]*self.ndocc + [0]*self.virtual)
        self.ref = Bra([oc, oc])
        determinants = [self.ref]

        # GENERATE EXCITATIONS

        print("Generating singly excited states")
        prog_total = self.virtual*self.ndocc*2
        prog = 0
        for i in range(self.ndocc, self.nbf):
            for a in range(self.ndocc):
                for s in [0, 1]:
                    new = self.ref.an(a, s).cr(i, s)
                    determinants.append(new)
                    prog += 1
                    print("Progress: {:2.0f}%".format(100*prog/prog_total))

        print("Generating doubly excited states")
        prog_total = len(range(self.ndocc, self.nbf))
        prog = 0
        for i in range(self.ndocc, self.nbf):
            for j in range(self.ndocc, self.nbf):
                if i == j: break
                for a in range(self.ndocc):
                    for b in range(self.ndocc):
                        if a == b: break
                        for s1 in [0, 1]:
                            for s2 in [0, 1]:
                                new = self.ref.an(a, s1).cr(i, s1).an(b, s2).cr(j,s2)
                                if new.p != 0:
                                    determinants.append(new)
            prog += 1
            print("Progress: {:2.0f}%".format(100*prog/prog_total))

        # COMPUTE HAMILTONIAN MATRIX

        print("Generating Hamiltonian Matrix")
        H = get_H(determinants, self.MIone, self.MItwo, v = True, t = True)

        # DIAGONALIZE HAMILTONIAN MATRIX

        print("Diagonalizing Hamiltonian Matrix")
        t0 = timeit.default_timer()
        E, C = la.eigh(H)
        tf = timeit.default_timer()
        print("Completed. Time needed: {}".format(tf - t0))
        print("Energies:")
        print(E + self.V_nuc)
