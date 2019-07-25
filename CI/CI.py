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
            
# Function: Compute Htwo and Hone at same time

def Htot(bra1, bra2, molint1, molint2):
    dif = bra1 - bra2

    if dif == 0:
        phase = bra1.p * bra2.p
        alphas = bra1.occ[0]
        betas = bra1.occ[1]
        one = np.einsum('mm,m->', molint1, alphas) + np.einsum('mm,m->', molint1, betas)
    
        # Compute J for all combinations of m n being alpha or beta
        x1 = np.einsum('mmnn, m, n', molint2, alphas, alphas)
        x2 = np.einsum('mmnn, m, n', molint2, betas, betas)
        x3 = np.einsum('mmnn, m, n', molint2, alphas, betas)
        x4 = np.einsum('mmnn, m, n', molint2, betas, alphas)
        J = x1 + x2 + x3 + x4
        # For K m and n have to have the same spin, thus only two cases are considered
        x1 = np.einsum('mnnm, m, n', molint2, alphas, alphas)
        x2 = np.einsum('mnnm, m, n', molint2, betas, betas)
        K = x1 + x2
        return phase * (0.5 * (J - K) + one)

    elif dif == 2:
        # Use notin to return a list of [orbital, spin] that are present in the first bra, but not in the second
        [o1, s1] = bra1.notin(bra2)[0]
        [o2, s2] = bra2.notin(bra1)[0]
        if s1 != s2:  # Check if the different orbitals have same spin
            return 0
        phase = bra1.an(o1, s1).cr(o2, s2).p * bra2.p # Annihilating o1 and creating o2 generates the same phase as moving o1 to the o2 position
        # For J, (mp|nn), n can have any spin. Two cases are considered then. Obs: bra1.occ or bra2.occ would yield the same result. When n = m or p J - K = 0
        J = np.einsum('nn, n->', molint2[o1,o2], bra1.occ[0]) + np.einsum('nn, n->', molint2[o1,o2], bra1.occ[1]) 
        K = np.einsum('nn, n->', molint2.swapaxes(1,3)[o1,o2], bra1.occ[s1])
        return phase * (molint1[o1,o2] + J - K)

    elif dif == 4:
        [[o1, s1], [o2, s2]] = bra1.notin(bra2)
        [[o3, s3], [o4, s4]] = bra2.notin(bra1)
        phase = bra1.an(o1, s1).cr(o3, s3).an(o2, s2).cr(o4, s4).p * bra2.p
        if s1 == s3 and s2 == s4:
            J = molint2[o1, o3, o2, o4] 
        else:
            J = 0
        if s1 == s4 and s2 == s3:
            K = molint2[o1, o4, o2, o3]
        else:
            K = 0
        return phase * (J - K)
    else:
        return 0

# FUNCTION: Given a list of determinants, compute the Hamiltonian matrix

def get_H(dets, molint1, molint2, v = False, t = False):
        l = len(dets)
        H = np.zeros((l,l))
        t0 = timeit.default_timer()
        prog = 0
        for i,d1 in enumerate(dets):
            for j,d2 in enumerate(dets):
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
        print("Number of electrons: {}".format(self.nelec))
        print("Number of basis functions: {}".format(self.nbf))
        print("Number of doubly occupied orbitals: {}".format(self.ndocc))
        print("Number of virtual spatial orbitals: {}".format(self.virtual))

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
                for a in range(self.ndocc):
                    for b in range(self.ndocc):
                        for s1 in [0, 1]:
                            for s2 in [0, 1]:
                                new = self.ref.an(a, s1).cr(i, s1).an(b, s2).cr(j,s2)
                                if new.p != 0 and new not in determinants:
                                    determinants.append(new)
            prog += 1
            print("Progress: {:2.0f}%".format(100*prog/prog_total))
        print("Number of Determinants: {}".format(len(determinants)))

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

    # CAS assuming nbeta = nalpha

    def compute_CAS(self, active_space ='',nfrozen=0, nvirtual=0):

        if len(active_space) != self.nbf:
            raise NameError("Invalid active space. Please check the number of basis functions")
        space = []
        n_ac_orb = 0
        n_ac_elec_pair = int(self.nelec/2)
        print("Reading Active space")
        for i in active_space:
            if i == 'o':
                space.append(int(1))
                n_ac_elec_pair -= 1
            elif i == 'a':
                space.append('a')
                n_ac_orb += 1
            elif i == 'u':
                space.append(int(0))
            else:
                raise NameError("Invalid active space entry: {}".format(i))
        active = np.array([1]*n_ac_elec_pair + [0]*(n_ac_orb - n_ac_elec_pair))
        print("Number of active orbitals: {}".format(n_ac_orb))
        print("Number of active electrons: {}".format(2*n_ac_elec_pair))

        print("Generating excitations")
        perms = set(permutations(active))
        print("Generating determinants")
        determinants = []
        for p1 in perms:
            for p2 in perms:
                alpha = space.copy()
                beta =  space.copy()
                for i,x in enumerate(np.where(np.array(space) == 'a')[0]):
                    alpha[x] = p1[i]
                    beta[x] = p2[i]
                determinants.append(Bra([alpha, beta]))
        print(determinants[0] - determinants[1])
        print("Number of determinants: {}".format(len(determinants)))

        # This part only works if # alpha elec = # beta elec
        if False:
            oc = np.array([1]*self.ndocc + [0]*self.virtual)
            self.ref = Bra([oc, oc])
            print("Generating determinants")
            t0 = timeit.default_timer()
            frozen = self.ref.occ[0][0:nfrozen]
            print("Number of frozen orbitals: {}".format(2*len(frozen)))

            virtual = self.ref.occ[0][self.nbf-nvirtual:]
            print("Number of virtual orbitals: {}".format(2*len(virtual)))

            active = self.ref.occ[0][nfrozen:self.nbf-nvirtual]
            print("Number of active orbitals: {}".format(2*len(active)))
            dets = []
            print("Generation permutations")
            for i in set(permutations(active)):
                dets.append(np.hstack((frozen, i, virtual)))
            print("Storing determinants") 
            determinants = []
            for a in dets:
                for b in dets:
                    determinants.append(Bra([a,b]))
            tf = timeit.default_timer()
            print("Number of Determinants: {}".format(len(determinants)))
            print("Completed. Time needed: {}".format(tf - t0))

        # COMPUTE HAMILTONIAN MATRIX

        print("Generating Hamiltonian Matrix")
        H = get_H(determinants, self.MIone, self.MItwo, v = True, t = True)

        # DIAGONALIZE HAMILTONIAN MATRIX

        print("Diagonalizing Hamiltonian Matrix")
        t0 = timeit.default_timer()
        E, C = la.eigh(H)
        tf = timeit.default_timer()
        print("Completed. Time needed: {}".format(tf - t0))
        print("CAS Energy:")
        self.Ecas = E[0] + self.V_nuc
        print(self.Ecas)
        psi4.core.print_out("CAS Energy: {:<15.10f} ".format(self.Ecas) + emoji('whale'))
        return self.Ecas
