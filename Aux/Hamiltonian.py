from tools import *
from fock import *
from rhf import RHF
import numpy as np
import timeit

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
