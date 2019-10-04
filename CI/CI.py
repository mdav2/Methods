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
from fock_old import *
from rhf import RHF
from Hamil_old import *

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
        # chemists
        self.g = HF.g.swapaxes(1,2)
        self.S = HF.S
        print("Number of electrons: {}".format(self.nelec))
        print("Number of basis functions: {}".format(self.nbf))
        print("Number of doubly occupied orbitals: {}".format(self.ndocc))
        print("Number of virtual spatial orbitals: {}".format(self.virtual))

# Convert atomic integrals to MO integrals
        psi4_orb = psi4.core.Matrix.from_array(self.orbitals)
        print("Converting atomic integrals to MO integrals...")
        self.MIone = np.einsum('up,vq,uv->pq', self.orbitals, self.orbitals, self.h)
        # Einsum to get molecular intergral -> slow!!!!
        #self.MItwo = np.einsum('up,vq,hr,zs,uvhz->pqrs', self.orbitals, self.orbitals, self.orbitals, self.orbitals,self.g)
        self.MItwo = np.asarray(HF.mints.mo_eri(psi4_orb, psi4_orb, psi4_orb, psi4_orb))
        print("Completed!")

    def compute_CIS(self):
        print("Starting CIS computation")
        oc = np.array([1]*self.ndocc + [0]*self.virtual)
        self.ref = Bra([oc, oc])
        self.determinants = [self.ref]

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
        psi4.core.print_out("\nCISD Energy: {:<15.10f} ".format(E[0]+self.V_nuc) + emoji('viva'))

    # CAS assuming nbeta = nalpha

    def compute_CAS(self, active_space ='',nfrozen=0, nvirtual=0):

        # Read active space

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

        # GENERATE DETERMINANTS

        print("Generating excitations")
        perms = set(permutations(active))
        print("Generating determinants")
        self.determinants = []
        for p1 in perms:
            for p2 in perms:
                alpha = space.copy()
                beta =  space.copy()
                for i,x in enumerate(np.where(np.array(space) == 'a')[0]):
                    alpha[x] = p1[i]
                    beta[x] = p2[i]
                self.determinants.append(Bra([alpha, beta]))
        print("Number of determinants: {}".format(len(self.determinants)))

        # COMPUTE HAMILTONIAN MATRIX

        print("Generating Hamiltonian Matrix")
        H = get_H(self.determinants, self.MIone, self.MItwo, v = True, t = True)
        print(H[0])

        # DIAGONALIZE HAMILTONIAN MATRIX

        print("Diagonalizing Hamiltonian Matrix")
        t0 = timeit.default_timer()
        E, Ccas = la.eigh(H)
        tf = timeit.default_timer()
        print("Completed. Time needed: {}".format(tf - t0))
        print("CAS Energy:")
        self.E = E[0]
        self.C0 = Ccas[:,0]
        self.Ecas = E[0] + self.V_nuc
        print(self.Ecas)
        psi4.core.print_out("\nCAS Energy: {:<15.10f} ".format(self.Ecas) + emoji('whale'))
        return self.Ecas
