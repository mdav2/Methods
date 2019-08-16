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
from rhf import RHF

class CC:

# Compute Coupled Cluster Energies

    def __init__(self, HF):
        self.E0 = HF.E
        self.orbitals = HF.orbitals
        self.ndocc = HF.ndocc
        self.nelec = HF.nelec
        self.nbf = HF.nbf
        self.virtual = self.nbf - self.ndocc
        self.V_nuc = HF.V_nuc
        self.h = HF.T + HF.V
        self.F = HF.Fmol
        self.eo = HF.Eorb
        self.holes = range(0, self.ndocc)
        self.particles = range(self.ndocc, self.nbf)
        print("Number of electrons: {}".format(self.nelec))
        print("Number of basis functions: {}".format(self.nbf))
        print("Number of doubly occupied orbitals: {}".format(self.ndocc))
        print("Number of virtual spatial orbitals: {}".format(self.virtual))

# Convert atomic integrals to MO integrals
        psi4_orb = psi4.core.Matrix.from_array(self.orbitals)
        print("Converting atomic integrals to MO integrals...")
        self.MIone = np.einsum('up,vq,uv->pq', self.orbitals, self.orbitals, self.h)
        self.MItwo = np.asarray(HF.mints.mo_eri(psi4_orb, psi4_orb, psi4_orb, psi4_orb))
        # Convert to physicist notation
        self.MItwo = self.MItwo.swapaxes(1,2)
        print("Completed!")

# Functions to compute Amplitudes


    def CC_Energy(self, T1a, T2a):
        #return np.einsum('ijab,ijab->',self.MItwo, T2a) - np.einsum('ijba,ijab->',self.MItwo, T2a)
        e2 = 0
        mp2 = 0
        for i in self.holes:
            for j in self.holes:
                for a in self.particles:
                    for b in self.particles:
                        e2 = e2 + 0.5*(3*self.MItwo[i,j,a,b] - self.MItwo[i,j,b,a])*T2a[i,j,a,b]
                        mp2 += (self.MItwo[i,j,a,b]*(2*self.MItwo[i,j,a,b]-self.MItwo[i,j,b,a]))/(self.eo[i] + self.eo[j] - self.eo[a] - self.eo[b])
        print('MP2 corelation: {}'.format(mp2))
        return e2
        
        return np.einsum('ijab,ijab->',self.g, T2a) 

    def SDT1(self, T1_init, T2_init):
        pass

    def SDT2(self, T1_init, T2_init):
        pass

    def CCSD(self, CC_CONV, CC_MAXITER):
        
        # Compute initial guess for T1 and T2 amplitudes
        T1 = np.zeros([self.nbf, self.nbf])

        # Build auxiliar D matrix

        D = np.zeros([self.nbf, self.nbf, self.nbf, self.nbf])
        T2loop = np.zeros([self.nbf, self.nbf, self.nbf, self.nbf])
        for i in self.holes:
            for j in self.holes:
                for a in self.particles:
                    for b in self.particles:
                        D[i,j,a,b] = 1/(self.eo[i] + self.eo[j] - self.eo[a] - self.eo[b])
                        D[j,i,a,b] = D[i,j,a,b]
                        D[i,j,b,a] = D[i,j,a,b]
                        D[j,i,b,a] = D[i,j,a,b]
                        T2loop[i,j,a,b] =  4*(self.MItwo[i,j,a,b]-self.MItwo[i,j,b,a])/(self.eo[i] + self.eo[j] - self.eo[a] - self.eo[b])
        
        #T2 = 4*np.einsum('abij,ijab->ijab',self.g, D)
        
        Emp2 = self.CC_Energy(T1, T2loop) + self.E0
        psi4.core.print_out('CC MP2 Energy:    {:<5.10f}\n'.format(Emp2))
