import psi4
import os
import sys
import numpy as np
import time

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

from tools import *

### FUNCTIONS ###

def cc_energy(T2,Aoovv):
    return (1/4)*np.einsum('ijab,ijab->',Aoovv,T2)

# Input Geometry    

H2 = psi4.geometry("""
    0 1
    H 
    H 1 0.76
    symmetry c1
""")

#water = psi4.geometry("""
#    0 1
#    O
#    H 1 0.96
#    H 1 0.96 2 104.5
#    symmetry c1
#""")
#
#form = psi4.geometry("""
#0 1
#O
#C 1 1.22
#H 2 1.08 1 120.0
#H 2 1.08 1 120.0 3 -180.0
#symmetry c1
#""")

# Basis set

basis = 'sto-3g'

# Psi4 Options

psi4.core.be_quiet()
psi4.set_options({'basis': basis,
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false'})

# Run Psi4 Energie
print('---------------- RUNNING PSI4 ------------------')
t = time.time()
scf_e, wfn = psi4.energy('scf', return_wfn=True)
p4_mp2 = psi4.energy('mp2')
p4_ccsd = psi4.energy('ccsd')

print('SCF  Energy from Psi4: {:<5.10f}'.format(scf_e))
print('MP2  Energy from Psi4: {:<5.10f}'.format(p4_mp2))
print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
print('------------------------------------------------')
print('Psi4 computations completed in {:.5f} seconds\n'.format(time.time() - t))

nelec = wfn.nalpha() + wfn.nbeta()
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nocc = ndocc * 2 + nelec % 2
nmo = wfn.nmo()
nso = nmo * 2
nvir = nso - nocc
eps = np.asarray(wfn.epsilon_a())
eps = np.repeat(eps, 2)
nbf = C.shape[0]

print("Number of Basis Functions:      {}".format(nbf))
print("Number of Electrons:            {}".format(nelec))
print("Number of Spin-Orbitals (SO):   {}".format(nso))
print("Number of Ocuppied SO:          {}".format(nocc))

# Get Integrals

mints = psi4.core.MintsHelper(wfn.basisset())
V = np.asarray(mints.mo_spin_eri(C,C)).swapaxes(1,2)

# Slices

o = slice(0, nocc)
v = slice(nocc, 2*nbf)

# START CCD CODE

# Build the Auxiliar Matrix D

print('\n----------------- RUNNING CCD ------------------')

print('\nBuilding Auxiliar D matrix...')
t = time.time()
D  = np.zeros([nocc, nocc, nvir, nvir])

for i,ei in enumerate(eps[o]):
    for j,ej in enumerate(eps[o]):
        for a,ea in enumerate(eps[v]):
            for b,eb in enumerate(eps[v]):
                D[i,j,a,b] = 1/(ei + ej - ea - eb)
print('Done. Time required: {:.5f} seconds'.format(time.time() - t))

Aoovv = V[o,o,v,v] - np.einsum('ijab->ijba',V[o,o,v,v])

print('\nComputing MP2 guess')

t = time.time()

T2 = np.einsum('ijab,ijab->ijab', Aoovv, D)

Eold = cc_energy(T2, Aoovv)

print('MP2 Energy: {:<5.10f}     Time required: {:.5f}'.format(Eold+scf_e, time.time()-t))
    

