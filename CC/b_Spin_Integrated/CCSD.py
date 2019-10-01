import psi4
import os
import sys
import numpy as np
import time
import copy

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

from tools import *

### FUNCTIONS ###

def cc_energy(T1, T2):

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
    X = 2*tau - np.einsum('ijab->jiab',tau)
    E = np.einsum('abij,ijab->', Vint[v,v,o,o], X)
    return E

def T2_iter(T1, T2):

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
    V = 2*Vint - np.einsum('abij->baij', Vint)
    Te = 0.5*T2 + np.einsum('ia,jb->ijab', T1, T1)
    
    T2new = copy.deepcopy(Vint[o,o,v,v])
    
    T2new += np.einsum('uvij,ijpg->uvpg', Vint[o,o,o,o], tau)
    
    T2new += np.einsum('abpg,uvab->uvpg', Vint[v,v,v,v], tau)
    
    T2new += np.einsum('abij,ijpg,uvab->uvpg', Vint[v,v,o,o], tau, tau)
    
    T2new = 0.5*T2new
    
    T2new += np.einsum('uapg,va->uvpg', Vint[o,v,v,v], T1)
    
    T2new -= np.einsum('uvpi,ig->uvpg', Vint[o,o,v,o], T1)
    
    T2new += np.einsum('auip,viga->uvpg', V[v,o,o,v], T2)
    
    T2new -= np.einsum('uaig,ivpa->uvpg', Vint[o,v,o,v], tau)
    
    T2new -= np.einsum('auip,viag->uvpg', Vint[v,o,o,v], tau)
    
    X = T2 - np.einsum('uipa->uiap',tau)
    
    T2new += np.einsum('abij,vjgb,uipa->uvpg', V[v,v,o,o], T2, X)
    
    T2new -= np.einsum('abij,ijgb,uvpa->uvpg', V[v,v,o,o], tau, T2)
    
    T2new -= np.einsum('abij,vjab,uipg->uvpg', V[v,v,o,o], tau, T2)
    
    T2new += np.einsum('abij,vjbg,uiap->uvpg', Vint[v,v,o,o], T2, Te)
    
    T2new += np.einsum('abij,ujag,vibp->uvpg', Vint[v,v,o,o], T2, Te)
    
    X = np.einsum('ivpg,ja->ivjpga', T2, T1) + np.einsum('jvag,ip->ivjpga', T2, T1)
    
    Y = np.einsum('vjag,ip->vjiagp', T2, T1) + np.einsum('viap,jg->vjiagp', T2, T1) + np.einsum('ijpg,va->vjiagp', tau, T1)
    
    T2new -= np.einsum('uaij,ivjpga,uaij,vjiagp->uvpg', V[o,v,o,o], X, Vint[o,v,o,o], Y)
    
    X = np.einsum('uvag,ib->uviagb', T2, T1) + np.einsum('ivbg,ua->uviagb', T2, T1)
    
    T2new += np.einsum('abpi,uviagb->uvpg', V[v,v,v,o], X)
    
    Y = np.einsum('ivgb,ua->ivugba', T2, T1) + np.einsum('iuga,vb->ivugba', T2, T1) + np.einsum('uvab,ig->ivugba', tau, T1)
    
    T2new -= np.einsum('abpi,ivugba->uvpg', Vint[v,v,v,o], Y)
    
    # Finish it up: Apply the permutation operator and divide by orbital energies [D(ijab)]
    
    T2new = T2new + np.einsum('uvpg->vugp',T2new)
    
    T2new = np.einsum('uvpg,uvpg->uvpg', T2new, D)
    
    res = np.sum(np.abs(T2new - T2))
    
    return T2new, res

def T1_iter(T1, T2):
    # Auxiliar (Intermediate) Arrays 
    
    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
    V = 2*Vint - np.einsum('abij->baij', Vint)
    
    T1new  = -np.einsum('auip,ia->up', V[v,o,o,v], T1)
    
    Y = 2*T2 - tau.swapaxes(2,3) 

    X = np.einsum('ijpb,ua->ijupba',T2,T1) + np.einsum('ujab,ip->ijupba',T2,T1) - np.einsum('ujpb,ia->ijupba', Y, T1)
    
    T1new += np.einsum('abij,ijupba->up', V[v,v,o,o], X)
    
    T1new += np.einsum('auij,ijap->up', V[v,o,o,o], tau)
    
    T1new -= np.einsum('abip,iuab->up', V[v,v,o,v], tau)
    
    T1new = np.einsum('up,up->up', T1new, d)
    
    res = np.sum(np.abs(T1new - T1))
    
    return T1new, res

    

# Input Geometry    

#H2 = psi4.geometry("""
#    0 1
#    H 
#    H 1 0.76
#    symmetry c1
#""")

water = psi4.geometry("""
    0 1
    O
    H 1 0.96
    H 1 0.96 2 104.5
    symmetry c1
""")
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

basis = '3-21g'

# Psi4 Options

psi4.core.be_quiet()
psi4.set_options({'basis': basis,
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'e_convergence' : 1e-10,
                  'freeze_core': 'false'})

# Run Psi4 Energie
print('---------------- RUNNING PSI4 ------------------')
tinit = time.time()
scf_e, wfn = psi4.energy('scf', return_wfn=True)
p4_mp2 = psi4.energy('mp2')
p4_ccsd = psi4.energy('ccsd')

print('SCF  Energy from Psi4: {:<5.10f}'.format(scf_e))
print('MP2  Energy from Psi4: {:<5.10f}'.format(p4_mp2))
print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
print('------------------------------------------------')
print('Psi4 computations completed in {:.5f} seconds\n'.format(time.time() - tinit))

nelec = wfn.nalpha() + wfn.nbeta()
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
nvir = nmo - ndocc
eps = np.asarray(wfn.epsilon_a())
nbf = C.shape[0]

print("Number of Basis Functions:      {}".format(nbf))
print("Number of Electrons:            {}".format(nelec))
print("Number of Molecular Orbitals:   {}".format(nmo))
print("Number of Doubly ocuppied MOs:  {}".format(ndocc))

# Get Integrals

print("Converting atomic integrals to MO integrals...")
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
Vint = np.asarray(mints.mo_eri(C, C, C, C))
# Convert to physicist notation
Vint = Vint.swapaxes(1,2)
print("Completed in {} seconds!".format(time.time()-t))

# Slices

o = slice(0, ndocc)
v = slice(ndocc, nbf)

# START CCSD CODE

# Build the Auxiliar Matrix D

print('\n----------------- RUNNING CCD ------------------')

print('\nBuilding Auxiliar D matrix...')
t = time.time()
D  = np.zeros([ndocc, ndocc, nvir, nvir])
d  = np.zeros([ndocc, nvir])
for i,ei in enumerate(eps[o]):
    for j,ej in enumerate(eps[o]):
        for a,ea in enumerate(eps[v]):
            d[i,a] = 1/(ea - ei)
            for b,eb in enumerate(eps[v]):
                D[i,j,a,b] = 1/(ei + ej - ea - eb)

print('Done. Time required: {:.5f} seconds'.format(time.time() - t))

print('\nComputing MP2 guess')

t = time.time()

T1 = np.zeros([ndocc, nvir])
T2 = np.einsum('abij,ijab->ijab', Vint[v,v,o,o], D)

E = cc_energy(T1, T2)

print('MP2 Energy: {:<5.10f}     Time required: {:.5f}'.format(E+scf_e, time.time()-t))

r1 = 0
r2 = 1
CC_CONV = 6
CC_MAXITER = 20
    
LIM = 10**(-CC_CONV)

ite = 0

while r2 > LIM or r1 > LIM:
    ite += 1
    if ite > CC_MAXITER:
        raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
    Eold = E
    t = time.time()
    T1N, r1 = T1_iter(T1, T2)
    T2,r2 = T2_iter(T1, T2)
    T1 = copy.deepcopy(T1N)
    E = cc_energy(T1, T2)
    dE = E - Eold
    print('-'*50)
    print("Iteration {}".format(ite))
    print("CC Correlation energy: {}".format(E))
    print("Energy change:         {}".format(dE))
    print("T1 Residue:            {}".format(r1))
    print("T2 Residue:            {}".format(r2))
    print("Max T1 Amplitude:      {}".format(np.max(T1)))
    print("Max T2 Amplitude:      {}".format(np.max(T2)))
    print("Time required:         {}".format(time.time() - t))
    print('-'*50)

print("\nCC Equations Converged!!!")
print("Final CCD Energy: {}".format(E + scf_e))
print("Total Computation time:        {}".format(time.time() - tinit))


