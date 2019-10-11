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

def cc_energy(T1aa, T1bb, T2aaaa, T2abab, T2abba, T2bbbb, T2baba, T2baab):

    X = T2aaaa + 2*np.einsum('ia,jb->ijab',T1aa,T1aa, optimize='optimal')
    E = np.einsum('ijab,ijab->', X, Vaaaa[oa,oa,va,va], optimize='optimal')

    X = T2abab + 2*np.einsum('ia,jb->ijab',T1aa,T1bb, optimize='optimal')
    E += np.einsum('ijab,ijab->', X, Vabab[oa,ob,va,vb], optimize='optimal')

    E += np.einsum('ijab,ijab->', T2abba, Vabba[oa,ob,vb,va], optimize='optimal')

    X = T2bbbb + 2*np.einsum('ia,jb->ijab',T1bb,T1bb, optimize='optimal')
    E += np.einsum('ijab,ijab->', X, Vbbbb[ob,ob,vb,vb], optimize='optimal')

    X = T2baba + 2*np.einsum('ia,jb->ijab',T1bb,T1aa, optimize='optimal')
    E += np.einsum('ijab,ijab->', X, Vbaba[ob,oa,vb,va], optimize='optimal')

    E += np.einsum('ijab,ijab->', T2baab, Vbaab[ob,oa,va,vb], optimize='optimal')

    return (1.0/4.0) * E

def CCSD_Iter(T1, T2, EINSUMOPT='optimal'):

    # Intermediate arrays

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1,optimize=EINSUMOPT)
    Te = 0.5*T2 + np.einsum('ia,jb->ijab', T1, T1,optimize=EINSUMOPT)

    A2l = np.einsum('uvij,ijpg->uvpg', Vint[o,o,o,o], tau,optimize=EINSUMOPT)
    B2l = np.einsum('abpg,uvab->uvpg', Vint[v,v,v,v], tau,optimize=EINSUMOPT)
    C1  = np.einsum('uaip,ia->uip', Vint[o,v,o,v], T1,optimize=EINSUMOPT) 
    C2  = np.einsum('aupi,viga->pvug', Vint[v,o,v,o], T2,optimize=EINSUMOPT)
    C2l = np.einsum('iaug,ivpa->pvug', Vint[o,v,o,v], tau,optimize=EINSUMOPT)
    D1  = np.einsum('uapi,va->uvpi', Vint[o,v,v,o], T1,optimize=EINSUMOPT)
    D2l = np.einsum('abij,uvab->uvij',Vint[v,v,o,o], tau,optimize=EINSUMOPT)
    Ds2l= np.einsum('acij,ijpb->acpb',Vint[v,v,o,o], tau,optimize=EINSUMOPT)
    D2a = np.einsum('baji,vjgb->avig', Vint[v,v,o,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)
    D2b = np.einsum('baij,vjgb->avig', Vint[v,v,o,o], T2,optimize=EINSUMOPT)
    D2c = np.einsum('baij,vjbg->avig', Vint[v,v,o,o], T2,optimize=EINSUMOPT)
    Es1 = np.einsum('uvpi,ig->uvpg', Vint[o,o,v,o], T1,optimize=EINSUMOPT)
    E1  = np.einsum('uaij,va->uvij', Vint[o,v,o,o], T1,optimize=EINSUMOPT)
    E2a = np.einsum('buji,vjgb->uvig', Vint[v,o,o,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)
    E2b = np.einsum('buij,vjgb->uvig', Vint[v,o,o,o], T2,optimize=EINSUMOPT)
    E2c = np.einsum('buij,vjbg->uvig', Vint[v,o,o,o], T2,optimize=EINSUMOPT)
    F11 = np.einsum('bapi,va->bvpi', Vint[v,v,v,o], T1,optimize=EINSUMOPT)
    F12 = np.einsum('baip,va->bvip', Vint[v,v,o,v], T1,optimize=EINSUMOPT)
    Fs1 = np.einsum('acpi,ib->acpb', Vint[v,v,v,o], T1,optimize=EINSUMOPT)
    F2a = np.einsum('abpi,uiab->aup', Vint[v,v,v,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT) 
    F2l = np.einsum('abpi,uvab->uvpi', Vint[v,v,v,o], tau,optimize=EINSUMOPT)

    X = E1 + D2l
    giu = np.einsum('ujij->ui', 2*X - X.transpose(0,1,3,2),optimize=EINSUMOPT)
    
    X = Fs1 - Ds2l
    gap = np.einsum('abpb->ap', 2*X - X.transpose(1,0,2,3),optimize=EINSUMOPT)

    # T2 Amplitudes update

    J = np.einsum('ag,uvpa->uvpg', gap, T2,optimize=EINSUMOPT) - np.einsum('vi,uipg->uvpg', giu, T2,optimize=EINSUMOPT)

    S = 0.5*A2l + 0.5*B2l - Es1 - (C2 + C2l - D2a - F12).transpose(2,1,0,3)  
    S += np.einsum('avig,uipa->uvpg', (D2a-D2b), T2 - Te.transpose(0,1,3,2),optimize=EINSUMOPT)
    S += 0.5*np.einsum('avig,uipa->uvpg', D2c, T2,optimize=EINSUMOPT)
    S += np.einsum('auig,viap->uvpg', D2c, Te,optimize=EINSUMOPT)
    S += np.einsum('uvij,ijpg->uvpg', 0.5*D2l + E1, tau,optimize=EINSUMOPT)
    S -= np.einsum('uvpi,ig->uvpg', D1 + F2l, T1,optimize=EINSUMOPT)
    S -= np.einsum('uvig,ip->uvpg',E2a - E2b - E2c.transpose(1,0,2,3), T1,optimize=EINSUMOPT)
    S -= np.einsum('avgi,uipa->uvpg', F11, T2,optimize=EINSUMOPT)
    S -= np.einsum('avpi,uiag->uvpg', F11, T2,optimize=EINSUMOPT)
    S += np.einsum('avig,uipa->uvpg', F12, 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)

    T2new = Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2)

    T2new = np.einsum('uvpg,uvpg->uvpg', T2new, D,optimize=EINSUMOPT)

    res2 = np.sum(np.abs(T2new - T2))

    # T1 Amplitudes update
    
    T1new = np.einsum('ui,ip->up', giu, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('ap,ua->up', gap, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('juai,ja,ip->up', 2*D1 - D1.transpose(3,1,2,0), T1, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('auip,ia->up', 2*(D2a - D2b) + D2c, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('aup->up', F2a,optimize=EINSUMOPT)
    T1new += np.einsum('uiip->up', 1.0/2.0*(E2a - E2b) + E2c,optimize=EINSUMOPT)
    T1new += np.einsum('uip->up', C1,optimize=EINSUMOPT)
    T1new -= 2*np.einsum('uipi->up', D1,optimize=EINSUMOPT)

    T1new = np.einsum('up,up->up', T1new, d,optimize=EINSUMOPT)
    
    res1 = np.sum(np.abs(T1new - T1))

    return T1new, T2new, res1, res2

# Input Geometry    

water = psi4.geometry("""
    0 1
    O
    H 1 0.96
    H 1 0.96 2 104.5
    symmetry c1
""")

# Psi4 Options

psi4.core.be_quiet()
psi4.set_options({'basis': 'sto-3g',
                  'reference': 'uhf',                
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

neleca = wfn.nalpha()
nelecb = wfn.nbeta()
nelec = neleca + nelecb
Ca = wfn.Ca()
Cb = wfn.Cb()
ndocc = wfn.doccpi()[0] # Is this meaningfull?
nbf = Ca.shape[0]
naso = Ca.shape[0] 
nbso = Cb.shape[0]
navir = naso - neleca
nbvir = nbso - nelecb
epsa = np.asarray(wfn.epsilon_a())
epsb = np.asarray(wfn.epsilon_b())

print("Number of Basis Functions:      {}".format(nbf))
print("Number of Alpha Electrons:      {}".format(neleca))
print("Number of Beta Electrons:       {}".format(nelecb))
print("Number of Alpha Spin-Orbitals:  {}".format(naso))
print("Number of Beta Spin-Orbitals:   {}".format(nbso))

# Get Integrals

print("Creating Antisymmetrized MO Integrals")

t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())

print("AA|AA")
Vaaaa = np.asarray(mints.mo_eri(Ca, Ca, Ca, Ca))
Vaaaa = Vaaaa - Vaaaa.transpose(0,3,2,1)
# Physicist's Notation
Vaaaa = Vaaaa.transpose(0,2,1,3)

print("BB|BB")
Vbbbb = np.asarray(mints.mo_eri(Cb, Cb, Cb, Cb))
Vbbbb = Vbbbb - Vbbbb.transpose(0,3,2,1)
# Physicist's Notation
Vbbbb = Vbbbb.transpose(0,2,1,3)

print("AB|AB")
Vabab = np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb))
# Physicist's Notation
Vabab = Vabab.transpose(0,2,1,3)

print("AB|BA")
Vabba = (-1)*np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb))
# Physicist's Notation
Vabba = Vabba.transpose(0,2,3,1)

print("BA|BA")
Vbaba = np.asarray(mints.mo_eri(Cb, Cb, Ca, Ca))
# Physicist's Notation
Vbaba = Vbaba.transpose(0,2,1,3)

print("BA|AB")
Vbaab = (-1)*np.asarray(mints.mo_eri(Cb, Cb, Ca, Ca))
# Physicist's Notation
Vbaab = Vbaab.transpose(0,2,3,1)

print("Completed in {} seconds!".format(time.time()-t))

# Slices

oa = slice(0, neleca)
va = slice(neleca, naso)

ob = slice(0, nelecb)
vb = slice(nelecb, nbso)

# START CCSD CODE

# Build the Auxiliar Matrix D

print('\n----------------- RUNNING CCD ------------------')

print('\nBuilding Auxiliar D matrices...')
t = time.time()

print('d(AA)')
daa  = np.zeros([neleca, navir])
for i,ei in enumerate(epsa[oa]):
    for a,ea in enumerate(epsa[va]): 
        daa[i,a] = 1/(ei - ea)

print('d(BB)')
dbb  = np.zeros([nelecb, nbvir])
for i,ei in enumerate(epsb[ob]):
    for a,ea in enumerate(epsb[vb]): 
        dbb[i,a] = 1/(ei - ea)

print('D(AAAA)')
Daaaa  = np.zeros([neleca, neleca, navir, navir])
for i,ei in enumerate(epsa[oa]):
    for j,ej in enumerate(epsa[oa]):
        for a,ea in enumerate(epsa[va]):
            for b,eb in enumerate(epsa[va]):
                Daaaa[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(BBBB)')
Dbbbb  = np.zeros([nelecb, nelecb, nbvir, nbvir])
for i,ei in enumerate(epsb[ob]):
    for j,ej in enumerate(epsb[ob]):
        for a,ea in enumerate(epsb[vb]):
            for b,eb in enumerate(epsb[vb]):
                Dbbbb[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(ABAB)')
Dabab  = np.zeros([neleca, nelecb, navir, nbvir])
for i,ei in enumerate(epsa[oa]):
    for j,ej in enumerate(epsb[ob]):
        for a,ea in enumerate(epsa[va]):
            for b,eb in enumerate(epsb[vb]):
                Dabab[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(ABBA)')
Dabba  = np.zeros([neleca, nelecb, nbvir, navir])
for i,ei in enumerate(epsa[oa]):
    for j,ej in enumerate(epsb[ob]):
        for a,ea in enumerate(epsb[vb]):
            for b,eb in enumerate(epsa[va]):
                Dabba[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(BABA)')
Dbaba  = np.zeros([nelecb, neleca, nbvir, navir])
for i,ei in enumerate(epsb[ob]):
    for j,ej in enumerate(epsa[oa]):
        for a,ea in enumerate(epsb[vb]):
            for b,eb in enumerate(epsa[va]):
                Dbaba[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(BAAB)')
Dbaab  = np.zeros([nelecb, neleca, navir, nbvir])
for i,ei in enumerate(epsb[ob]):
    for j,ej in enumerate(epsa[oa]):
        for a,ea in enumerate(epsa[va]):
            for b,eb in enumerate(epsb[vb]):
                Dbaab[i,j,a,b] = 1/(ei + ej - ea - eb)

print('Done. Time required: {:.5f} seconds'.format(time.time() - t))

print('\nComputing MP2 guess')

t = time.time()

T1aa = np.zeros([neleca, navir])
T1bb = np.zeros([nelecb, nbvir])

T2aaaa = np.einsum('ijab,ijab->ijab', Vaaaa[oa,oa,va,va], Daaaa)
T2bbbb = np.einsum('ijab,ijab->ijab', Vbbbb[ob,ob,vb,vb], Dbbbb)
T2abab = np.einsum('ijab,ijab->ijab', Vabab[oa,ob,va,vb], Dabab)
T2abba = np.einsum('ijab,ijab->ijab', Vabba[oa,ob,vb,va], Dabba)
T2baba = np.einsum('ijab,ijab->ijab', Vbaba[ob,oa,vb,va], Dbaba)
T2baab = np.einsum('ijab,ijab->ijab', Vbaab[ob,oa,va,vb], Dbaab)

E = cc_energy(T1aa, T1bb, T2aaaa, T2abab, T2abba, T2bbbb, T2baba, T2baab)

print('MP2 Energy: {:<5.10f}     Time required: {:.5f}'.format(E+scf_e, time.time()-t))

r1 = 0
r2 = 1
CC_CONV = 6
CC_MAXITER = 30
    
LIM = 10**(-CC_CONV)

ite = 0

while r2 > LIM or r1 > LIM:
    ite += 1
    if ite > CC_MAXITER:
        raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
    Eold = E
    t = time.time()
    T1, T2, r1, r2 = CCSD_Iter(T1, T2)
    T1aa, T1bb, T2aaaa, T2abab, Tabba, T2bbbb, T2baba, T2baab = CCSD_Iter(T1aa, T1bb, T2aaaa, T2abab, T2abba, T2bbbb, T2baba, T2baab)
    E = cc_energy(T1aa, T1bb, T2aaaa, T2abab, T2abba, T2bbbb, T2baba, T2baab)
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
print("Final CCSD Energy:     {:<5.10f}".format(E + scf_e))
print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
print("Total Computation time:        {}".format(time.time() - tinit))


