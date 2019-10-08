import psi4
import os
import sys
import numpy as np
import scipy.linalg as la
import time
import copy
from itertools import permutations

file_dir = os.path.dirname('../Aux/')
sys.path.append(file_dir)

from tools import *
from fock import *
from Hamiltonian import *

class TCCSD:

    def __init__(self, mol):

        # Run SCF information 

        self.Escf, wfn = psi4.energy('scf', return_wfn = True)
        self.nelec = wfn.nalpha() + wfn.nbeta()
        self.C = wfn.Ca()
        self.ndocc = wfn.doccpi()[0]
        self.nmo = wfn.nmo()
        self.nvir = self.nmo - self.ndocc
        self.eps = np.asarray(wfn.epsilon_a())
        self.nbf = self.C.shape[0]
        self.Vnuc = mol.nuclear_repulsion_energy()
        
        print("Number of Electrons:            {}".format(self.nelec))
        print("Number of Basis Functions:      {}".format(self.nbf))
        print("Number of Molecular Orbitals:   {}".format(self.nmo))
        print("Number of Doubly ocuppied MOs:  {}\n".format(self.ndocc))
    
        # Get Integrals.
    
        print("Converting atomic integrals to MO integrals...")
        t = time.time()
        mints = psi4.core.MintsHelper(wfn.basisset())
        self.Vint = np.asarray(mints.mo_eri(self.C, self.C, self.C, self.C))
        self.Vint = self.Vint.swapaxes(1,2) # Convert to Physicists' notation
        self.h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        self.h = np.einsum('up,vq,uv->pq', self.C, self.C, self.h)
        print("Completed in {} seconds!".format(time.time()-t))

    
    def CAS(self, active_space='', show_prog = False):

        # Read active space and determine number of active electrons. 
        # Format: sequence of letters ordered
        # according to orbital energies. 
        # Legend:
        # o = frozen doubly occupied orbital
        # a = active orbital
        # u = frozen unnocupied orbital

        template_space = ''
        n_ac_orb = 0
        n_ac_elec_pair = int(self.nelec/2)
        print("Reading Active space")
        if active_space == 'full':
            active_space = 'a'*self.nmo
        if len(active_space) != self.nbf:
            raise NameError("Invalid active space format. Please check the number of basis functions.")
        for i in active_space:
            if i == 'o':
                template_space += '1'
                n_ac_elec_pair -= 1
            elif i == 'a':
                template_space += '{}'
                n_ac_orb += 1
            elif i == 'u':
                template_space += '0'
            else:
                raise NameError("Invalid active space entry: {}".format(i))

        # Produce a reference determinant

        self.ref = Det(a = ('1'*self.ndocc + '0'*self.nvir), \
                       b = ('1'*self.ndocc + '0'*self.nvir))

        print("Number of active orbitals: {}".format(n_ac_orb))
        print("Number of active electrons: {}\n".format(2*n_ac_elec_pair))

        # Use permutations to generate strings that will represent excited determinants

        print("Generating excitations...")
        perms = set(permutations('1'*n_ac_elec_pair + '0'*(n_ac_orb - n_ac_elec_pair)))
        print("Done.\n")

        # Use the strings to generate Det objects 

        self.determinants = []
        progress = 0
        file = sys.stdout
        for p1 in perms:
            for p2 in perms:
                self.determinants.append(Det(a=template_space.format(*p1), \
                                             b=template_space.format(*p2)))
            progress += 1
            showout(progress, len(perms), 50, "Generating Determinants: ", file)
        file.write('\n')
        file.flush()
        print("Number of determinants: {}".format(len(self.determinants)))

        # Construct the Hamiltonian Matrix
        # Note: Input for two electron integral must be using Chemists' notation

        H = get_H(self.determinants, self.h, self.Vint.swapaxes(1,2), v = True, t = True)

        # Diagonalize the Hamiltonian Matrix

        print("Diagonalizing Hamiltonian Matrix")
        t0 = time.time()
        E, Ccas = la.eigh(H)
        tf = time.time()
        self.Ecas = E[0]
        self.Ccas = Ccas[:,0]
        print("Completed. Time needed: {}".format(tf - t0))
        print("CAS Energy: {:<5.10f}".format(self.Ecas + self.Vnuc))

    def cc_energy(self):
    
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)
        tau = self.T2 + np.einsum('ia,jb->ijab', self.T1, self.T1)
        X = 2*tau - np.einsum('ijab->jiab',tau)
        self.Ecc = np.einsum('abij,ijab->', self.Vint[v,v,o,o], X)

    def T1_T2_Update(self, EINSUMOPT='optimal'):
    
        # Compute CCSD Amplitudes. Only the T1 (alpha -> alpha) are considered since the beta -> beta case yields the same amplitude and the mixed case is zero.
        # For T2 amplitudes we consider the case (alpha -> alpha, beta -> beta) the other spin cases can be writen in terms of this one.
        # Equations from J. Chem. Phys. 86, 2881 (1987): G. E. Scuseria et al.
        # CC Intermediate arrays
    
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)

        tau = self.T2 + np.einsum('ia,jb->ijab', self.T1, self.T1,optimize=EINSUMOPT)
        Te = 0.5*self.T2 + np.einsum('ia,jb->ijab', self.T1, self.T1,optimize=EINSUMOPT)
    
        A2l = np.einsum('uvij,ijpg->uvpg', self.Vint[o,o,o,o], tau,                                    optimize=EINSUMOPT)
        B2l = np.einsum('abpg,uvab->uvpg', self.Vint[v,v,v,v], tau,                                    optimize=EINSUMOPT)
        C1  = np.einsum('uaip,ia->uip',    self.Vint[o,v,o,v], self.T1,                                optimize=EINSUMOPT) 
        C2  = np.einsum('aupi,viga->pvug', self.Vint[v,o,v,o], self.T2,                                optimize=EINSUMOPT)
        C2l = np.einsum('iaug,ivpa->pvug', self.Vint[o,v,o,v], tau,                                    optimize=EINSUMOPT)
        D1  = np.einsum('uapi,va->uvpi',   self.Vint[o,v,v,o], self.T1,                                optimize=EINSUMOPT)
        D2l = np.einsum('abij,uvab->uvij', self.Vint[v,v,o,o], tau,                                    optimize=EINSUMOPT)
        Ds2l= np.einsum('acij,ijpb->acpb', self.Vint[v,v,o,o], tau,                                    optimize=EINSUMOPT)
        D2a = np.einsum('baji,vjgb->avig', self.Vint[v,v,o,o], 2*self.T2 - self.T2.transpose(0,1,3,2), optimize=EINSUMOPT)
        D2b = np.einsum('baij,vjgb->avig', self.Vint[v,v,o,o], self.T2,                                optimize=EINSUMOPT)
        D2c = np.einsum('baij,vjbg->avig', self.Vint[v,v,o,o], self.T2,                                optimize=EINSUMOPT)
        Es1 = np.einsum('uvpi,ig->uvpg',   self.Vint[o,o,v,o], self.T1,                                optimize=EINSUMOPT)
        E1  = np.einsum('uaij,va->uvij',   self.Vint[o,v,o,o], self.T1,                                optimize=EINSUMOPT)
        E2a = np.einsum('buji,vjgb->uvig', self.Vint[v,o,o,o], 2*self.T2 - self.T2.transpose(0,1,3,2), optimize=EINSUMOPT)
        E2b = np.einsum('buij,vjgb->uvig', self.Vint[v,o,o,o], self.T2,                                optimize=EINSUMOPT)
        E2c = np.einsum('buij,vjbg->uvig', self.Vint[v,o,o,o], self.T2,                                optimize=EINSUMOPT)
        F11 = np.einsum('bapi,va->bvpi',   self.Vint[v,v,v,o], self.T1,                                optimize=EINSUMOPT)
        F12 = np.einsum('baip,va->bvip',   self.Vint[v,v,o,v], self.T1,                                optimize=EINSUMOPT)
        Fs1 = np.einsum('acpi,ib->acpb',   self.Vint[v,v,v,o], self.T1,                                optimize=EINSUMOPT)
        F2a = np.einsum('abpi,uiab->aup',  self.Vint[v,v,v,o], 2*self.T2 - self.T2.transpose(0,1,3,2), optimize=EINSUMOPT) 
        F2l = np.einsum('abpi,uvab->uvpi', self.Vint[v,v,v,o], tau,                                    optimize=EINSUMOPT)
    
        X = E1 + D2l

        giu = np.einsum('ujij->ui', 2*X - X.transpose(0,1,3,2), optimize=EINSUMOPT)
        
        X = Fs1 - Ds2l
        gap = np.einsum('abpb->ap', 2*X - X.transpose(1,0,2,3), optimize=EINSUMOPT)
    
        # T2 Amplitudes update
    
        J = np.einsum('ag,uvpa->uvpg', gap, self.T2, optimize=EINSUMOPT) - np.einsum('vi,uipg->uvpg', giu, self.T2, optimize=EINSUMOPT)
    
        S = 0.5*A2l + 0.5*B2l - Es1 - (C2 + C2l - D2a - F12).transpose(2,1,0,3)  
        S +=     np.einsum('avig,uipa->uvpg', (D2a-D2b), self.T2 - Te.transpose(0,1,3,2),  optimize=EINSUMOPT)
        S += 0.5*np.einsum('avig,uipa->uvpg', D2c, self.T2,                                optimize=EINSUMOPT)
        S +=     np.einsum('auig,viap->uvpg', D2c, Te,                                     optimize=EINSUMOPT)
        S +=     np.einsum('uvij,ijpg->uvpg', 0.5*D2l + E1, tau,                           optimize=EINSUMOPT)
        S -=     np.einsum('uvpi,ig->uvpg',   D1 + F2l, self.T1,                           optimize=EINSUMOPT)
        S -=     np.einsum('uvig,ip->uvpg',   E2a - E2b - E2c.transpose(1,0,2,3), self.T1, optimize=EINSUMOPT)
        S -=     np.einsum('avgi,uipa->uvpg', F11, self.T2,                                optimize=EINSUMOPT)
        S -=     np.einsum('avpi,uiag->uvpg', F11, self.T2,                                optimize=EINSUMOPT)
        S +=     np.einsum('avig,uipa->uvpg', F12, 2*self.T2 - self.T2.transpose(0,1,3,2), optimize=EINSUMOPT)
    
        T2new = self.Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2)
    
        T2new = np.einsum('uvpg,uvpg->uvpg', T2new, self.D,optimize=EINSUMOPT)
    
        self.r2 = np.sum(np.abs(T2new - self.T2))
    
        # T1 Amplitudes update
        
        T1new =    np.einsum('ui,ip->up',      giu, self.T1,                                   optimize=EINSUMOPT)
        T1new -=   np.einsum('ap,ua->up',      gap, self.T1,                                   optimize=EINSUMOPT)
        T1new -=   np.einsum('juai,ja,ip->up', 2*D1 - D1.transpose(3,1,2,0), self.T1, self.T1, optimize=EINSUMOPT)
        T1new -=   np.einsum('auip,ia->up',    2*(D2a - D2b) + D2c, self.T1,                   optimize=EINSUMOPT)
        T1new -=   np.einsum('aup->up',        F2a,                                            optimize=EINSUMOPT)
        T1new +=   np.einsum('uiip->up',       1.0/2.0*(E2a - E2b) + E2c,                      optimize=EINSUMOPT)
        T1new +=   np.einsum('uip->up',        C1,                                             optimize=EINSUMOPT)
        T1new -= 2*np.einsum('uipi->up',       D1,                                             optimize=EINSUMOPT)
    
        T1new = np.einsum('up,up->up', T1new, self.d, optimize=EINSUMOPT)
        
        self.r1 = np.sum(np.abs(T1new - self.T1))
    
        self.T1, self.T2 = T1new, T2new

    def CCSD(self, CC_CONV=6, CC_MAXITER = 50, psi4_compare=False):
        
        # Run Psi4 Energie
        tinit = time.time()
        if psi4_compare:
            print('---------------- RUNNING PSI4 ------------------')
            p4_mp2 = psi4.energy('mp2')
            p4_ccsd = psi4.energy('ccsd')

            print('MP2  Energy from Psi4: {:<5.10f}'.format(p4_mp2))
            print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
            print('------------------------------------------------')
            print('Psi4 computations completed in {:.5f} seconds\n'.format(time.time() - tinit))

        # Slices
        
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)

        # START CCSD CODE

        # Build the Auxiliar Matrix D

        print('\n----------------- RUNNING CCSD ------------------')
        
        print('\nBuilding Auxiliar D matrix...')
        t = time.time()
        self.D  = np.zeros([self.ndocc, self.ndocc, self.nvir, self.nvir])
        self.d  = np.zeros([self.ndocc, self.nvir])
        for i,ei in enumerate(self.eps[o]):
            for j,ej in enumerate(self.eps[o]):
                for a,ea in enumerate(self.eps[v]):
                    self.d[i,a] = 1/(ea - ei)
                    for b,eb in enumerate(self.eps[v]):
                        self.D[i,j,a,b] = 1/(ei + ej - ea - eb)
        
        print('Done. Time required: {:.5f} seconds'.format(time.time() - t))
        
        print('\nComputing MP2 guess')
        
        t = time.time()
        
        self.T1 = np.zeros([self.ndocc, self.nvir])
        self.T2 = np.einsum('abij,ijab->ijab', self.Vint[v,v,o,o], self.D)
        self.cc_energy()
        
        print('MP2 Energy: {:<5.10f}     Time required: {:.5f}'.format(self.Ecc+self.Escf, time.time()-t))
        
        self.r1 = 1
        self.r2 = 1
            
        LIM = 10**(-CC_CONV)
        ite = 0
        
        while self.r2 > LIM or self.r1 > LIM:
            ite += 1
            if ite > CC_MAXITER:
                raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
            Eold = self.Ecc
            t = time.time()
            self.T1_T2_Update()
            self.cc_energy()
            dE = self.Ecc - Eold
            print('-'*50)
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {}".format(self.Ecc))
            print("Energy change:         {}".format(dE))
            print("T1 Residue:            {}".format(self.r1))
            print("T2 Residue:            {}".format(self.r2))
            print("Time required:         {}".format(time.time() - t))
            print('-'*50)
        
        print("\nCC Equations Converged!!!")
        print("Final CCSD Energy:     {:<5.10f}".format(self.Ecc + self.Escf))
        if psi4_compare: print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
        print("Total Computation time:        {}".format(time.time() - tinit))

    def TCCSD(self, active_space=''):
        
        # Compute CAS
        print('------- COMPLETE ACTIVE SPACE CONFIGURATION INTERACTION STARTED -------\n')
        if active_space == 'full':
            active_space = 'a'*self.nmo

        self.CAS(active_space)
        
        print('------- COMPLETE ACTIVE SPACE CONFIGURATION INTERACTION FINISHED -------\n')

        print('Collecting C1 and C2 coefficients...\n')

        # Determine indexes for the CAS space

        CAS_holes = []
        for i,x in enumerate(active_space[0:self.ndocc]):
            if x == 'a':
                CAS_holes.append(i)

        CAS_particles = []
        for i,x in enumerate(active_space[self.ndocc:]):
            if x == 'a':
                CAS_particles.append(i)

        self.external_T1 = np.zeros([self.ndocc, self.nvir])
        self.external_T2 = np.zeros([self.ndocc, self.ndocc, self.nvir, self.nvir])

        C0 = self.Ccas[self.determinants.index(self.ref)]

        # Search for the appropriate coefficients using a model Determinant

        # Singles

        for i in CAS_holes:
            for a in CAS_particles:
                search = self.ref.copy()
                search.rmv_alpha(i)
                search.add_alpha(a+self.ndocc)
                index = self.determinants.index(search)
                self.external_T1[i,a] = self.Ccas[index]

        # Doubles

        for i in CAS_holes:
            for a in CAS_particles:
                for j in CAS_holes:
                    for b in CAS_particles:
                        search = self.ref.copy()
                        search.rmv_alpha(i)
                        search.add_alpha(a+self.ndocc)
                        search.rmv_beta(j)
                        search.add_beta(b+self.ndocc)
                        index = self.determinants.index(search)
                        self.external_T2[i,j,a,b] = self.Ccas[index]

        # Translate CI coefficients into CC amplitudes

        print('Translating CI coefficients into CC amplitudes...\n')

        self.external_T1 = self.external_T1/C0

        for i in CAS_holes:
            for j in CAS_holes:
                for a in CAS_particles:
                    for b in CAS_particles:
                        self.external_T2[i,j,a,b] = self.external_T2[i,j,a,b]/C0 - \
                        (self.external_T1[i,a]*self.external_T1[j,b] - self.external_T1[j,a]*self.external_T1[i,b])
        
       # Compute CCSD 

        print('------- TAILORED COUPLED CLUSTER STARTED -------\n')

        # Slices
        
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)

        # START CCSD CODE

        # Build the Auxiliar Matrix D

        print('Building Auxiliar D matrix...\n')
        print(self.eps)
        print(CAS_holes)
        print(CAS_particles)
        t = time.time()
        self.D  = np.zeros([self.ndocc, self.ndocc, self.nvir, self.nvir])
        self.d  = np.zeros([self.ndocc, self.nvir])
        for i,ei in enumerate(self.eps[o]):
            if i in CAS_holes: continue
            for j,ej in enumerate(self.eps[o]):
                if j in CAS_holes: continue
                for a,ea in enumerate(self.eps[v]):
                    if a in CAS_particles: continue
                    self.d[i,a] = 1/(ea - ei)
                    for b,eb in enumerate(self.eps[v]):
                        if b in CAS_particles: continue
                        self.D[i,j,a,b] = 1/(ei + ej - ea - eb)
        
        print('Done. Time required: {:.5f} seconds'.format(time.time() - t))
        
        t = time.time()
        
        self.T1 = np.zeros([self.ndocc, self.nvir])
        self.T2  = np.zeros([self.ndocc, self.ndocc, self.nvir, self.nvir])
        
        for i in CAS_holes:
            for a in CAS_particles:
                self.T1[i,a] = self.external_T1[i,j,a,b]
                for j in CAS_holes:
                    for b in CAS_particles:
                        self.T2[i,j,a,b] = self.external_T2[i,j,a,b]
        
        self.cc_energy()

        self.r1 = 1
        self.r2 = 1
            
        LIM = 10**(-CC_CONV)
        ite = 0
        
        while self.r2 > LIM or self.r1 > LIM:
            ite += 1
            if ite > CC_MAXITER:
                raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
            Eold = self.Ecc
            t = time.time()
            self.T1_T2_Update()
            self.cc_energy()
            dE = self.Ecc - Eold
            print('-'*50)
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {}".format(self.Ecc))
            print("Energy change:         {}".format(dE))
            print("T1 Residue:            {}".format(self.r1))
            print("T2 Residue:            {}".format(self.r2))
            print("Time required:         {}".format(time.time() - t))
            print('-'*50)
        
        print("\nCC Equations Converged!!!")
        print("Final CCSD Energy:     {:<5.10f}".format(self.Ecc + self.Escf))
        if psi4_compare: print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
        print("Total Computation time:        {}".format(time.time() - tinit))

        
        
