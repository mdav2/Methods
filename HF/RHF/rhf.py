import psi4
import numpy as np
import scipy.linalg as la

# Emojis. Very important stuff

viva = b'\xF0\x9F\x8E\x89'.decode('utf-8')
eyes = b'\xF0\x9F\x91\x80'.decode('utf-8')
cycle = b'\xF0\x9F\x94\x83'.decode('utf-8')
crying = b'\xF0\x9F\x98\xAD'.decode('utf-8')
pleft = b'\xF0\x9F\x91\x88'.decode('utf-8')

# Auxiliar functions, useful for debugging and such


# Clean up numerical zeros

def chop(number):
    if abs(number) < 1e-12:
        return 0
    else:
        return number

# Print a pretty matrix

def pretty(inp):
    Mat = inp.tolist()
    out = ''
    for row in Mat:
        for x in row:
            out += ' {:^ 10.7f}'.format(chop(x))
        out += '\n'
    return out

class RHF:
    """
    Restricted Hartree-Fock class for obtaining the restricted Hartree-Fock
    energy
    """

    def __init__(self, mol, mints):
        """
        Initialize the rhf
        :param mol: a Psi4 molecule object
        :param mints: a molecular integrals object (from MintsHelper)
        """
        self.mol = mol
        self.mints = mints

        self.V_nuc = mol.nuclear_repulsion_energy()
        self.T = np.matrix(mints.ao_kinetic())
        self.S = np.matrix(mints.ao_overlap())
        self.V = np.matrix(mints.ao_potential())

        # Change g to physicists notation
        self.g = np.array(mints.ao_eri()).swapaxes(1,2)

        # Determine the number of electrons and the number of doubly occupied orbitals

        self.nelec = -mol.molecular_charge()
        for A in range(mol.natom()):
            self.nelec += int(mol.Z(A))
        if mol.multiplicity() != 1 or self.nelec % 2:
            raise Exception("This code only allows closed-shell molecules")
        self.ndocc = int(self.nelec / 2)
        self.maxiter = psi4.core.get_global_option('MAXITER')
        self.e_convergence = psi4.core.get_global_option('E_CONVERGENCE')
        self.nbf = mints.basisset().nbf()
        # vu is the matrix of 2 electrons integrals [g(ijkl)] times the density matrix [D(jl)] contracted into vu(ik). Since the guess is D = 0, vu starts as 0
        self.vu = np.matrix(np.zeros((self.nbf, self.nbf)))
    
    def compute_energy(self, out = 'None'):
        """
        Compute the rhf energy
        :return: energy
        """
        X = np.matrix(la.inv(la.sqrtm(self.S)))
        D = np.matrix(np.zeros((self.nbf, self.nbf)))
        h = self.T + self.V
        E0 = 0
 
        # Just printing pretty stuff
        psi4.core.print_out('-'*50 + '\n')
        psi4.core.print_out('{:^50s}'.format('Beginning SCF iterations ' + cycle) + '\n')
        psi4.core.print_out('-'*50 + '\n')
        psi4.core.print_out('{:^10s}   {:^15s}   {:^15s}\n'.format('Iteration', 'Energy', 'Energy Diff'))

        for count in range(self.maxiter):
            F = h + self.vu
            Ft = X * F * X
            e, Ct = la.eigh(Ft)
            C = X * np.matrix(Ct)
            DOC = np.matrix(C[:,:self.ndocc])
            D = DOC*DOC.T
            G = 2*self.g - self.g.swapaxes(2,3)
            self.vu = np.einsum('upvq,pq->uv', G, D) 
            E1 = np.sum((2 * np.array(h) + np.array(self.vu))*np.array(D.T)) + self.V_nuc
            psi4.core.print_out('{:^10d}   {:<15.10f}   {:>15.10f}\n'.format(count, E1, E1-E0))
            if abs(E1 - E0) < self.e_convergence:
                psi4.core.print_out('\nSCF has converged!! ' + viva + '\n')
                psi4.core.print_out('\nFinal RHF Energy: {:<15.10f} '.format(E1) + pleft + '\n')
                break
            else:
                E0 = E1
        else:
            psi4.core.print_out('\n SCF did not converge ' + crying + '\n')

        if out == 'Density':
            print(pretty(D))
