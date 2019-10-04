import numpy as np
from math import log2

# This module allows you to use annihilation and creation operators
# on strings that represents states in the fock space

# Create a class of states. These will store strings with orbital ocupations and support operations such as
# annihilation, creation, and overlap

class Det:

    def __init__(self, alpha='', beta=''):
        self.alpha = int(alpha, 2)
        self.beta  = int(beta, 2)
        self.nmo = len(alpha)

# Printing a bra will return its alpha and beta strings

    def __str__(self):
        out = 'Phase: {} \n'.format(self.p)
        out += 'Alpha: ' + np.binary_repr(self.alpha, width=self.nmo)
        out += '\n'    
        out += 'Beta:  ' + np.binary_repr(self.beta, width=self.nmo)
        return out

# Return the alpha string as a list

    def alpha_list(self):
        return np.array([int(x) for x in list(np.binary_repr(self.alpha, width=self.nmo))])

# Return the beta string as a list

    def beta_list(self):
        return np.array([int(x) for x in list(np.binary_repr(self.beta, width=self.nmo))])

# Return alpha string

    def alpha_string(self):
        return np.binary_repr(self.alpha, width=self.nmo)

# Return beta string

    def beta_string(self):
        return np.binary_repr(self.beta, width=self.nmo)

# Return alpha and beta together in one string

    def alpha_beta_string(self):
        return self.alpha_string() + self.beta_string()

# When two bras are compared they are considered equal if their strings are the same (occupancies). Phases are not compared

    def __eq__(self, other):
        if self.alpha == other.alpha and self.beta == other.beta:
            return True
        else:
            return False

# Trying to call integer on a bra will return the number of occupied orbitals
#
#    def NumOcOrb(self):
#        return int(self.occ[0].sum() + self.occ[1].sum())
        
# Subtraction of two Dets returns the number of different occupied orbitals between the two

    def __sub__(self, other,v=False):
        a = bin(self.alpha ^ other.alpha).count("1")
        b = bin(self.beta ^ other.beta).count("1")
        return a + b

# Function: Given other bra, return a list of orbitals (e.g. [ [1,0], [3,1], etc]) that are found
# in the first bra, but not in the second

    def Exclusive(self, other):
        out = []
        a = self.alpha_list() - other.alpha_list()
        for i in np.where(a == 1)[0]:
            out.append([i, 0])
        b = self.beta_list() - other.beta_list()
        for i in np.where(b == 1)[0]:
            out.append([i, 1])
        return out

# Function to return another Bra object with an orbital annihilated keeping track of the sign.

    def copy(self):
        return Det(alpha = self.alpha, beta = self.beta, phase=self.p)

    def rmv_alpha(self, orb):
        self.alpha ^= (1 << (self.nmo -1 - orb))

    def rmv_beta(self, orb):
        self.beta ^= (1 << (self.nmo -1 - orb))

    def add_alpha(self, orb):
        self.alpha |= (1 << (self.nmo -1 - orb))

    def add_beta(self, orb):
        self.beta |= (1 << (self.nmo -1 - orb))

#    def sign_dif2(self, another):
#        if self.alpha == another.alpha:
#            x = self.beta    & (self.beta ^ another.beta)
#            y = another.beta & (self.beta ^ another.beta)
#            p = log2(max([x,y]/min[x,y]))
#            return (-1)**p
#        else:
#            x = self.alpha    & (self.alpha ^ another.alpha)
#            y = another.alpha & (self.alpha ^ another.alpha)
#            p = log2(max([x,y]/min[x,y]))
#            return (-1)**p

#    def sign_dif2(self, another):
#        det1 = int(self.alpha_beta_string(),2)
#        det2 = int(another.alpha_beta_string(),2)
#        x = det1 & (det1 ^ det2)
#        y = det2 & (det1 ^ det2)
#        p = log2(max([x,y])/min([x,y]))
#        return (-1)**p

    def sign_dif2(self, another):
        det1 = int(self.alpha_beta_string(),2)
        det2 = int(another.alpha_beta_string(),2)
        x = det1 & (det1 ^ det2)
        y = det2 & (det1 ^ det2)
        p = log2(max([x,y])/min([x,y]))
        return (-1)**p
            
    def sign_dif4(self, another):
        det1 = int(self.alpha_beta_string(),2)
        det2 = int(another.alpha_beta_string(),2)
        x = det1 & (det1 ^ det2)
        y = det2 & (det1 ^ det2)
        p = 0
        for i,v in enumerate(np.binary_repr(x)):
            if v == '1':
                p += i
        for i,v in enumerate(np.binary_repr(y)):
            if v == '1':
                p -= i
        return (-1)**p
    
    def phase(self, another):
        if self - another == 2:
            return self.sign_dif4(another)        
        if self - another == 4:
            return self.sign_dif4(another)
        else:
            return 1

    def an(self, orb, spin):    # Spin = 0 => Alpha Spin = 1 => Beta
        if self.p == 0:
            return Det(phase=0)
        elif spin == 0:
            occ = self.alpha_string()
            if occ[orb] == '0':
                return Det(phase=0)
            elif occ[orb] == '1':
                new_occ = occ[:orb] + '0' + occ[orb+1:]
                f = occ[:orb].count("1")
                new_p = self.p * (-1)**(f)
                return Det(alpha=new_occ,beta=self.alpha_string(), phase = new_p)
        elif spin == 1:
            occ = self.beta_string()
            if occ[orb] == '0':
                return Det(phase=0)
            elif occ[orb] == '1':
                new_occ = occ[:orb] + '0' + occ[orb+1:]
                f = occ[:orb].count("1")
                new_p = self.p * (-1)**(f + self.alpha_string().count("1"))
                return Det(alpha=self.alpha_string(),beta=new_occ, phase = new_p)

# Function to return another Bra object with a new orbital created, keeping track of the sign

    def cr(self, orb, spin):    # Spin = 0 => Alpha Spin = 1 => Beta
        if spin == 0:
            occ = self.alpha_string()
            if occ[orb] == '1':
                pass
        if occ[spin, orb] == 1:
            return Det(occ, phase = 0)
        else:
            new_occ = occ.copy()
            new_occ[spin, orb] = 1
            f = occ[spin][:orb].sum()
            if spin == 0:
                new_p = self.p * (-1)**(f)
            else:
                new_p = self.p * (-1)**(f+ occ[0].sum())
            return Det(new_occ, phase = new_p)
    
# Compute the overlap with another bra

def overlap(bra1, bra2):
    if bra1 == bra2:
        return bra1.p*bra2.p
    else:
        return 0

