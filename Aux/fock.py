import numpy as np
from math import log2

# This module allows you to use annihilation and creation operators
# on strings that represents states in the fock space

# Create a class of states. These will store strings with orbital ocupations and support operations such as
# annihilation, creation, and overlap

class Det:

    def __init__(self, a='', b=''):
        self.alpha = int(a[::-1], 2)
        self.beta  = int(b[::-1], 2)
        self.nmo = len(a)

# Printing a bra will return its alpha and beta strings

    def __str__(self):
        out = 'Alpha: ' + np.binary_repr(self.alpha, width=self.nmo)[::-1]
        out += '\n'    
        out += 'Beta:  ' + np.binary_repr(self.beta, width=self.nmo)[::-1]
        return out

# Return the alpha string as a list

    def alpha_list(self):
        return np.array([int(x) for x in list(np.binary_repr(self.alpha, width=self.nmo))])[::-1]

# Return the beta string as a list

    def beta_list(self):
        return np.array([int(x) for x in list(np.binary_repr(self.beta, width=self.nmo))])[::-1]

# Return alpha string as it is stored

    def alpha_string(self):
        return np.binary_repr(self.alpha, width=self.nmo)

# Return beta string as it is stored

    def beta_string(self):
        return np.binary_repr(self.beta, width=self.nmo)

# Return alpha and beta together in one string

    def alpha_beta_string(self):
        return self.beta_string() + self.alpha_string()

# When two bras are compared they are considered equal if their strings are the same (occupancies). Phases are not compared

    def __eq__(self, other):
        if self.alpha == other.alpha and self.beta == other.beta:
            return True
        else:
            return False

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
        return Det(a = self.alpha_string()[::-1], b = self.beta_string()[::-1])

    def rmv_alpha(self, orb):
        self.alpha ^= (1 << orb)

    def rmv_beta(self, orb):
        self.beta ^= (1 << orb)

    def add_alpha(self, orb):
        self.alpha |= (1 <<  orb)

    def add_beta(self, orb):
        self.beta |= (1 << orb)

    def sign_dif2(self, another):
        det1 = int(self.alpha_beta_string(),2)
        det2 = int(another.alpha_beta_string(),2)
        x1 = det1 & (det1 ^ det2)
        x2 = det2 & (det1 ^ det2)
        l = min(x1,x2)
        u = max(x1,x2)
        p = 0
        while l < u:
            u = u >> 1
            if u & (det1 & det2):
                p += 1
        return (-1)**p
        det1 = np.array(self.alpha_list() + self.beta_list())
        det2 = np.array(another.alpha_list() + another.beta_list())
        x = det1 - det2
            
    def sign_dif4(self, another):
        det1 = int(self.alpha_beta_string(),2)
        det2 = int(another.alpha_beta_string(),2)
        x1 = det1 & (det1 ^ det2)
        x2 = det2 & (det1 ^ det2)
        p = 0
        i = 1
        px1 = []
        px2 = []
        while i < max(x1,x2):
            if i & (det1 & det2):
                    p += 1
            if i & x1:
                px1.append(p)
            if i & x2:
                px2.append(p)
            i = i << 1
        p = abs(px1[0]-px2[0]) + abs(px1[1]-px2[1])
        return (-1)**p
        
    def phase(self, another):
        if self - another == 2:
            return self.sign_dif2(another)        
        if self - another == 4:
            return self.sign_dif4(another)
        else:
            return 0

# Compute the overlap with another bra

def overlap(bra1, bra2):
    if bra1 == bra2:
        return bra1.p*bra2.p
    else:
        return 0

