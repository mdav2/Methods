import numpy as np

# This module allows you to use annihilation and creation operators
# on strings that represents states in the fock space

# Create a class of states. These will store strings with orbital ocupations and support operations such as
# annihilation, creation, and overlap

class Bra:

    def __init__(self, occupancy, phase=1):
        self.occ = occupancy
        self.p = phase

    def __str__(self):
        out = 'Phase: {} \n'.format(self.p)
        out += 'Alpha:'
        for a in self.occ[0]:
            out += ' {}'.format(a)
        out += '\n'    
        out += 'Beta: '
        for b in self.occ[1]:
            out += ' {}'.format(b)
        return out

    def __eq__(self, other):
        return np.array_equal(self.occ, other.occ)

    def __int__(self):
        return int(self.occ[0].sum() + self.occ[1].sum())
        
    def __sub__(self, other):
        a = abs(self.occ[0] - other.occ[0])
        b = abs(self.occ[1] - other.occ[1])
        return Bra([a, b])

   # Function to return another Bra object with an orbital annihilated 

    def an(self, orb, spin):    # Spin = 0 => Alpha Spin = 1 => Beta
        if self.occ[spin, orb] == 0:
            return Bra(self.occ, phase = 0)
        else:
            new_occ = self.occ.copy()
            new_occ[spin, orb] = 0
            f = self.occ[spin][:orb].sum()
            if spin == 0:
                new_p = self.p * (-1)**(f)
            else:
                new_p = self.p * (-1)**(f+ self.occ[0].sum())
            return Bra(new_occ, phase = new_p)

   # Function to return another Bra object with a new orbital created

    def cr(self, orb, spin):    # Spin = 0 => Alpha Spin = 1 => Beta
        if self.occ[spin, orb] == 1:
            return Bra(self.occ, phase = 0)
        else:
            new_occ = self.occ.copy()
            new_occ[spin, orb] = 1
            f = self.occ[spin][:orb].sum()
            if spin == 0:
                new_p = self.p * (-1)**(f)
            else:
                new_p = self.p * (-1)**(f+ self.occ[0].sum())
            return Bra(new_occ, phase = new_p)
    
# Compute the overlap with another bra

def overlap(bra1, bra2):
    if bra1 == bra2:
        return bra1.p*bra2.p
    else:
        return 0

