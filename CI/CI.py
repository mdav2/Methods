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

# Create a class of states. These will store strings with orbital ocupations and support operations such as
# annihilation, creation

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

   # Function to return another Bra object with an orbital annihilated 

    def annihilate(self, orb, spin):    # Spin = 0 => Alpha Spin = 1 => Beta
        if self.occ[spin, orb] == 0:
            return Bra(self.occ, phase = 0)
        else:
            new_occ = self.occ.copy()
            new_occ[spin, orb] = 0
            f = self.occ[spin][:orb].sum()
            if spin == 0:
                new_p = self.p * (-1)**(f)
            else:
                new_p = self.p * (-1)**(f+ self.occ[spin].sum())
            return Bra(new_occ, phase = new_p)

   # Function to return another Bra object with a new orbital created

    def create(self, orb, spin):    # Spin = 0 => Alpha Spin = 1 => Beta
        if self.occ[spin, orb] == 1:
            return Bra(self.occ, phase = 0)
        else:
            new_occ = self.occ.copy()
            new_occ[spin, orb] = 1
            f = self.occ[spin][:orb].sum()
            if spin == 0:
                new_p = self.p * (-1)**(f)
            else:
                new_p = self.p * (-1)**(f+ self.occ[spin].sum())
            return Bra(new_occ, phase = new_p)

# Compute the overlap of two bras

def overlap(bra1, bra2):
    if bra1 == bra2:
        return bra1.p*bra2.p
    else:
        return 0

# Function to produce elements of the 1e Hamiltonian matrix a given pair of bras

def Hone(bra1, bra2):
    r = range(len(bra1.occ[0]))
    out = 0
    for o1 in r:
        for o2 in r:
            hold1 = overlap(bra1.annihilate(o1,0), bra2.annihilate(o2,0))
            hold2 = overlap(bra1.annihilate(o1,1), bra2.annihilate(o2,1))
            out += hold1 + hold2
            if hold1+hold2 != 0:
                print('Non zero for {} and {} soma {}'.format(o1, o2, hold1+hold2))
    return out

if __name__ == '__main__':
    alphas = np.array([1, 1, 1, 0, 0, 0])
    betas  = np.array([1, 1, 0, 0, 0, 1])
    occ = np.array([alphas, betas])
    ref = Bra(occ)
    ref2 = ref.annihilate(5,1).create(2,1)
    print(ref)
    print(ref2)
    print(Hone(ref, ref2))

