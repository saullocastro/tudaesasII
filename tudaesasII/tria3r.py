import numpy as np
from numpy.linalg import norm

DOF = 5

class Tria3R(object):
    """Reissner-Mindlin plate element

    Formulated based on the first-order shear deformation theory for plates

    """
    __slots__ = ['n1', 'n2', 'n3', 'ABDE', 'A', 'h', 'rho',
            'scf13', 'scf23']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        self.n3 = None
        self.A = None
        self.ABDE = None
        self.rho = None
        self.scf13 = 5/6. # transverse shear correction factor XZ
        self.scf23 = 5/6. # transverse shear correction factor YZ

def update_K(tria, nid_pos, ncoords, K):
    """Update K according to a tria element

    Properties
    ----------
    tria : `.Tria3R` object
        The tria element being added to K and M
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    K : np.array
        Global stiffness matrix

    """
    pos1 = nid_pos[tria.n1]
    pos2 = nid_pos[tria.n2]
    pos3 = nid_pos[tria.n3]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    x3, y3 = ncoords[pos3]

    l12 = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    l23 = ((x2 - x3)**2 + (y2 - y3)**2)**0.5
    l31 = ((x3 - x1)**2 + (y3 - y1)**2)**0.5
    maxl = max(l12, l23, l31)
    A = (-x1 + x2)*(-y2 + y3)/2 + (x2 - x3)*(-y1 + y2)/2
    assert A > 0
    tria.A = A

    A11 = tria.ABDE[0, 0]
    A12 = tria.ABDE[0, 1]
    A16 = tria.ABDE[0, 2]
    A22 = tria.ABDE[1, 1]
    A26 = tria.ABDE[1, 2]
    A66 = tria.ABDE[2, 2]
    B11 = tria.ABDE[3, 0]
    B12 = tria.ABDE[3, 1]
    B16 = tria.ABDE[3, 2]
    B22 = tria.ABDE[4, 1]
    B26 = tria.ABDE[4, 2]
    B66 = tria.ABDE[5, 2]
    D11 = tria.ABDE[3, 3]
    D12 = tria.ABDE[3, 4]
    D16 = tria.ABDE[3, 5]
    D22 = tria.ABDE[4, 4]
    D26 = tria.ABDE[4, 5]
    D66 = tria.ABDE[5, 5]
    E44 = tria.ABDE[6, 6]
    E45 = tria.ABDE[6, 7]
    E55 = tria.ABDE[7, 7]

    rho = tria.rho
    h = tria.h

    N1x = (y2 - y3)/(2*A)
    N2x = (-y1 + y3)/(2*A)
    N3x = (y1 - y2)/(2*A)
    N1y = (-x2 + x3)/(2*A)
    N2y = (x1 - x3)/(2*A)
    N3y = (-x1 + x2)/(2*A)

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3

    E44 = E44*tria.scf23
    E45 = E45*min(tria.scf23, tria.scf13)
    E55 = E55*tria.scf13

    #NOTE strategy to prevent shear locking used in BFG elements imported here...
    alpha = 1.15
    factor = alpha*maxl**2/h**2
    E44 = 1 / (1 + factor) * E44
    # E45 = 1 / (1 + factor) * E45
    E55 = 1 / (1 + factor) * E55

    K[0+c1, 0+c1] += A*(A11*N1x**2 + 2*A16*N1x*N1y + A66*N1y**2)
    K[0+c1, 1+c1] += A*(A12*N1x*N1y + A16*N1x**2 + A26*N1y**2 + A66*N1x*N1y)
    K[0+c1, 3+c1] += A*(B11*N1x**2 + 2*B16*N1x*N1y + B66*N1y**2)
    K[0+c1, 4+c1] += A*(B12*N1x*N1y + B16*N1x**2 + B26*N1y**2 + B66*N1x*N1y)
    K[0+c1, 0+c2] += A*(A11*N1x*N2x + A16*N1x*N2y + A16*N1y*N2x + A66*N1y*N2y)
    K[0+c1, 1+c2] += A*(A12*N1x*N2y + A16*N1x*N2x + A26*N1y*N2y + A66*N1y*N2x)
    K[0+c1, 3+c2] += A*(B11*N1x*N2x + B16*N1x*N2y + B16*N1y*N2x + B66*N1y*N2y)
    K[0+c1, 4+c2] += A*(B12*N1x*N2y + B16*N1x*N2x + B26*N1y*N2y + B66*N1y*N2x)
    K[0+c1, 0+c3] += A*(A11*N1x*N3x + A16*N1x*N3y + A16*N1y*N3x + A66*N1y*N3y)
    K[0+c1, 1+c3] += A*(A12*N1x*N3y + A16*N1x*N3x + A26*N1y*N3y + A66*N1y*N3x)
    K[0+c1, 3+c3] += A*(B11*N1x*N3x + B16*N1x*N3y + B16*N1y*N3x + B66*N1y*N3y)
    K[0+c1, 4+c3] += A*(B12*N1x*N3y + B16*N1x*N3x + B26*N1y*N3y + B66*N1y*N3x)
    K[1+c1, 0+c1] += A*(A12*N1x*N1y + A16*N1x**2 + A26*N1y**2 + A66*N1x*N1y)
    K[1+c1, 1+c1] += A*(A22*N1y**2 + 2*A26*N1x*N1y + A66*N1x**2)
    K[1+c1, 3+c1] += A*(B12*N1x*N1y + B16*N1x**2 + B26*N1y**2 + B66*N1x*N1y)
    K[1+c1, 4+c1] += A*(B22*N1y**2 + 2*B26*N1x*N1y + B66*N1x**2)
    K[1+c1, 0+c2] += A*(A12*N1y*N2x + A16*N1x*N2x + A26*N1y*N2y + A66*N1x*N2y)
    K[1+c1, 1+c2] += A*(A22*N1y*N2y + A26*N1x*N2y + A26*N1y*N2x + A66*N1x*N2x)
    K[1+c1, 3+c2] += A*(B12*N1y*N2x + B16*N1x*N2x + B26*N1y*N2y + B66*N1x*N2y)
    K[1+c1, 4+c2] += A*(B22*N1y*N2y + B26*N1x*N2y + B26*N1y*N2x + B66*N1x*N2x)
    K[1+c1, 0+c3] += A*(A12*N1y*N3x + A16*N1x*N3x + A26*N1y*N3y + A66*N1x*N3y)
    K[1+c1, 1+c3] += A*(A22*N1y*N3y + A26*N1x*N3y + A26*N1y*N3x + A66*N1x*N3x)
    K[1+c1, 3+c3] += A*(B12*N1y*N3x + B16*N1x*N3x + B26*N1y*N3y + B66*N1x*N3y)
    K[1+c1, 4+c3] += A*(B22*N1y*N3y + B26*N1x*N3y + B26*N1y*N3x + B66*N1x*N3x)
    K[2+c1, 2+c1] += A*(E44*N1y**2 + 2*E45*N1x*N1y + E55*N1x**2)
    K[2+c1, 3+c1] += A*(E45*N1y + E55*N1x)/3
    K[2+c1, 4+c1] += A*(E44*N1y + E45*N1x)/3
    K[2+c1, 2+c2] += A*(E44*N1y*N2y + E45*N1x*N2y + E45*N1y*N2x + E55*N1x*N2x)
    K[2+c1, 3+c2] += A*(E45*N1y + E55*N1x)/3
    K[2+c1, 4+c2] += A*(E44*N1y + E45*N1x)/3
    K[2+c1, 2+c3] += A*(E44*N1y*N3y + E45*N1x*N3y + E45*N1y*N3x + E55*N1x*N3x)
    K[2+c1, 3+c3] += A*(E45*N1y + E55*N1x)/3
    K[2+c1, 4+c3] += A*(E44*N1y + E45*N1x)/3
    K[3+c1, 0+c1] += A*(B11*N1x**2 + 2*B16*N1x*N1y + B66*N1y**2)
    K[3+c1, 1+c1] += A*(B12*N1x*N1y + B16*N1x**2 + B26*N1y**2 + B66*N1x*N1y)
    K[3+c1, 2+c1] += A*(E45*N1y + E55*N1x)/3
    K[3+c1, 3+c1] += A*(6*D11*N1x**2 + 12*D16*N1x*N1y + 6*D66*N1y**2 + E55)/6
    K[3+c1, 4+c1] += A*(6*D12*N1x*N1y + 6*D16*N1x**2 + 6*D26*N1y**2 + 6*D66*N1x*N1y + E45)/6
    K[3+c1, 0+c2] += A*(B11*N1x*N2x + B16*N1x*N2y + B16*N1y*N2x + B66*N1y*N2y)
    K[3+c1, 1+c2] += A*(B12*N1x*N2y + B16*N1x*N2x + B26*N1y*N2y + B66*N1y*N2x)
    K[3+c1, 2+c2] += A*(E45*N2y + E55*N2x)/3
    K[3+c1, 3+c2] += A*(12*D11*N1x*N2x + 12*D16*N1x*N2y + 12*D16*N1y*N2x + 12*D66*N1y*N2y + E55)/12
    K[3+c1, 4+c2] += A*(12*D12*N1x*N2y + 12*D16*N1x*N2x + 12*D26*N1y*N2y + 12*D66*N1y*N2x + E45)/12
    K[3+c1, 0+c3] += A*(B11*N1x*N3x + B16*N1x*N3y + B16*N1y*N3x + B66*N1y*N3y)
    K[3+c1, 1+c3] += A*(B12*N1x*N3y + B16*N1x*N3x + B26*N1y*N3y + B66*N1y*N3x)
    K[3+c1, 2+c3] += A*(E45*N3y + E55*N3x)/3
    K[3+c1, 3+c3] += A*(12*D11*N1x*N3x + 12*D16*N1x*N3y + 12*D16*N1y*N3x + 12*D66*N1y*N3y + E55)/12
    K[3+c1, 4+c3] += A*(12*D12*N1x*N3y + 12*D16*N1x*N3x + 12*D26*N1y*N3y + 12*D66*N1y*N3x + E45)/12
    K[4+c1, 0+c1] += A*(B12*N1x*N1y + B16*N1x**2 + B26*N1y**2 + B66*N1x*N1y)
    K[4+c1, 1+c1] += A*(B22*N1y**2 + 2*B26*N1x*N1y + B66*N1x**2)
    K[4+c1, 2+c1] += A*(E44*N1y + E45*N1x)/3
    K[4+c1, 3+c1] += A*(6*D12*N1x*N1y + 6*D16*N1x**2 + 6*D26*N1y**2 + 6*D66*N1x*N1y + E45)/6
    K[4+c1, 4+c1] += A*(6*D22*N1y**2 + 12*D26*N1x*N1y + 6*D66*N1x**2 + E44)/6
    K[4+c1, 0+c2] += A*(B12*N1y*N2x + B16*N1x*N2x + B26*N1y*N2y + B66*N1x*N2y)
    K[4+c1, 1+c2] += A*(B22*N1y*N2y + B26*N1x*N2y + B26*N1y*N2x + B66*N1x*N2x)
    K[4+c1, 2+c2] += A*(E44*N2y + E45*N2x)/3
    K[4+c1, 3+c2] += A*(12*D12*N1y*N2x + 12*D16*N1x*N2x + 12*D26*N1y*N2y + 12*D66*N1x*N2y + E45)/12
    K[4+c1, 4+c2] += A*(12*D22*N1y*N2y + 12*D26*N1x*N2y + 12*D26*N1y*N2x + 12*D66*N1x*N2x + E44)/12
    K[4+c1, 0+c3] += A*(B12*N1y*N3x + B16*N1x*N3x + B26*N1y*N3y + B66*N1x*N3y)
    K[4+c1, 1+c3] += A*(B22*N1y*N3y + B26*N1x*N3y + B26*N1y*N3x + B66*N1x*N3x)
    K[4+c1, 2+c3] += A*(E44*N3y + E45*N3x)/3
    K[4+c1, 3+c3] += A*(12*D12*N1y*N3x + 12*D16*N1x*N3x + 12*D26*N1y*N3y + 12*D66*N1x*N3y + E45)/12
    K[4+c1, 4+c3] += A*(12*D22*N1y*N3y + 12*D26*N1x*N3y + 12*D26*N1y*N3x + 12*D66*N1x*N3x + E44)/12
    K[0+c2, 0+c1] += A*(A11*N1x*N2x + A16*N1x*N2y + A16*N1y*N2x + A66*N1y*N2y)
    K[0+c2, 1+c1] += A*(A12*N1y*N2x + A16*N1x*N2x + A26*N1y*N2y + A66*N1x*N2y)
    K[0+c2, 3+c1] += A*(B11*N1x*N2x + B16*N1x*N2y + B16*N1y*N2x + B66*N1y*N2y)
    K[0+c2, 4+c1] += A*(B12*N1y*N2x + B16*N1x*N2x + B26*N1y*N2y + B66*N1x*N2y)
    K[0+c2, 0+c2] += A*(A11*N2x**2 + 2*A16*N2x*N2y + A66*N2y**2)
    K[0+c2, 1+c2] += A*(A12*N2x*N2y + A16*N2x**2 + A26*N2y**2 + A66*N2x*N2y)
    K[0+c2, 3+c2] += A*(B11*N2x**2 + 2*B16*N2x*N2y + B66*N2y**2)
    K[0+c2, 4+c2] += A*(B12*N2x*N2y + B16*N2x**2 + B26*N2y**2 + B66*N2x*N2y)
    K[0+c2, 0+c3] += A*(A11*N2x*N3x + A16*N2x*N3y + A16*N2y*N3x + A66*N2y*N3y)
    K[0+c2, 1+c3] += A*(A12*N2x*N3y + A16*N2x*N3x + A26*N2y*N3y + A66*N2y*N3x)
    K[0+c2, 3+c3] += A*(B11*N2x*N3x + B16*N2x*N3y + B16*N2y*N3x + B66*N2y*N3y)
    K[0+c2, 4+c3] += A*(B12*N2x*N3y + B16*N2x*N3x + B26*N2y*N3y + B66*N2y*N3x)
    K[1+c2, 0+c1] += A*(A12*N1x*N2y + A16*N1x*N2x + A26*N1y*N2y + A66*N1y*N2x)
    K[1+c2, 1+c1] += A*(A22*N1y*N2y + A26*N1x*N2y + A26*N1y*N2x + A66*N1x*N2x)
    K[1+c2, 3+c1] += A*(B12*N1x*N2y + B16*N1x*N2x + B26*N1y*N2y + B66*N1y*N2x)
    K[1+c2, 4+c1] += A*(B22*N1y*N2y + B26*N1x*N2y + B26*N1y*N2x + B66*N1x*N2x)
    K[1+c2, 0+c2] += A*(A12*N2x*N2y + A16*N2x**2 + A26*N2y**2 + A66*N2x*N2y)
    K[1+c2, 1+c2] += A*(A22*N2y**2 + 2*A26*N2x*N2y + A66*N2x**2)
    K[1+c2, 3+c2] += A*(B12*N2x*N2y + B16*N2x**2 + B26*N2y**2 + B66*N2x*N2y)
    K[1+c2, 4+c2] += A*(B22*N2y**2 + 2*B26*N2x*N2y + B66*N2x**2)
    K[1+c2, 0+c3] += A*(A12*N2y*N3x + A16*N2x*N3x + A26*N2y*N3y + A66*N2x*N3y)
    K[1+c2, 1+c3] += A*(A22*N2y*N3y + A26*N2x*N3y + A26*N2y*N3x + A66*N2x*N3x)
    K[1+c2, 3+c3] += A*(B12*N2y*N3x + B16*N2x*N3x + B26*N2y*N3y + B66*N2x*N3y)
    K[1+c2, 4+c3] += A*(B22*N2y*N3y + B26*N2x*N3y + B26*N2y*N3x + B66*N2x*N3x)
    K[2+c2, 2+c1] += A*(E44*N1y*N2y + E45*N1x*N2y + E45*N1y*N2x + E55*N1x*N2x)
    K[2+c2, 3+c1] += A*(E45*N2y + E55*N2x)/3
    K[2+c2, 4+c1] += A*(E44*N2y + E45*N2x)/3
    K[2+c2, 2+c2] += A*(E44*N2y**2 + 2*E45*N2x*N2y + E55*N2x**2)
    K[2+c2, 3+c2] += A*(E45*N2y + E55*N2x)/3
    K[2+c2, 4+c2] += A*(E44*N2y + E45*N2x)/3
    K[2+c2, 2+c3] += A*(E44*N2y*N3y + E45*N2x*N3y + E45*N2y*N3x + E55*N2x*N3x)
    K[2+c2, 3+c3] += A*(E45*N2y + E55*N2x)/3
    K[2+c2, 4+c3] += A*(E44*N2y + E45*N2x)/3
    K[3+c2, 0+c1] += A*(B11*N1x*N2x + B16*N1x*N2y + B16*N1y*N2x + B66*N1y*N2y)
    K[3+c2, 1+c1] += A*(B12*N1y*N2x + B16*N1x*N2x + B26*N1y*N2y + B66*N1x*N2y)
    K[3+c2, 2+c1] += A*(E45*N1y + E55*N1x)/3
    K[3+c2, 3+c1] += A*(12*D11*N1x*N2x + 12*D16*N1x*N2y + 12*D16*N1y*N2x + 12*D66*N1y*N2y + E55)/12
    K[3+c2, 4+c1] += A*(12*D12*N1y*N2x + 12*D16*N1x*N2x + 12*D26*N1y*N2y + 12*D66*N1x*N2y + E45)/12
    K[3+c2, 0+c2] += A*(B11*N2x**2 + 2*B16*N2x*N2y + B66*N2y**2)
    K[3+c2, 1+c2] += A*(B12*N2x*N2y + B16*N2x**2 + B26*N2y**2 + B66*N2x*N2y)
    K[3+c2, 2+c2] += A*(E45*N2y + E55*N2x)/3
    K[3+c2, 3+c2] += A*(6*D11*N2x**2 + 12*D16*N2x*N2y + 6*D66*N2y**2 + E55)/6
    K[3+c2, 4+c2] += A*(6*D12*N2x*N2y + 6*D16*N2x**2 + 6*D26*N2y**2 + 6*D66*N2x*N2y + E45)/6
    K[3+c2, 0+c3] += A*(B11*N2x*N3x + B16*N2x*N3y + B16*N2y*N3x + B66*N2y*N3y)
    K[3+c2, 1+c3] += A*(B12*N2x*N3y + B16*N2x*N3x + B26*N2y*N3y + B66*N2y*N3x)
    K[3+c2, 2+c3] += A*(E45*N3y + E55*N3x)/3
    K[3+c2, 3+c3] += A*(12*D11*N2x*N3x + 12*D16*N2x*N3y + 12*D16*N2y*N3x + 12*D66*N2y*N3y + E55)/12
    K[3+c2, 4+c3] += A*(12*D12*N2x*N3y + 12*D16*N2x*N3x + 12*D26*N2y*N3y + 12*D66*N2y*N3x + E45)/12
    K[4+c2, 0+c1] += A*(B12*N1x*N2y + B16*N1x*N2x + B26*N1y*N2y + B66*N1y*N2x)
    K[4+c2, 1+c1] += A*(B22*N1y*N2y + B26*N1x*N2y + B26*N1y*N2x + B66*N1x*N2x)
    K[4+c2, 2+c1] += A*(E44*N1y + E45*N1x)/3
    K[4+c2, 3+c1] += A*(12*D12*N1x*N2y + 12*D16*N1x*N2x + 12*D26*N1y*N2y + 12*D66*N1y*N2x + E45)/12
    K[4+c2, 4+c1] += A*(12*D22*N1y*N2y + 12*D26*N1x*N2y + 12*D26*N1y*N2x + 12*D66*N1x*N2x + E44)/12
    K[4+c2, 0+c2] += A*(B12*N2x*N2y + B16*N2x**2 + B26*N2y**2 + B66*N2x*N2y)
    K[4+c2, 1+c2] += A*(B22*N2y**2 + 2*B26*N2x*N2y + B66*N2x**2)
    K[4+c2, 2+c2] += A*(E44*N2y + E45*N2x)/3
    K[4+c2, 3+c2] += A*(6*D12*N2x*N2y + 6*D16*N2x**2 + 6*D26*N2y**2 + 6*D66*N2x*N2y + E45)/6
    K[4+c2, 4+c2] += A*(6*D22*N2y**2 + 12*D26*N2x*N2y + 6*D66*N2x**2 + E44)/6
    K[4+c2, 0+c3] += A*(B12*N2y*N3x + B16*N2x*N3x + B26*N2y*N3y + B66*N2x*N3y)
    K[4+c2, 1+c3] += A*(B22*N2y*N3y + B26*N2x*N3y + B26*N2y*N3x + B66*N2x*N3x)
    K[4+c2, 2+c3] += A*(E44*N3y + E45*N3x)/3
    K[4+c2, 3+c3] += A*(12*D12*N2y*N3x + 12*D16*N2x*N3x + 12*D26*N2y*N3y + 12*D66*N2x*N3y + E45)/12
    K[4+c2, 4+c3] += A*(12*D22*N2y*N3y + 12*D26*N2x*N3y + 12*D26*N2y*N3x + 12*D66*N2x*N3x + E44)/12
    K[0+c3, 0+c1] += A*(A11*N1x*N3x + A16*N1x*N3y + A16*N1y*N3x + A66*N1y*N3y)
    K[0+c3, 1+c1] += A*(A12*N1y*N3x + A16*N1x*N3x + A26*N1y*N3y + A66*N1x*N3y)
    K[0+c3, 3+c1] += A*(B11*N1x*N3x + B16*N1x*N3y + B16*N1y*N3x + B66*N1y*N3y)
    K[0+c3, 4+c1] += A*(B12*N1y*N3x + B16*N1x*N3x + B26*N1y*N3y + B66*N1x*N3y)
    K[0+c3, 0+c2] += A*(A11*N2x*N3x + A16*N2x*N3y + A16*N2y*N3x + A66*N2y*N3y)
    K[0+c3, 1+c2] += A*(A12*N2y*N3x + A16*N2x*N3x + A26*N2y*N3y + A66*N2x*N3y)
    K[0+c3, 3+c2] += A*(B11*N2x*N3x + B16*N2x*N3y + B16*N2y*N3x + B66*N2y*N3y)
    K[0+c3, 4+c2] += A*(B12*N2y*N3x + B16*N2x*N3x + B26*N2y*N3y + B66*N2x*N3y)
    K[0+c3, 0+c3] += A*(A11*N3x**2 + 2*A16*N3x*N3y + A66*N3y**2)
    K[0+c3, 1+c3] += A*(A12*N3x*N3y + A16*N3x**2 + A26*N3y**2 + A66*N3x*N3y)
    K[0+c3, 3+c3] += A*(B11*N3x**2 + 2*B16*N3x*N3y + B66*N3y**2)
    K[0+c3, 4+c3] += A*(B12*N3x*N3y + B16*N3x**2 + B26*N3y**2 + B66*N3x*N3y)
    K[1+c3, 0+c1] += A*(A12*N1x*N3y + A16*N1x*N3x + A26*N1y*N3y + A66*N1y*N3x)
    K[1+c3, 1+c1] += A*(A22*N1y*N3y + A26*N1x*N3y + A26*N1y*N3x + A66*N1x*N3x)
    K[1+c3, 3+c1] += A*(B12*N1x*N3y + B16*N1x*N3x + B26*N1y*N3y + B66*N1y*N3x)
    K[1+c3, 4+c1] += A*(B22*N1y*N3y + B26*N1x*N3y + B26*N1y*N3x + B66*N1x*N3x)
    K[1+c3, 0+c2] += A*(A12*N2x*N3y + A16*N2x*N3x + A26*N2y*N3y + A66*N2y*N3x)
    K[1+c3, 1+c2] += A*(A22*N2y*N3y + A26*N2x*N3y + A26*N2y*N3x + A66*N2x*N3x)
    K[1+c3, 3+c2] += A*(B12*N2x*N3y + B16*N2x*N3x + B26*N2y*N3y + B66*N2y*N3x)
    K[1+c3, 4+c2] += A*(B22*N2y*N3y + B26*N2x*N3y + B26*N2y*N3x + B66*N2x*N3x)
    K[1+c3, 0+c3] += A*(A12*N3x*N3y + A16*N3x**2 + A26*N3y**2 + A66*N3x*N3y)
    K[1+c3, 1+c3] += A*(A22*N3y**2 + 2*A26*N3x*N3y + A66*N3x**2)
    K[1+c3, 3+c3] += A*(B12*N3x*N3y + B16*N3x**2 + B26*N3y**2 + B66*N3x*N3y)
    K[1+c3, 4+c3] += A*(B22*N3y**2 + 2*B26*N3x*N3y + B66*N3x**2)
    K[2+c3, 2+c1] += A*(E44*N1y*N3y + E45*N1x*N3y + E45*N1y*N3x + E55*N1x*N3x)
    K[2+c3, 3+c1] += A*(E45*N3y + E55*N3x)/3
    K[2+c3, 4+c1] += A*(E44*N3y + E45*N3x)/3
    K[2+c3, 2+c2] += A*(E44*N2y*N3y + E45*N2x*N3y + E45*N2y*N3x + E55*N2x*N3x)
    K[2+c3, 3+c2] += A*(E45*N3y + E55*N3x)/3
    K[2+c3, 4+c2] += A*(E44*N3y + E45*N3x)/3
    K[2+c3, 2+c3] += A*(E44*N3y**2 + 2*E45*N3x*N3y + E55*N3x**2)
    K[2+c3, 3+c3] += A*(E45*N3y + E55*N3x)/3
    K[2+c3, 4+c3] += A*(E44*N3y + E45*N3x)/3
    K[3+c3, 0+c1] += A*(B11*N1x*N3x + B16*N1x*N3y + B16*N1y*N3x + B66*N1y*N3y)
    K[3+c3, 1+c1] += A*(B12*N1y*N3x + B16*N1x*N3x + B26*N1y*N3y + B66*N1x*N3y)
    K[3+c3, 2+c1] += A*(E45*N1y + E55*N1x)/3
    K[3+c3, 3+c1] += A*(12*D11*N1x*N3x + 12*D16*N1x*N3y + 12*D16*N1y*N3x + 12*D66*N1y*N3y + E55)/12
    K[3+c3, 4+c1] += A*(12*D12*N1y*N3x + 12*D16*N1x*N3x + 12*D26*N1y*N3y + 12*D66*N1x*N3y + E45)/12
    K[3+c3, 0+c2] += A*(B11*N2x*N3x + B16*N2x*N3y + B16*N2y*N3x + B66*N2y*N3y)
    K[3+c3, 1+c2] += A*(B12*N2y*N3x + B16*N2x*N3x + B26*N2y*N3y + B66*N2x*N3y)
    K[3+c3, 2+c2] += A*(E45*N2y + E55*N2x)/3
    K[3+c3, 3+c2] += A*(12*D11*N2x*N3x + 12*D16*N2x*N3y + 12*D16*N2y*N3x + 12*D66*N2y*N3y + E55)/12
    K[3+c3, 4+c2] += A*(12*D12*N2y*N3x + 12*D16*N2x*N3x + 12*D26*N2y*N3y + 12*D66*N2x*N3y + E45)/12
    K[3+c3, 0+c3] += A*(B11*N3x**2 + 2*B16*N3x*N3y + B66*N3y**2)
    K[3+c3, 1+c3] += A*(B12*N3x*N3y + B16*N3x**2 + B26*N3y**2 + B66*N3x*N3y)
    K[3+c3, 2+c3] += A*(E45*N3y + E55*N3x)/3
    K[3+c3, 3+c3] += A*(6*D11*N3x**2 + 12*D16*N3x*N3y + 6*D66*N3y**2 + E55)/6
    K[3+c3, 4+c3] += A*(6*D12*N3x*N3y + 6*D16*N3x**2 + 6*D26*N3y**2 + 6*D66*N3x*N3y + E45)/6
    K[4+c3, 0+c1] += A*(B12*N1x*N3y + B16*N1x*N3x + B26*N1y*N3y + B66*N1y*N3x)
    K[4+c3, 1+c1] += A*(B22*N1y*N3y + B26*N1x*N3y + B26*N1y*N3x + B66*N1x*N3x)
    K[4+c3, 2+c1] += A*(E44*N1y + E45*N1x)/3
    K[4+c3, 3+c1] += A*(12*D12*N1x*N3y + 12*D16*N1x*N3x + 12*D26*N1y*N3y + 12*D66*N1y*N3x + E45)/12
    K[4+c3, 4+c1] += A*(12*D22*N1y*N3y + 12*D26*N1x*N3y + 12*D26*N1y*N3x + 12*D66*N1x*N3x + E44)/12
    K[4+c3, 0+c2] += A*(B12*N2x*N3y + B16*N2x*N3x + B26*N2y*N3y + B66*N2y*N3x)
    K[4+c3, 1+c2] += A*(B22*N2y*N3y + B26*N2x*N3y + B26*N2y*N3x + B66*N2x*N3x)
    K[4+c3, 2+c2] += A*(E44*N2y + E45*N2x)/3
    K[4+c3, 3+c2] += A*(12*D12*N2x*N3y + 12*D16*N2x*N3x + 12*D26*N2y*N3y + 12*D66*N2y*N3x + E45)/12
    K[4+c3, 4+c2] += A*(12*D22*N2y*N3y + 12*D26*N2x*N3y + 12*D26*N2y*N3x + 12*D66*N2x*N3x + E44)/12
    K[4+c3, 0+c3] += A*(B12*N3x*N3y + B16*N3x**2 + B26*N3y**2 + B66*N3x*N3y)
    K[4+c3, 1+c3] += A*(B22*N3y**2 + 2*B26*N3x*N3y + B66*N3x**2)
    K[4+c3, 2+c3] += A*(E44*N3y + E45*N3x)/3
    K[4+c3, 3+c3] += A*(6*D12*N3x*N3y + 6*D16*N3x**2 + 6*D26*N3y**2 + 6*D66*N3x*N3y + E45)/6
    K[4+c3, 4+c3] += A*(6*D22*N3y**2 + 12*D26*N3x*N3y + 6*D66*N3x**2 + E44)/6


def update_M(tria, nid_pos, ncoords, M):
    """Update M according to a tria element

    Properties
    ----------
    tria : `.Tria3R` object
        The tria element being added to M
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    M : np.array
        Global mass matrix

    """
    pos1 = nid_pos[tria.n1]
    pos2 = nid_pos[tria.n2]
    pos3 = nid_pos[tria.n3]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    x3, y3 = ncoords[pos3]

    A = (-x1 + x2)*(-y2 + y3)/2 + (x2 - x3)*(-y1 + y2)/2
    assert A > 0
    tria.A = A

    A11 = tria.ABDE[0, 0]
    A12 = tria.ABDE[0, 1]
    A16 = tria.ABDE[0, 2]
    A22 = tria.ABDE[1, 1]
    A26 = tria.ABDE[1, 2]
    A66 = tria.ABDE[2, 2]
    B11 = tria.ABDE[3, 0]
    B12 = tria.ABDE[3, 1]
    B16 = tria.ABDE[3, 2]
    B22 = tria.ABDE[4, 1]
    B26 = tria.ABDE[4, 2]
    B66 = tria.ABDE[5, 2]
    D11 = tria.ABDE[3, 3]
    D12 = tria.ABDE[3, 4]
    D16 = tria.ABDE[3, 5]
    D22 = tria.ABDE[4, 4]
    D26 = tria.ABDE[4, 5]
    D66 = tria.ABDE[5, 5]
    E44 = tria.ABDE[6, 6]
    E45 = tria.ABDE[6, 7]
    E55 = tria.ABDE[7, 7]

    rho = tria.rho
    h = tria.h

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3

    M[0+c1, 0+c1] += A*h*rho/6
    M[0+c1, 0+c2] += A*h*rho/12
    M[0+c1, 0+c3] += A*h*rho/12
    M[1+c1, 1+c1] += A*h*rho/6
    M[1+c1, 1+c2] += A*h*rho/12
    M[1+c1, 1+c3] += A*h*rho/12
    M[2+c1, 2+c1] += A*h*rho/6
    M[2+c1, 2+c2] += A*h*rho/12
    M[2+c1, 2+c3] += A*h*rho/12
    M[3+c1, 3+c1] += A*h**3*rho/72
    M[3+c1, 3+c2] += A*h**3*rho/144
    M[3+c1, 3+c3] += A*h**3*rho/144
    M[4+c1, 4+c1] += A*h**3*rho/72
    M[4+c1, 4+c2] += A*h**3*rho/144
    M[4+c1, 4+c3] += A*h**3*rho/144
    M[0+c2, 0+c1] += A*h*rho/12
    M[0+c2, 0+c2] += A*h*rho/6
    M[0+c2, 0+c3] += A*h*rho/12
    M[1+c2, 1+c1] += A*h*rho/12
    M[1+c2, 1+c2] += A*h*rho/6
    M[1+c2, 1+c3] += A*h*rho/12
    M[2+c2, 2+c1] += A*h*rho/12
    M[2+c2, 2+c2] += A*h*rho/6
    M[2+c2, 2+c3] += A*h*rho/12
    M[3+c2, 3+c1] += A*h**3*rho/144
    M[3+c2, 3+c2] += A*h**3*rho/72
    M[3+c2, 3+c3] += A*h**3*rho/144
    M[4+c2, 4+c1] += A*h**3*rho/144
    M[4+c2, 4+c2] += A*h**3*rho/72
    M[4+c2, 4+c3] += A*h**3*rho/144
    M[0+c3, 0+c1] += A*h*rho/12
    M[0+c3, 0+c2] += A*h*rho/12
    M[0+c3, 0+c3] += A*h*rho/6
    M[1+c3, 1+c1] += A*h*rho/12
    M[1+c3, 1+c2] += A*h*rho/12
    M[1+c3, 1+c3] += A*h*rho/6
    M[2+c3, 2+c1] += A*h*rho/12
    M[2+c3, 2+c2] += A*h*rho/12
    M[2+c3, 2+c3] += A*h*rho/6
    M[3+c3, 3+c1] += A*h**3*rho/144
    M[3+c3, 3+c2] += A*h**3*rho/144
    M[3+c3, 3+c3] += A*h**3*rho/72
    M[4+c3, 4+c1] += A*h**3*rho/144
    M[4+c3, 4+c2] += A*h**3*rho/144
    M[4+c3, 4+c3] += A*h**3*rho/72

