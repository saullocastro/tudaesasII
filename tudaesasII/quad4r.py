import numpy as np

DOF = 5

class Quad4R(object):
    """Reissner-Mindlin plate element with reduced integration

    Formulated based on the first-order shear deformation theory for plates

    Reduced integration is achieved by having only 1 integration point at the
    center, while integrating the stiffness matrix. This removes shear locking.

    NOTE: No hour-glass control is implemented, so be careful, specially in
    modal analysis, because some spurious modes will appear first.

    """
    __slots__ = ['n1', 'n2', 'n3', 'n4', 'ABDE', 'h', 'rho',
            'scf13', 'scf23']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        self.n3 = None
        self.n4 = None
        self.ABDE = None
        self.rho = None
        self.scf13 = 5/6. # transverse shear correction factor XZ
        self.scf23 = 5/6. # transverse shear correction factor YZ

def update_K(quad, nid_pos, ncoords, K):
    """Update global K with Ke from a quad element

    Properties
    ----------
    quad : `.Quad`object
        The quad element being added to K
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    K : np.array
        Global stiffness matrix

    """
    pos1 = nid_pos[quad.n1]
    pos2 = nid_pos[quad.n2]
    pos3 = nid_pos[quad.n3]
    pos4 = nid_pos[quad.n4]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    x3, y3 = ncoords[pos3]
    x4, y4 = ncoords[pos4]

    A11 = quad.ABDE[0, 0]
    A12 = quad.ABDE[0, 1]
    A16 = quad.ABDE[0, 2]
    A22 = quad.ABDE[1, 1]
    A26 = quad.ABDE[1, 2]
    A66 = quad.ABDE[2, 2]
    B11 = quad.ABDE[3, 0]
    B12 = quad.ABDE[3, 1]
    B16 = quad.ABDE[3, 2]
    B22 = quad.ABDE[4, 1]
    B26 = quad.ABDE[4, 2]
    B66 = quad.ABDE[5, 2]
    D11 = quad.ABDE[3, 3]
    D12 = quad.ABDE[3, 4]
    D16 = quad.ABDE[3, 5]
    D22 = quad.ABDE[4, 4]
    D26 = quad.ABDE[4, 5]
    D66 = quad.ABDE[5, 5]
    E44 = quad.ABDE[6, 6]
    E45 = quad.ABDE[6, 7]
    E55 = quad.ABDE[7, 7]

    # applying shear correction factors
    E44 = E44*quad.scf23
    E45 = E45*min(quad.scf23, quad.scf13)
    E55 = E55*quad.scf13

    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    x3, y3 = ncoords[pos3]
    x4, y4 = ncoords[pos4]

    A = np.cross([x2 - x1, y2 - x1], [x4 - x1, y4 - y1])

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3
    c4 = DOF*pos4

    h = quad.h

    if quad.n1 == 0:
        print("NOTE reduced integration")

    #NOTE reduced integration to remove shear locking
    xi = eta = 0
    wi = wj = 2.

    wij = wi*wj
    detJ = (-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16 - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16
    j11 = 4*(2*y1 - 2*y4 - (xi + 1)*(y1 - y2) - (xi + 1)*(y3 - y4))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
    j12 = 4*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
    j21 = 4*(-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
    j22 = 4*(2*x1 - 2*x2 - (eta + 1)*(x1 - x2 + x3 - x4))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
    N1 = eta*xi/4 - eta/4 - xi/4 + 1/4
    N2 = -eta*xi/4 - eta/4 + xi/4 + 1/4
    N3 = eta*xi/4 + eta/4 + xi/4 + 1/4
    N4 = -eta*xi/4 + eta/4 - xi/4 + 1/4
    N1x = j11*(eta - 1)/4 + j12*(xi - 1)/4
    N2x = -eta*j11/4 + j11/4 - j12*xi/4 - j12/4
    N3x = j11*(eta + 1)/4 + j12*(xi + 1)/4
    N4x = -eta*j11/4 - j11/4 - j12*xi/4 + j12/4
    N1y = j21*(eta - 1)/4 + j22*(xi - 1)/4
    N2y = -eta*j21/4 + j21/4 - j22*xi/4 - j22/4
    N3y = j21*(eta + 1)/4 + j22*(xi + 1)/4
    N4y = -eta*j21/4 - j21/4 - j22*xi/4 + j22/4

    K[0+c1, 0+c1] += N1x*(A11*N1x*detJ*wij + A16*N1y*detJ*wij) + N1y*(A16*N1x*detJ*wij + A66*N1y*detJ*wij)
    K[0+c1, 1+c1] += N1x*(A16*N1x*detJ*wij + A66*N1y*detJ*wij) + N1y*(A12*N1x*detJ*wij + A26*N1y*detJ*wij)
    K[0+c1, 3+c1] += N1x*(B11*N1x*detJ*wij + B16*N1y*detJ*wij) + N1y*(B16*N1x*detJ*wij + B66*N1y*detJ*wij)
    K[0+c1, 4+c1] += N1x*(B16*N1x*detJ*wij + B66*N1y*detJ*wij) + N1y*(B12*N1x*detJ*wij + B26*N1y*detJ*wij)
    K[0+c1, 0+c2] += N2x*(A11*N1x*detJ*wij + A16*N1y*detJ*wij) + N2y*(A16*N1x*detJ*wij + A66*N1y*detJ*wij)
    K[0+c1, 1+c2] += N2x*(A16*N1x*detJ*wij + A66*N1y*detJ*wij) + N2y*(A12*N1x*detJ*wij + A26*N1y*detJ*wij)
    K[0+c1, 3+c2] += N2x*(B11*N1x*detJ*wij + B16*N1y*detJ*wij) + N2y*(B16*N1x*detJ*wij + B66*N1y*detJ*wij)
    K[0+c1, 4+c2] += N2x*(B16*N1x*detJ*wij + B66*N1y*detJ*wij) + N2y*(B12*N1x*detJ*wij + B26*N1y*detJ*wij)
    K[0+c1, 0+c3] += N3x*(A11*N1x*detJ*wij + A16*N1y*detJ*wij) + N3y*(A16*N1x*detJ*wij + A66*N1y*detJ*wij)
    K[0+c1, 1+c3] += N3x*(A16*N1x*detJ*wij + A66*N1y*detJ*wij) + N3y*(A12*N1x*detJ*wij + A26*N1y*detJ*wij)
    K[0+c1, 3+c3] += N3x*(B11*N1x*detJ*wij + B16*N1y*detJ*wij) + N3y*(B16*N1x*detJ*wij + B66*N1y*detJ*wij)
    K[0+c1, 4+c3] += N3x*(B16*N1x*detJ*wij + B66*N1y*detJ*wij) + N3y*(B12*N1x*detJ*wij + B26*N1y*detJ*wij)
    K[0+c1, 0+c4] += N4x*(A11*N1x*detJ*wij + A16*N1y*detJ*wij) + N4y*(A16*N1x*detJ*wij + A66*N1y*detJ*wij)
    K[0+c1, 1+c4] += N4x*(A16*N1x*detJ*wij + A66*N1y*detJ*wij) + N4y*(A12*N1x*detJ*wij + A26*N1y*detJ*wij)
    K[0+c1, 3+c4] += N4x*(B11*N1x*detJ*wij + B16*N1y*detJ*wij) + N4y*(B16*N1x*detJ*wij + B66*N1y*detJ*wij)
    K[0+c1, 4+c4] += N4x*(B16*N1x*detJ*wij + B66*N1y*detJ*wij) + N4y*(B12*N1x*detJ*wij + B26*N1y*detJ*wij)
    K[1+c1, 0+c1] += N1x*(A12*N1y*detJ*wij + A16*N1x*detJ*wij) + N1y*(A26*N1y*detJ*wij + A66*N1x*detJ*wij)
    K[1+c1, 1+c1] += N1x*(A26*N1y*detJ*wij + A66*N1x*detJ*wij) + N1y*(A22*N1y*detJ*wij + A26*N1x*detJ*wij)
    K[1+c1, 3+c1] += N1x*(B12*N1y*detJ*wij + B16*N1x*detJ*wij) + N1y*(B26*N1y*detJ*wij + B66*N1x*detJ*wij)
    K[1+c1, 4+c1] += N1x*(B26*N1y*detJ*wij + B66*N1x*detJ*wij) + N1y*(B22*N1y*detJ*wij + B26*N1x*detJ*wij)
    K[1+c1, 0+c2] += N2x*(A12*N1y*detJ*wij + A16*N1x*detJ*wij) + N2y*(A26*N1y*detJ*wij + A66*N1x*detJ*wij)
    K[1+c1, 1+c2] += N2x*(A26*N1y*detJ*wij + A66*N1x*detJ*wij) + N2y*(A22*N1y*detJ*wij + A26*N1x*detJ*wij)
    K[1+c1, 3+c2] += N2x*(B12*N1y*detJ*wij + B16*N1x*detJ*wij) + N2y*(B26*N1y*detJ*wij + B66*N1x*detJ*wij)
    K[1+c1, 4+c2] += N2x*(B26*N1y*detJ*wij + B66*N1x*detJ*wij) + N2y*(B22*N1y*detJ*wij + B26*N1x*detJ*wij)
    K[1+c1, 0+c3] += N3x*(A12*N1y*detJ*wij + A16*N1x*detJ*wij) + N3y*(A26*N1y*detJ*wij + A66*N1x*detJ*wij)
    K[1+c1, 1+c3] += N3x*(A26*N1y*detJ*wij + A66*N1x*detJ*wij) + N3y*(A22*N1y*detJ*wij + A26*N1x*detJ*wij)
    K[1+c1, 3+c3] += N3x*(B12*N1y*detJ*wij + B16*N1x*detJ*wij) + N3y*(B26*N1y*detJ*wij + B66*N1x*detJ*wij)
    K[1+c1, 4+c3] += N3x*(B26*N1y*detJ*wij + B66*N1x*detJ*wij) + N3y*(B22*N1y*detJ*wij + B26*N1x*detJ*wij)
    K[1+c1, 0+c4] += N4x*(A12*N1y*detJ*wij + A16*N1x*detJ*wij) + N4y*(A26*N1y*detJ*wij + A66*N1x*detJ*wij)
    K[1+c1, 1+c4] += N4x*(A26*N1y*detJ*wij + A66*N1x*detJ*wij) + N4y*(A22*N1y*detJ*wij + A26*N1x*detJ*wij)
    K[1+c1, 3+c4] += N4x*(B12*N1y*detJ*wij + B16*N1x*detJ*wij) + N4y*(B26*N1y*detJ*wij + B66*N1x*detJ*wij)
    K[1+c1, 4+c4] += N4x*(B26*N1y*detJ*wij + B66*N1x*detJ*wij) + N4y*(B22*N1y*detJ*wij + B26*N1x*detJ*wij)
    K[2+c1, 2+c1] += N1x*(E45*N1y*detJ*wij + E55*N1x*detJ*wij) + N1y*(E44*N1y*detJ*wij + E45*N1x*detJ*wij)
    K[2+c1, 3+c1] += N1*(E45*N1y*detJ*wij + E55*N1x*detJ*wij)
    K[2+c1, 4+c1] += N1*(E44*N1y*detJ*wij + E45*N1x*detJ*wij)
    K[2+c1, 2+c2] += N2x*(E45*N1y*detJ*wij + E55*N1x*detJ*wij) + N2y*(E44*N1y*detJ*wij + E45*N1x*detJ*wij)
    K[2+c1, 3+c2] += N2*(E45*N1y*detJ*wij + E55*N1x*detJ*wij)
    K[2+c1, 4+c2] += N2*(E44*N1y*detJ*wij + E45*N1x*detJ*wij)
    K[2+c1, 2+c3] += N3x*(E45*N1y*detJ*wij + E55*N1x*detJ*wij) + N3y*(E44*N1y*detJ*wij + E45*N1x*detJ*wij)
    K[2+c1, 3+c3] += N3*(E45*N1y*detJ*wij + E55*N1x*detJ*wij)
    K[2+c1, 4+c3] += N3*(E44*N1y*detJ*wij + E45*N1x*detJ*wij)
    K[2+c1, 2+c4] += N4x*(E45*N1y*detJ*wij + E55*N1x*detJ*wij) + N4y*(E44*N1y*detJ*wij + E45*N1x*detJ*wij)
    K[2+c1, 3+c4] += N4*(E45*N1y*detJ*wij + E55*N1x*detJ*wij)
    K[2+c1, 4+c4] += N4*(E44*N1y*detJ*wij + E45*N1x*detJ*wij)
    K[3+c1, 0+c1] += N1x*(B11*N1x*detJ*wij + B16*N1y*detJ*wij) + N1y*(B16*N1x*detJ*wij + B66*N1y*detJ*wij)
    K[3+c1, 1+c1] += N1x*(B16*N1x*detJ*wij + B66*N1y*detJ*wij) + N1y*(B12*N1x*detJ*wij + B26*N1y*detJ*wij)
    K[3+c1, 2+c1] += E45*N1*N1y*detJ*wij + E55*N1*N1x*detJ*wij
    K[3+c1, 3+c1] += E55*N1**2*detJ*wij + N1x*(D11*N1x*detJ*wij + D16*N1y*detJ*wij) + N1y*(D16*N1x*detJ*wij + D66*N1y*detJ*wij)
    K[3+c1, 4+c1] += E45*N1**2*detJ*wij + N1x*(D16*N1x*detJ*wij + D66*N1y*detJ*wij) + N1y*(D12*N1x*detJ*wij + D26*N1y*detJ*wij)
    K[3+c1, 0+c2] += N2x*(B11*N1x*detJ*wij + B16*N1y*detJ*wij) + N2y*(B16*N1x*detJ*wij + B66*N1y*detJ*wij)
    K[3+c1, 1+c2] += N2x*(B16*N1x*detJ*wij + B66*N1y*detJ*wij) + N2y*(B12*N1x*detJ*wij + B26*N1y*detJ*wij)
    K[3+c1, 2+c2] += E45*N1*N2y*detJ*wij + E55*N1*N2x*detJ*wij
    K[3+c1, 3+c2] += E55*N1*N2*detJ*wij + N2x*(D11*N1x*detJ*wij + D16*N1y*detJ*wij) + N2y*(D16*N1x*detJ*wij + D66*N1y*detJ*wij)
    K[3+c1, 4+c2] += E45*N1*N2*detJ*wij + N2x*(D16*N1x*detJ*wij + D66*N1y*detJ*wij) + N2y*(D12*N1x*detJ*wij + D26*N1y*detJ*wij)
    K[3+c1, 0+c3] += N3x*(B11*N1x*detJ*wij + B16*N1y*detJ*wij) + N3y*(B16*N1x*detJ*wij + B66*N1y*detJ*wij)
    K[3+c1, 1+c3] += N3x*(B16*N1x*detJ*wij + B66*N1y*detJ*wij) + N3y*(B12*N1x*detJ*wij + B26*N1y*detJ*wij)
    K[3+c1, 2+c3] += E45*N1*N3y*detJ*wij + E55*N1*N3x*detJ*wij
    K[3+c1, 3+c3] += E55*N1*N3*detJ*wij + N3x*(D11*N1x*detJ*wij + D16*N1y*detJ*wij) + N3y*(D16*N1x*detJ*wij + D66*N1y*detJ*wij)
    K[3+c1, 4+c3] += E45*N1*N3*detJ*wij + N3x*(D16*N1x*detJ*wij + D66*N1y*detJ*wij) + N3y*(D12*N1x*detJ*wij + D26*N1y*detJ*wij)
    K[3+c1, 0+c4] += N4x*(B11*N1x*detJ*wij + B16*N1y*detJ*wij) + N4y*(B16*N1x*detJ*wij + B66*N1y*detJ*wij)
    K[3+c1, 1+c4] += N4x*(B16*N1x*detJ*wij + B66*N1y*detJ*wij) + N4y*(B12*N1x*detJ*wij + B26*N1y*detJ*wij)
    K[3+c1, 2+c4] += E45*N1*N4y*detJ*wij + E55*N1*N4x*detJ*wij
    K[3+c1, 3+c4] += E55*N1*N4*detJ*wij + N4x*(D11*N1x*detJ*wij + D16*N1y*detJ*wij) + N4y*(D16*N1x*detJ*wij + D66*N1y*detJ*wij)
    K[3+c1, 4+c4] += E45*N1*N4*detJ*wij + N4x*(D16*N1x*detJ*wij + D66*N1y*detJ*wij) + N4y*(D12*N1x*detJ*wij + D26*N1y*detJ*wij)
    K[4+c1, 0+c1] += N1x*(B12*N1y*detJ*wij + B16*N1x*detJ*wij) + N1y*(B26*N1y*detJ*wij + B66*N1x*detJ*wij)
    K[4+c1, 1+c1] += N1x*(B26*N1y*detJ*wij + B66*N1x*detJ*wij) + N1y*(B22*N1y*detJ*wij + B26*N1x*detJ*wij)
    K[4+c1, 2+c1] += E44*N1*N1y*detJ*wij + E45*N1*N1x*detJ*wij
    K[4+c1, 3+c1] += E45*N1**2*detJ*wij + N1x*(D12*N1y*detJ*wij + D16*N1x*detJ*wij) + N1y*(D26*N1y*detJ*wij + D66*N1x*detJ*wij)
    K[4+c1, 4+c1] += E44*N1**2*detJ*wij + N1x*(D26*N1y*detJ*wij + D66*N1x*detJ*wij) + N1y*(D22*N1y*detJ*wij + D26*N1x*detJ*wij)
    K[4+c1, 0+c2] += N2x*(B12*N1y*detJ*wij + B16*N1x*detJ*wij) + N2y*(B26*N1y*detJ*wij + B66*N1x*detJ*wij)
    K[4+c1, 1+c2] += N2x*(B26*N1y*detJ*wij + B66*N1x*detJ*wij) + N2y*(B22*N1y*detJ*wij + B26*N1x*detJ*wij)
    K[4+c1, 2+c2] += E44*N1*N2y*detJ*wij + E45*N1*N2x*detJ*wij
    K[4+c1, 3+c2] += E45*N1*N2*detJ*wij + N2x*(D12*N1y*detJ*wij + D16*N1x*detJ*wij) + N2y*(D26*N1y*detJ*wij + D66*N1x*detJ*wij)
    K[4+c1, 4+c2] += E44*N1*N2*detJ*wij + N2x*(D26*N1y*detJ*wij + D66*N1x*detJ*wij) + N2y*(D22*N1y*detJ*wij + D26*N1x*detJ*wij)
    K[4+c1, 0+c3] += N3x*(B12*N1y*detJ*wij + B16*N1x*detJ*wij) + N3y*(B26*N1y*detJ*wij + B66*N1x*detJ*wij)
    K[4+c1, 1+c3] += N3x*(B26*N1y*detJ*wij + B66*N1x*detJ*wij) + N3y*(B22*N1y*detJ*wij + B26*N1x*detJ*wij)
    K[4+c1, 2+c3] += E44*N1*N3y*detJ*wij + E45*N1*N3x*detJ*wij
    K[4+c1, 3+c3] += E45*N1*N3*detJ*wij + N3x*(D12*N1y*detJ*wij + D16*N1x*detJ*wij) + N3y*(D26*N1y*detJ*wij + D66*N1x*detJ*wij)
    K[4+c1, 4+c3] += E44*N1*N3*detJ*wij + N3x*(D26*N1y*detJ*wij + D66*N1x*detJ*wij) + N3y*(D22*N1y*detJ*wij + D26*N1x*detJ*wij)
    K[4+c1, 0+c4] += N4x*(B12*N1y*detJ*wij + B16*N1x*detJ*wij) + N4y*(B26*N1y*detJ*wij + B66*N1x*detJ*wij)
    K[4+c1, 1+c4] += N4x*(B26*N1y*detJ*wij + B66*N1x*detJ*wij) + N4y*(B22*N1y*detJ*wij + B26*N1x*detJ*wij)
    K[4+c1, 2+c4] += E44*N1*N4y*detJ*wij + E45*N1*N4x*detJ*wij
    K[4+c1, 3+c4] += E45*N1*N4*detJ*wij + N4x*(D12*N1y*detJ*wij + D16*N1x*detJ*wij) + N4y*(D26*N1y*detJ*wij + D66*N1x*detJ*wij)
    K[4+c1, 4+c4] += E44*N1*N4*detJ*wij + N4x*(D26*N1y*detJ*wij + D66*N1x*detJ*wij) + N4y*(D22*N1y*detJ*wij + D26*N1x*detJ*wij)
    K[0+c2, 0+c1] += N1x*(A11*N2x*detJ*wij + A16*N2y*detJ*wij) + N1y*(A16*N2x*detJ*wij + A66*N2y*detJ*wij)
    K[0+c2, 1+c1] += N1x*(A16*N2x*detJ*wij + A66*N2y*detJ*wij) + N1y*(A12*N2x*detJ*wij + A26*N2y*detJ*wij)
    K[0+c2, 3+c1] += N1x*(B11*N2x*detJ*wij + B16*N2y*detJ*wij) + N1y*(B16*N2x*detJ*wij + B66*N2y*detJ*wij)
    K[0+c2, 4+c1] += N1x*(B16*N2x*detJ*wij + B66*N2y*detJ*wij) + N1y*(B12*N2x*detJ*wij + B26*N2y*detJ*wij)
    K[0+c2, 0+c2] += N2x*(A11*N2x*detJ*wij + A16*N2y*detJ*wij) + N2y*(A16*N2x*detJ*wij + A66*N2y*detJ*wij)
    K[0+c2, 1+c2] += N2x*(A16*N2x*detJ*wij + A66*N2y*detJ*wij) + N2y*(A12*N2x*detJ*wij + A26*N2y*detJ*wij)
    K[0+c2, 3+c2] += N2x*(B11*N2x*detJ*wij + B16*N2y*detJ*wij) + N2y*(B16*N2x*detJ*wij + B66*N2y*detJ*wij)
    K[0+c2, 4+c2] += N2x*(B16*N2x*detJ*wij + B66*N2y*detJ*wij) + N2y*(B12*N2x*detJ*wij + B26*N2y*detJ*wij)
    K[0+c2, 0+c3] += N3x*(A11*N2x*detJ*wij + A16*N2y*detJ*wij) + N3y*(A16*N2x*detJ*wij + A66*N2y*detJ*wij)
    K[0+c2, 1+c3] += N3x*(A16*N2x*detJ*wij + A66*N2y*detJ*wij) + N3y*(A12*N2x*detJ*wij + A26*N2y*detJ*wij)
    K[0+c2, 3+c3] += N3x*(B11*N2x*detJ*wij + B16*N2y*detJ*wij) + N3y*(B16*N2x*detJ*wij + B66*N2y*detJ*wij)
    K[0+c2, 4+c3] += N3x*(B16*N2x*detJ*wij + B66*N2y*detJ*wij) + N3y*(B12*N2x*detJ*wij + B26*N2y*detJ*wij)
    K[0+c2, 0+c4] += N4x*(A11*N2x*detJ*wij + A16*N2y*detJ*wij) + N4y*(A16*N2x*detJ*wij + A66*N2y*detJ*wij)
    K[0+c2, 1+c4] += N4x*(A16*N2x*detJ*wij + A66*N2y*detJ*wij) + N4y*(A12*N2x*detJ*wij + A26*N2y*detJ*wij)
    K[0+c2, 3+c4] += N4x*(B11*N2x*detJ*wij + B16*N2y*detJ*wij) + N4y*(B16*N2x*detJ*wij + B66*N2y*detJ*wij)
    K[0+c2, 4+c4] += N4x*(B16*N2x*detJ*wij + B66*N2y*detJ*wij) + N4y*(B12*N2x*detJ*wij + B26*N2y*detJ*wij)
    K[1+c2, 0+c1] += N1x*(A12*N2y*detJ*wij + A16*N2x*detJ*wij) + N1y*(A26*N2y*detJ*wij + A66*N2x*detJ*wij)
    K[1+c2, 1+c1] += N1x*(A26*N2y*detJ*wij + A66*N2x*detJ*wij) + N1y*(A22*N2y*detJ*wij + A26*N2x*detJ*wij)
    K[1+c2, 3+c1] += N1x*(B12*N2y*detJ*wij + B16*N2x*detJ*wij) + N1y*(B26*N2y*detJ*wij + B66*N2x*detJ*wij)
    K[1+c2, 4+c1] += N1x*(B26*N2y*detJ*wij + B66*N2x*detJ*wij) + N1y*(B22*N2y*detJ*wij + B26*N2x*detJ*wij)
    K[1+c2, 0+c2] += N2x*(A12*N2y*detJ*wij + A16*N2x*detJ*wij) + N2y*(A26*N2y*detJ*wij + A66*N2x*detJ*wij)
    K[1+c2, 1+c2] += N2x*(A26*N2y*detJ*wij + A66*N2x*detJ*wij) + N2y*(A22*N2y*detJ*wij + A26*N2x*detJ*wij)
    K[1+c2, 3+c2] += N2x*(B12*N2y*detJ*wij + B16*N2x*detJ*wij) + N2y*(B26*N2y*detJ*wij + B66*N2x*detJ*wij)
    K[1+c2, 4+c2] += N2x*(B26*N2y*detJ*wij + B66*N2x*detJ*wij) + N2y*(B22*N2y*detJ*wij + B26*N2x*detJ*wij)
    K[1+c2, 0+c3] += N3x*(A12*N2y*detJ*wij + A16*N2x*detJ*wij) + N3y*(A26*N2y*detJ*wij + A66*N2x*detJ*wij)
    K[1+c2, 1+c3] += N3x*(A26*N2y*detJ*wij + A66*N2x*detJ*wij) + N3y*(A22*N2y*detJ*wij + A26*N2x*detJ*wij)
    K[1+c2, 3+c3] += N3x*(B12*N2y*detJ*wij + B16*N2x*detJ*wij) + N3y*(B26*N2y*detJ*wij + B66*N2x*detJ*wij)
    K[1+c2, 4+c3] += N3x*(B26*N2y*detJ*wij + B66*N2x*detJ*wij) + N3y*(B22*N2y*detJ*wij + B26*N2x*detJ*wij)
    K[1+c2, 0+c4] += N4x*(A12*N2y*detJ*wij + A16*N2x*detJ*wij) + N4y*(A26*N2y*detJ*wij + A66*N2x*detJ*wij)
    K[1+c2, 1+c4] += N4x*(A26*N2y*detJ*wij + A66*N2x*detJ*wij) + N4y*(A22*N2y*detJ*wij + A26*N2x*detJ*wij)
    K[1+c2, 3+c4] += N4x*(B12*N2y*detJ*wij + B16*N2x*detJ*wij) + N4y*(B26*N2y*detJ*wij + B66*N2x*detJ*wij)
    K[1+c2, 4+c4] += N4x*(B26*N2y*detJ*wij + B66*N2x*detJ*wij) + N4y*(B22*N2y*detJ*wij + B26*N2x*detJ*wij)
    K[2+c2, 2+c1] += N1x*(E45*N2y*detJ*wij + E55*N2x*detJ*wij) + N1y*(E44*N2y*detJ*wij + E45*N2x*detJ*wij)
    K[2+c2, 3+c1] += N1*(E45*N2y*detJ*wij + E55*N2x*detJ*wij)
    K[2+c2, 4+c1] += N1*(E44*N2y*detJ*wij + E45*N2x*detJ*wij)
    K[2+c2, 2+c2] += N2x*(E45*N2y*detJ*wij + E55*N2x*detJ*wij) + N2y*(E44*N2y*detJ*wij + E45*N2x*detJ*wij)
    K[2+c2, 3+c2] += N2*(E45*N2y*detJ*wij + E55*N2x*detJ*wij)
    K[2+c2, 4+c2] += N2*(E44*N2y*detJ*wij + E45*N2x*detJ*wij)
    K[2+c2, 2+c3] += N3x*(E45*N2y*detJ*wij + E55*N2x*detJ*wij) + N3y*(E44*N2y*detJ*wij + E45*N2x*detJ*wij)
    K[2+c2, 3+c3] += N3*(E45*N2y*detJ*wij + E55*N2x*detJ*wij)
    K[2+c2, 4+c3] += N3*(E44*N2y*detJ*wij + E45*N2x*detJ*wij)
    K[2+c2, 2+c4] += N4x*(E45*N2y*detJ*wij + E55*N2x*detJ*wij) + N4y*(E44*N2y*detJ*wij + E45*N2x*detJ*wij)
    K[2+c2, 3+c4] += N4*(E45*N2y*detJ*wij + E55*N2x*detJ*wij)
    K[2+c2, 4+c4] += N4*(E44*N2y*detJ*wij + E45*N2x*detJ*wij)
    K[3+c2, 0+c1] += N1x*(B11*N2x*detJ*wij + B16*N2y*detJ*wij) + N1y*(B16*N2x*detJ*wij + B66*N2y*detJ*wij)
    K[3+c2, 1+c1] += N1x*(B16*N2x*detJ*wij + B66*N2y*detJ*wij) + N1y*(B12*N2x*detJ*wij + B26*N2y*detJ*wij)
    K[3+c2, 2+c1] += E45*N1y*N2*detJ*wij + E55*N1x*N2*detJ*wij
    K[3+c2, 3+c1] += E55*N1*N2*detJ*wij + N1x*(D11*N2x*detJ*wij + D16*N2y*detJ*wij) + N1y*(D16*N2x*detJ*wij + D66*N2y*detJ*wij)
    K[3+c2, 4+c1] += E45*N1*N2*detJ*wij + N1x*(D16*N2x*detJ*wij + D66*N2y*detJ*wij) + N1y*(D12*N2x*detJ*wij + D26*N2y*detJ*wij)
    K[3+c2, 0+c2] += N2x*(B11*N2x*detJ*wij + B16*N2y*detJ*wij) + N2y*(B16*N2x*detJ*wij + B66*N2y*detJ*wij)
    K[3+c2, 1+c2] += N2x*(B16*N2x*detJ*wij + B66*N2y*detJ*wij) + N2y*(B12*N2x*detJ*wij + B26*N2y*detJ*wij)
    K[3+c2, 2+c2] += E45*N2*N2y*detJ*wij + E55*N2*N2x*detJ*wij
    K[3+c2, 3+c2] += E55*N2**2*detJ*wij + N2x*(D11*N2x*detJ*wij + D16*N2y*detJ*wij) + N2y*(D16*N2x*detJ*wij + D66*N2y*detJ*wij)
    K[3+c2, 4+c2] += E45*N2**2*detJ*wij + N2x*(D16*N2x*detJ*wij + D66*N2y*detJ*wij) + N2y*(D12*N2x*detJ*wij + D26*N2y*detJ*wij)
    K[3+c2, 0+c3] += N3x*(B11*N2x*detJ*wij + B16*N2y*detJ*wij) + N3y*(B16*N2x*detJ*wij + B66*N2y*detJ*wij)
    K[3+c2, 1+c3] += N3x*(B16*N2x*detJ*wij + B66*N2y*detJ*wij) + N3y*(B12*N2x*detJ*wij + B26*N2y*detJ*wij)
    K[3+c2, 2+c3] += E45*N2*N3y*detJ*wij + E55*N2*N3x*detJ*wij
    K[3+c2, 3+c3] += E55*N2*N3*detJ*wij + N3x*(D11*N2x*detJ*wij + D16*N2y*detJ*wij) + N3y*(D16*N2x*detJ*wij + D66*N2y*detJ*wij)
    K[3+c2, 4+c3] += E45*N2*N3*detJ*wij + N3x*(D16*N2x*detJ*wij + D66*N2y*detJ*wij) + N3y*(D12*N2x*detJ*wij + D26*N2y*detJ*wij)
    K[3+c2, 0+c4] += N4x*(B11*N2x*detJ*wij + B16*N2y*detJ*wij) + N4y*(B16*N2x*detJ*wij + B66*N2y*detJ*wij)
    K[3+c2, 1+c4] += N4x*(B16*N2x*detJ*wij + B66*N2y*detJ*wij) + N4y*(B12*N2x*detJ*wij + B26*N2y*detJ*wij)
    K[3+c2, 2+c4] += E45*N2*N4y*detJ*wij + E55*N2*N4x*detJ*wij
    K[3+c2, 3+c4] += E55*N2*N4*detJ*wij + N4x*(D11*N2x*detJ*wij + D16*N2y*detJ*wij) + N4y*(D16*N2x*detJ*wij + D66*N2y*detJ*wij)
    K[3+c2, 4+c4] += E45*N2*N4*detJ*wij + N4x*(D16*N2x*detJ*wij + D66*N2y*detJ*wij) + N4y*(D12*N2x*detJ*wij + D26*N2y*detJ*wij)
    K[4+c2, 0+c1] += N1x*(B12*N2y*detJ*wij + B16*N2x*detJ*wij) + N1y*(B26*N2y*detJ*wij + B66*N2x*detJ*wij)
    K[4+c2, 1+c1] += N1x*(B26*N2y*detJ*wij + B66*N2x*detJ*wij) + N1y*(B22*N2y*detJ*wij + B26*N2x*detJ*wij)
    K[4+c2, 2+c1] += E44*N1y*N2*detJ*wij + E45*N1x*N2*detJ*wij
    K[4+c2, 3+c1] += E45*N1*N2*detJ*wij + N1x*(D12*N2y*detJ*wij + D16*N2x*detJ*wij) + N1y*(D26*N2y*detJ*wij + D66*N2x*detJ*wij)
    K[4+c2, 4+c1] += E44*N1*N2*detJ*wij + N1x*(D26*N2y*detJ*wij + D66*N2x*detJ*wij) + N1y*(D22*N2y*detJ*wij + D26*N2x*detJ*wij)
    K[4+c2, 0+c2] += N2x*(B12*N2y*detJ*wij + B16*N2x*detJ*wij) + N2y*(B26*N2y*detJ*wij + B66*N2x*detJ*wij)
    K[4+c2, 1+c2] += N2x*(B26*N2y*detJ*wij + B66*N2x*detJ*wij) + N2y*(B22*N2y*detJ*wij + B26*N2x*detJ*wij)
    K[4+c2, 2+c2] += E44*N2*N2y*detJ*wij + E45*N2*N2x*detJ*wij
    K[4+c2, 3+c2] += E45*N2**2*detJ*wij + N2x*(D12*N2y*detJ*wij + D16*N2x*detJ*wij) + N2y*(D26*N2y*detJ*wij + D66*N2x*detJ*wij)
    K[4+c2, 4+c2] += E44*N2**2*detJ*wij + N2x*(D26*N2y*detJ*wij + D66*N2x*detJ*wij) + N2y*(D22*N2y*detJ*wij + D26*N2x*detJ*wij)
    K[4+c2, 0+c3] += N3x*(B12*N2y*detJ*wij + B16*N2x*detJ*wij) + N3y*(B26*N2y*detJ*wij + B66*N2x*detJ*wij)
    K[4+c2, 1+c3] += N3x*(B26*N2y*detJ*wij + B66*N2x*detJ*wij) + N3y*(B22*N2y*detJ*wij + B26*N2x*detJ*wij)
    K[4+c2, 2+c3] += E44*N2*N3y*detJ*wij + E45*N2*N3x*detJ*wij
    K[4+c2, 3+c3] += E45*N2*N3*detJ*wij + N3x*(D12*N2y*detJ*wij + D16*N2x*detJ*wij) + N3y*(D26*N2y*detJ*wij + D66*N2x*detJ*wij)
    K[4+c2, 4+c3] += E44*N2*N3*detJ*wij + N3x*(D26*N2y*detJ*wij + D66*N2x*detJ*wij) + N3y*(D22*N2y*detJ*wij + D26*N2x*detJ*wij)
    K[4+c2, 0+c4] += N4x*(B12*N2y*detJ*wij + B16*N2x*detJ*wij) + N4y*(B26*N2y*detJ*wij + B66*N2x*detJ*wij)
    K[4+c2, 1+c4] += N4x*(B26*N2y*detJ*wij + B66*N2x*detJ*wij) + N4y*(B22*N2y*detJ*wij + B26*N2x*detJ*wij)
    K[4+c2, 2+c4] += E44*N2*N4y*detJ*wij + E45*N2*N4x*detJ*wij
    K[4+c2, 3+c4] += E45*N2*N4*detJ*wij + N4x*(D12*N2y*detJ*wij + D16*N2x*detJ*wij) + N4y*(D26*N2y*detJ*wij + D66*N2x*detJ*wij)
    K[4+c2, 4+c4] += E44*N2*N4*detJ*wij + N4x*(D26*N2y*detJ*wij + D66*N2x*detJ*wij) + N4y*(D22*N2y*detJ*wij + D26*N2x*detJ*wij)
    K[0+c3, 0+c1] += N1x*(A11*N3x*detJ*wij + A16*N3y*detJ*wij) + N1y*(A16*N3x*detJ*wij + A66*N3y*detJ*wij)
    K[0+c3, 1+c1] += N1x*(A16*N3x*detJ*wij + A66*N3y*detJ*wij) + N1y*(A12*N3x*detJ*wij + A26*N3y*detJ*wij)
    K[0+c3, 3+c1] += N1x*(B11*N3x*detJ*wij + B16*N3y*detJ*wij) + N1y*(B16*N3x*detJ*wij + B66*N3y*detJ*wij)
    K[0+c3, 4+c1] += N1x*(B16*N3x*detJ*wij + B66*N3y*detJ*wij) + N1y*(B12*N3x*detJ*wij + B26*N3y*detJ*wij)
    K[0+c3, 0+c2] += N2x*(A11*N3x*detJ*wij + A16*N3y*detJ*wij) + N2y*(A16*N3x*detJ*wij + A66*N3y*detJ*wij)
    K[0+c3, 1+c2] += N2x*(A16*N3x*detJ*wij + A66*N3y*detJ*wij) + N2y*(A12*N3x*detJ*wij + A26*N3y*detJ*wij)
    K[0+c3, 3+c2] += N2x*(B11*N3x*detJ*wij + B16*N3y*detJ*wij) + N2y*(B16*N3x*detJ*wij + B66*N3y*detJ*wij)
    K[0+c3, 4+c2] += N2x*(B16*N3x*detJ*wij + B66*N3y*detJ*wij) + N2y*(B12*N3x*detJ*wij + B26*N3y*detJ*wij)
    K[0+c3, 0+c3] += N3x*(A11*N3x*detJ*wij + A16*N3y*detJ*wij) + N3y*(A16*N3x*detJ*wij + A66*N3y*detJ*wij)
    K[0+c3, 1+c3] += N3x*(A16*N3x*detJ*wij + A66*N3y*detJ*wij) + N3y*(A12*N3x*detJ*wij + A26*N3y*detJ*wij)
    K[0+c3, 3+c3] += N3x*(B11*N3x*detJ*wij + B16*N3y*detJ*wij) + N3y*(B16*N3x*detJ*wij + B66*N3y*detJ*wij)
    K[0+c3, 4+c3] += N3x*(B16*N3x*detJ*wij + B66*N3y*detJ*wij) + N3y*(B12*N3x*detJ*wij + B26*N3y*detJ*wij)
    K[0+c3, 0+c4] += N4x*(A11*N3x*detJ*wij + A16*N3y*detJ*wij) + N4y*(A16*N3x*detJ*wij + A66*N3y*detJ*wij)
    K[0+c3, 1+c4] += N4x*(A16*N3x*detJ*wij + A66*N3y*detJ*wij) + N4y*(A12*N3x*detJ*wij + A26*N3y*detJ*wij)
    K[0+c3, 3+c4] += N4x*(B11*N3x*detJ*wij + B16*N3y*detJ*wij) + N4y*(B16*N3x*detJ*wij + B66*N3y*detJ*wij)
    K[0+c3, 4+c4] += N4x*(B16*N3x*detJ*wij + B66*N3y*detJ*wij) + N4y*(B12*N3x*detJ*wij + B26*N3y*detJ*wij)
    K[1+c3, 0+c1] += N1x*(A12*N3y*detJ*wij + A16*N3x*detJ*wij) + N1y*(A26*N3y*detJ*wij + A66*N3x*detJ*wij)
    K[1+c3, 1+c1] += N1x*(A26*N3y*detJ*wij + A66*N3x*detJ*wij) + N1y*(A22*N3y*detJ*wij + A26*N3x*detJ*wij)
    K[1+c3, 3+c1] += N1x*(B12*N3y*detJ*wij + B16*N3x*detJ*wij) + N1y*(B26*N3y*detJ*wij + B66*N3x*detJ*wij)
    K[1+c3, 4+c1] += N1x*(B26*N3y*detJ*wij + B66*N3x*detJ*wij) + N1y*(B22*N3y*detJ*wij + B26*N3x*detJ*wij)
    K[1+c3, 0+c2] += N2x*(A12*N3y*detJ*wij + A16*N3x*detJ*wij) + N2y*(A26*N3y*detJ*wij + A66*N3x*detJ*wij)
    K[1+c3, 1+c2] += N2x*(A26*N3y*detJ*wij + A66*N3x*detJ*wij) + N2y*(A22*N3y*detJ*wij + A26*N3x*detJ*wij)
    K[1+c3, 3+c2] += N2x*(B12*N3y*detJ*wij + B16*N3x*detJ*wij) + N2y*(B26*N3y*detJ*wij + B66*N3x*detJ*wij)
    K[1+c3, 4+c2] += N2x*(B26*N3y*detJ*wij + B66*N3x*detJ*wij) + N2y*(B22*N3y*detJ*wij + B26*N3x*detJ*wij)
    K[1+c3, 0+c3] += N3x*(A12*N3y*detJ*wij + A16*N3x*detJ*wij) + N3y*(A26*N3y*detJ*wij + A66*N3x*detJ*wij)
    K[1+c3, 1+c3] += N3x*(A26*N3y*detJ*wij + A66*N3x*detJ*wij) + N3y*(A22*N3y*detJ*wij + A26*N3x*detJ*wij)
    K[1+c3, 3+c3] += N3x*(B12*N3y*detJ*wij + B16*N3x*detJ*wij) + N3y*(B26*N3y*detJ*wij + B66*N3x*detJ*wij)
    K[1+c3, 4+c3] += N3x*(B26*N3y*detJ*wij + B66*N3x*detJ*wij) + N3y*(B22*N3y*detJ*wij + B26*N3x*detJ*wij)
    K[1+c3, 0+c4] += N4x*(A12*N3y*detJ*wij + A16*N3x*detJ*wij) + N4y*(A26*N3y*detJ*wij + A66*N3x*detJ*wij)
    K[1+c3, 1+c4] += N4x*(A26*N3y*detJ*wij + A66*N3x*detJ*wij) + N4y*(A22*N3y*detJ*wij + A26*N3x*detJ*wij)
    K[1+c3, 3+c4] += N4x*(B12*N3y*detJ*wij + B16*N3x*detJ*wij) + N4y*(B26*N3y*detJ*wij + B66*N3x*detJ*wij)
    K[1+c3, 4+c4] += N4x*(B26*N3y*detJ*wij + B66*N3x*detJ*wij) + N4y*(B22*N3y*detJ*wij + B26*N3x*detJ*wij)
    K[2+c3, 2+c1] += N1x*(E45*N3y*detJ*wij + E55*N3x*detJ*wij) + N1y*(E44*N3y*detJ*wij + E45*N3x*detJ*wij)
    K[2+c3, 3+c1] += N1*(E45*N3y*detJ*wij + E55*N3x*detJ*wij)
    K[2+c3, 4+c1] += N1*(E44*N3y*detJ*wij + E45*N3x*detJ*wij)
    K[2+c3, 2+c2] += N2x*(E45*N3y*detJ*wij + E55*N3x*detJ*wij) + N2y*(E44*N3y*detJ*wij + E45*N3x*detJ*wij)
    K[2+c3, 3+c2] += N2*(E45*N3y*detJ*wij + E55*N3x*detJ*wij)
    K[2+c3, 4+c2] += N2*(E44*N3y*detJ*wij + E45*N3x*detJ*wij)
    K[2+c3, 2+c3] += N3x*(E45*N3y*detJ*wij + E55*N3x*detJ*wij) + N3y*(E44*N3y*detJ*wij + E45*N3x*detJ*wij)
    K[2+c3, 3+c3] += N3*(E45*N3y*detJ*wij + E55*N3x*detJ*wij)
    K[2+c3, 4+c3] += N3*(E44*N3y*detJ*wij + E45*N3x*detJ*wij)
    K[2+c3, 2+c4] += N4x*(E45*N3y*detJ*wij + E55*N3x*detJ*wij) + N4y*(E44*N3y*detJ*wij + E45*N3x*detJ*wij)
    K[2+c3, 3+c4] += N4*(E45*N3y*detJ*wij + E55*N3x*detJ*wij)
    K[2+c3, 4+c4] += N4*(E44*N3y*detJ*wij + E45*N3x*detJ*wij)
    K[3+c3, 0+c1] += N1x*(B11*N3x*detJ*wij + B16*N3y*detJ*wij) + N1y*(B16*N3x*detJ*wij + B66*N3y*detJ*wij)
    K[3+c3, 1+c1] += N1x*(B16*N3x*detJ*wij + B66*N3y*detJ*wij) + N1y*(B12*N3x*detJ*wij + B26*N3y*detJ*wij)
    K[3+c3, 2+c1] += E45*N1y*N3*detJ*wij + E55*N1x*N3*detJ*wij
    K[3+c3, 3+c1] += E55*N1*N3*detJ*wij + N1x*(D11*N3x*detJ*wij + D16*N3y*detJ*wij) + N1y*(D16*N3x*detJ*wij + D66*N3y*detJ*wij)
    K[3+c3, 4+c1] += E45*N1*N3*detJ*wij + N1x*(D16*N3x*detJ*wij + D66*N3y*detJ*wij) + N1y*(D12*N3x*detJ*wij + D26*N3y*detJ*wij)
    K[3+c3, 0+c2] += N2x*(B11*N3x*detJ*wij + B16*N3y*detJ*wij) + N2y*(B16*N3x*detJ*wij + B66*N3y*detJ*wij)
    K[3+c3, 1+c2] += N2x*(B16*N3x*detJ*wij + B66*N3y*detJ*wij) + N2y*(B12*N3x*detJ*wij + B26*N3y*detJ*wij)
    K[3+c3, 2+c2] += E45*N2y*N3*detJ*wij + E55*N2x*N3*detJ*wij
    K[3+c3, 3+c2] += E55*N2*N3*detJ*wij + N2x*(D11*N3x*detJ*wij + D16*N3y*detJ*wij) + N2y*(D16*N3x*detJ*wij + D66*N3y*detJ*wij)
    K[3+c3, 4+c2] += E45*N2*N3*detJ*wij + N2x*(D16*N3x*detJ*wij + D66*N3y*detJ*wij) + N2y*(D12*N3x*detJ*wij + D26*N3y*detJ*wij)
    K[3+c3, 0+c3] += N3x*(B11*N3x*detJ*wij + B16*N3y*detJ*wij) + N3y*(B16*N3x*detJ*wij + B66*N3y*detJ*wij)
    K[3+c3, 1+c3] += N3x*(B16*N3x*detJ*wij + B66*N3y*detJ*wij) + N3y*(B12*N3x*detJ*wij + B26*N3y*detJ*wij)
    K[3+c3, 2+c3] += E45*N3*N3y*detJ*wij + E55*N3*N3x*detJ*wij
    K[3+c3, 3+c3] += E55*N3**2*detJ*wij + N3x*(D11*N3x*detJ*wij + D16*N3y*detJ*wij) + N3y*(D16*N3x*detJ*wij + D66*N3y*detJ*wij)
    K[3+c3, 4+c3] += E45*N3**2*detJ*wij + N3x*(D16*N3x*detJ*wij + D66*N3y*detJ*wij) + N3y*(D12*N3x*detJ*wij + D26*N3y*detJ*wij)
    K[3+c3, 0+c4] += N4x*(B11*N3x*detJ*wij + B16*N3y*detJ*wij) + N4y*(B16*N3x*detJ*wij + B66*N3y*detJ*wij)
    K[3+c3, 1+c4] += N4x*(B16*N3x*detJ*wij + B66*N3y*detJ*wij) + N4y*(B12*N3x*detJ*wij + B26*N3y*detJ*wij)
    K[3+c3, 2+c4] += E45*N3*N4y*detJ*wij + E55*N3*N4x*detJ*wij
    K[3+c3, 3+c4] += E55*N3*N4*detJ*wij + N4x*(D11*N3x*detJ*wij + D16*N3y*detJ*wij) + N4y*(D16*N3x*detJ*wij + D66*N3y*detJ*wij)
    K[3+c3, 4+c4] += E45*N3*N4*detJ*wij + N4x*(D16*N3x*detJ*wij + D66*N3y*detJ*wij) + N4y*(D12*N3x*detJ*wij + D26*N3y*detJ*wij)
    K[4+c3, 0+c1] += N1x*(B12*N3y*detJ*wij + B16*N3x*detJ*wij) + N1y*(B26*N3y*detJ*wij + B66*N3x*detJ*wij)
    K[4+c3, 1+c1] += N1x*(B26*N3y*detJ*wij + B66*N3x*detJ*wij) + N1y*(B22*N3y*detJ*wij + B26*N3x*detJ*wij)
    K[4+c3, 2+c1] += E44*N1y*N3*detJ*wij + E45*N1x*N3*detJ*wij
    K[4+c3, 3+c1] += E45*N1*N3*detJ*wij + N1x*(D12*N3y*detJ*wij + D16*N3x*detJ*wij) + N1y*(D26*N3y*detJ*wij + D66*N3x*detJ*wij)
    K[4+c3, 4+c1] += E44*N1*N3*detJ*wij + N1x*(D26*N3y*detJ*wij + D66*N3x*detJ*wij) + N1y*(D22*N3y*detJ*wij + D26*N3x*detJ*wij)
    K[4+c3, 0+c2] += N2x*(B12*N3y*detJ*wij + B16*N3x*detJ*wij) + N2y*(B26*N3y*detJ*wij + B66*N3x*detJ*wij)
    K[4+c3, 1+c2] += N2x*(B26*N3y*detJ*wij + B66*N3x*detJ*wij) + N2y*(B22*N3y*detJ*wij + B26*N3x*detJ*wij)
    K[4+c3, 2+c2] += E44*N2y*N3*detJ*wij + E45*N2x*N3*detJ*wij
    K[4+c3, 3+c2] += E45*N2*N3*detJ*wij + N2x*(D12*N3y*detJ*wij + D16*N3x*detJ*wij) + N2y*(D26*N3y*detJ*wij + D66*N3x*detJ*wij)
    K[4+c3, 4+c2] += E44*N2*N3*detJ*wij + N2x*(D26*N3y*detJ*wij + D66*N3x*detJ*wij) + N2y*(D22*N3y*detJ*wij + D26*N3x*detJ*wij)
    K[4+c3, 0+c3] += N3x*(B12*N3y*detJ*wij + B16*N3x*detJ*wij) + N3y*(B26*N3y*detJ*wij + B66*N3x*detJ*wij)
    K[4+c3, 1+c3] += N3x*(B26*N3y*detJ*wij + B66*N3x*detJ*wij) + N3y*(B22*N3y*detJ*wij + B26*N3x*detJ*wij)
    K[4+c3, 2+c3] += E44*N3*N3y*detJ*wij + E45*N3*N3x*detJ*wij
    K[4+c3, 3+c3] += E45*N3**2*detJ*wij + N3x*(D12*N3y*detJ*wij + D16*N3x*detJ*wij) + N3y*(D26*N3y*detJ*wij + D66*N3x*detJ*wij)
    K[4+c3, 4+c3] += E44*N3**2*detJ*wij + N3x*(D26*N3y*detJ*wij + D66*N3x*detJ*wij) + N3y*(D22*N3y*detJ*wij + D26*N3x*detJ*wij)
    K[4+c3, 0+c4] += N4x*(B12*N3y*detJ*wij + B16*N3x*detJ*wij) + N4y*(B26*N3y*detJ*wij + B66*N3x*detJ*wij)
    K[4+c3, 1+c4] += N4x*(B26*N3y*detJ*wij + B66*N3x*detJ*wij) + N4y*(B22*N3y*detJ*wij + B26*N3x*detJ*wij)
    K[4+c3, 2+c4] += E44*N3*N4y*detJ*wij + E45*N3*N4x*detJ*wij
    K[4+c3, 3+c4] += E45*N3*N4*detJ*wij + N4x*(D12*N3y*detJ*wij + D16*N3x*detJ*wij) + N4y*(D26*N3y*detJ*wij + D66*N3x*detJ*wij)
    K[4+c3, 4+c4] += E44*N3*N4*detJ*wij + N4x*(D26*N3y*detJ*wij + D66*N3x*detJ*wij) + N4y*(D22*N3y*detJ*wij + D26*N3x*detJ*wij)
    K[0+c4, 0+c1] += N1x*(A11*N4x*detJ*wij + A16*N4y*detJ*wij) + N1y*(A16*N4x*detJ*wij + A66*N4y*detJ*wij)
    K[0+c4, 1+c1] += N1x*(A16*N4x*detJ*wij + A66*N4y*detJ*wij) + N1y*(A12*N4x*detJ*wij + A26*N4y*detJ*wij)
    K[0+c4, 3+c1] += N1x*(B11*N4x*detJ*wij + B16*N4y*detJ*wij) + N1y*(B16*N4x*detJ*wij + B66*N4y*detJ*wij)
    K[0+c4, 4+c1] += N1x*(B16*N4x*detJ*wij + B66*N4y*detJ*wij) + N1y*(B12*N4x*detJ*wij + B26*N4y*detJ*wij)
    K[0+c4, 0+c2] += N2x*(A11*N4x*detJ*wij + A16*N4y*detJ*wij) + N2y*(A16*N4x*detJ*wij + A66*N4y*detJ*wij)
    K[0+c4, 1+c2] += N2x*(A16*N4x*detJ*wij + A66*N4y*detJ*wij) + N2y*(A12*N4x*detJ*wij + A26*N4y*detJ*wij)
    K[0+c4, 3+c2] += N2x*(B11*N4x*detJ*wij + B16*N4y*detJ*wij) + N2y*(B16*N4x*detJ*wij + B66*N4y*detJ*wij)
    K[0+c4, 4+c2] += N2x*(B16*N4x*detJ*wij + B66*N4y*detJ*wij) + N2y*(B12*N4x*detJ*wij + B26*N4y*detJ*wij)
    K[0+c4, 0+c3] += N3x*(A11*N4x*detJ*wij + A16*N4y*detJ*wij) + N3y*(A16*N4x*detJ*wij + A66*N4y*detJ*wij)
    K[0+c4, 1+c3] += N3x*(A16*N4x*detJ*wij + A66*N4y*detJ*wij) + N3y*(A12*N4x*detJ*wij + A26*N4y*detJ*wij)
    K[0+c4, 3+c3] += N3x*(B11*N4x*detJ*wij + B16*N4y*detJ*wij) + N3y*(B16*N4x*detJ*wij + B66*N4y*detJ*wij)
    K[0+c4, 4+c3] += N3x*(B16*N4x*detJ*wij + B66*N4y*detJ*wij) + N3y*(B12*N4x*detJ*wij + B26*N4y*detJ*wij)
    K[0+c4, 0+c4] += N4x*(A11*N4x*detJ*wij + A16*N4y*detJ*wij) + N4y*(A16*N4x*detJ*wij + A66*N4y*detJ*wij)
    K[0+c4, 1+c4] += N4x*(A16*N4x*detJ*wij + A66*N4y*detJ*wij) + N4y*(A12*N4x*detJ*wij + A26*N4y*detJ*wij)
    K[0+c4, 3+c4] += N4x*(B11*N4x*detJ*wij + B16*N4y*detJ*wij) + N4y*(B16*N4x*detJ*wij + B66*N4y*detJ*wij)
    K[0+c4, 4+c4] += N4x*(B16*N4x*detJ*wij + B66*N4y*detJ*wij) + N4y*(B12*N4x*detJ*wij + B26*N4y*detJ*wij)
    K[1+c4, 0+c1] += N1x*(A12*N4y*detJ*wij + A16*N4x*detJ*wij) + N1y*(A26*N4y*detJ*wij + A66*N4x*detJ*wij)
    K[1+c4, 1+c1] += N1x*(A26*N4y*detJ*wij + A66*N4x*detJ*wij) + N1y*(A22*N4y*detJ*wij + A26*N4x*detJ*wij)
    K[1+c4, 3+c1] += N1x*(B12*N4y*detJ*wij + B16*N4x*detJ*wij) + N1y*(B26*N4y*detJ*wij + B66*N4x*detJ*wij)
    K[1+c4, 4+c1] += N1x*(B26*N4y*detJ*wij + B66*N4x*detJ*wij) + N1y*(B22*N4y*detJ*wij + B26*N4x*detJ*wij)
    K[1+c4, 0+c2] += N2x*(A12*N4y*detJ*wij + A16*N4x*detJ*wij) + N2y*(A26*N4y*detJ*wij + A66*N4x*detJ*wij)
    K[1+c4, 1+c2] += N2x*(A26*N4y*detJ*wij + A66*N4x*detJ*wij) + N2y*(A22*N4y*detJ*wij + A26*N4x*detJ*wij)
    K[1+c4, 3+c2] += N2x*(B12*N4y*detJ*wij + B16*N4x*detJ*wij) + N2y*(B26*N4y*detJ*wij + B66*N4x*detJ*wij)
    K[1+c4, 4+c2] += N2x*(B26*N4y*detJ*wij + B66*N4x*detJ*wij) + N2y*(B22*N4y*detJ*wij + B26*N4x*detJ*wij)
    K[1+c4, 0+c3] += N3x*(A12*N4y*detJ*wij + A16*N4x*detJ*wij) + N3y*(A26*N4y*detJ*wij + A66*N4x*detJ*wij)
    K[1+c4, 1+c3] += N3x*(A26*N4y*detJ*wij + A66*N4x*detJ*wij) + N3y*(A22*N4y*detJ*wij + A26*N4x*detJ*wij)
    K[1+c4, 3+c3] += N3x*(B12*N4y*detJ*wij + B16*N4x*detJ*wij) + N3y*(B26*N4y*detJ*wij + B66*N4x*detJ*wij)
    K[1+c4, 4+c3] += N3x*(B26*N4y*detJ*wij + B66*N4x*detJ*wij) + N3y*(B22*N4y*detJ*wij + B26*N4x*detJ*wij)
    K[1+c4, 0+c4] += N4x*(A12*N4y*detJ*wij + A16*N4x*detJ*wij) + N4y*(A26*N4y*detJ*wij + A66*N4x*detJ*wij)
    K[1+c4, 1+c4] += N4x*(A26*N4y*detJ*wij + A66*N4x*detJ*wij) + N4y*(A22*N4y*detJ*wij + A26*N4x*detJ*wij)
    K[1+c4, 3+c4] += N4x*(B12*N4y*detJ*wij + B16*N4x*detJ*wij) + N4y*(B26*N4y*detJ*wij + B66*N4x*detJ*wij)
    K[1+c4, 4+c4] += N4x*(B26*N4y*detJ*wij + B66*N4x*detJ*wij) + N4y*(B22*N4y*detJ*wij + B26*N4x*detJ*wij)
    K[2+c4, 2+c1] += N1x*(E45*N4y*detJ*wij + E55*N4x*detJ*wij) + N1y*(E44*N4y*detJ*wij + E45*N4x*detJ*wij)
    K[2+c4, 3+c1] += N1*(E45*N4y*detJ*wij + E55*N4x*detJ*wij)
    K[2+c4, 4+c1] += N1*(E44*N4y*detJ*wij + E45*N4x*detJ*wij)
    K[2+c4, 2+c2] += N2x*(E45*N4y*detJ*wij + E55*N4x*detJ*wij) + N2y*(E44*N4y*detJ*wij + E45*N4x*detJ*wij)
    K[2+c4, 3+c2] += N2*(E45*N4y*detJ*wij + E55*N4x*detJ*wij)
    K[2+c4, 4+c2] += N2*(E44*N4y*detJ*wij + E45*N4x*detJ*wij)
    K[2+c4, 2+c3] += N3x*(E45*N4y*detJ*wij + E55*N4x*detJ*wij) + N3y*(E44*N4y*detJ*wij + E45*N4x*detJ*wij)
    K[2+c4, 3+c3] += N3*(E45*N4y*detJ*wij + E55*N4x*detJ*wij)
    K[2+c4, 4+c3] += N3*(E44*N4y*detJ*wij + E45*N4x*detJ*wij)
    K[2+c4, 2+c4] += N4x*(E45*N4y*detJ*wij + E55*N4x*detJ*wij) + N4y*(E44*N4y*detJ*wij + E45*N4x*detJ*wij)
    K[2+c4, 3+c4] += N4*(E45*N4y*detJ*wij + E55*N4x*detJ*wij)
    K[2+c4, 4+c4] += N4*(E44*N4y*detJ*wij + E45*N4x*detJ*wij)
    K[3+c4, 0+c1] += N1x*(B11*N4x*detJ*wij + B16*N4y*detJ*wij) + N1y*(B16*N4x*detJ*wij + B66*N4y*detJ*wij)
    K[3+c4, 1+c1] += N1x*(B16*N4x*detJ*wij + B66*N4y*detJ*wij) + N1y*(B12*N4x*detJ*wij + B26*N4y*detJ*wij)
    K[3+c4, 2+c1] += E45*N1y*N4*detJ*wij + E55*N1x*N4*detJ*wij
    K[3+c4, 3+c1] += E55*N1*N4*detJ*wij + N1x*(D11*N4x*detJ*wij + D16*N4y*detJ*wij) + N1y*(D16*N4x*detJ*wij + D66*N4y*detJ*wij)
    K[3+c4, 4+c1] += E45*N1*N4*detJ*wij + N1x*(D16*N4x*detJ*wij + D66*N4y*detJ*wij) + N1y*(D12*N4x*detJ*wij + D26*N4y*detJ*wij)
    K[3+c4, 0+c2] += N2x*(B11*N4x*detJ*wij + B16*N4y*detJ*wij) + N2y*(B16*N4x*detJ*wij + B66*N4y*detJ*wij)
    K[3+c4, 1+c2] += N2x*(B16*N4x*detJ*wij + B66*N4y*detJ*wij) + N2y*(B12*N4x*detJ*wij + B26*N4y*detJ*wij)
    K[3+c4, 2+c2] += E45*N2y*N4*detJ*wij + E55*N2x*N4*detJ*wij
    K[3+c4, 3+c2] += E55*N2*N4*detJ*wij + N2x*(D11*N4x*detJ*wij + D16*N4y*detJ*wij) + N2y*(D16*N4x*detJ*wij + D66*N4y*detJ*wij)
    K[3+c4, 4+c2] += E45*N2*N4*detJ*wij + N2x*(D16*N4x*detJ*wij + D66*N4y*detJ*wij) + N2y*(D12*N4x*detJ*wij + D26*N4y*detJ*wij)
    K[3+c4, 0+c3] += N3x*(B11*N4x*detJ*wij + B16*N4y*detJ*wij) + N3y*(B16*N4x*detJ*wij + B66*N4y*detJ*wij)
    K[3+c4, 1+c3] += N3x*(B16*N4x*detJ*wij + B66*N4y*detJ*wij) + N3y*(B12*N4x*detJ*wij + B26*N4y*detJ*wij)
    K[3+c4, 2+c3] += E45*N3y*N4*detJ*wij + E55*N3x*N4*detJ*wij
    K[3+c4, 3+c3] += E55*N3*N4*detJ*wij + N3x*(D11*N4x*detJ*wij + D16*N4y*detJ*wij) + N3y*(D16*N4x*detJ*wij + D66*N4y*detJ*wij)
    K[3+c4, 4+c3] += E45*N3*N4*detJ*wij + N3x*(D16*N4x*detJ*wij + D66*N4y*detJ*wij) + N3y*(D12*N4x*detJ*wij + D26*N4y*detJ*wij)
    K[3+c4, 0+c4] += N4x*(B11*N4x*detJ*wij + B16*N4y*detJ*wij) + N4y*(B16*N4x*detJ*wij + B66*N4y*detJ*wij)
    K[3+c4, 1+c4] += N4x*(B16*N4x*detJ*wij + B66*N4y*detJ*wij) + N4y*(B12*N4x*detJ*wij + B26*N4y*detJ*wij)
    K[3+c4, 2+c4] += E45*N4*N4y*detJ*wij + E55*N4*N4x*detJ*wij
    K[3+c4, 3+c4] += E55*N4**2*detJ*wij + N4x*(D11*N4x*detJ*wij + D16*N4y*detJ*wij) + N4y*(D16*N4x*detJ*wij + D66*N4y*detJ*wij)
    K[3+c4, 4+c4] += E45*N4**2*detJ*wij + N4x*(D16*N4x*detJ*wij + D66*N4y*detJ*wij) + N4y*(D12*N4x*detJ*wij + D26*N4y*detJ*wij)
    K[4+c4, 0+c1] += N1x*(B12*N4y*detJ*wij + B16*N4x*detJ*wij) + N1y*(B26*N4y*detJ*wij + B66*N4x*detJ*wij)
    K[4+c4, 1+c1] += N1x*(B26*N4y*detJ*wij + B66*N4x*detJ*wij) + N1y*(B22*N4y*detJ*wij + B26*N4x*detJ*wij)
    K[4+c4, 2+c1] += E44*N1y*N4*detJ*wij + E45*N1x*N4*detJ*wij
    K[4+c4, 3+c1] += E45*N1*N4*detJ*wij + N1x*(D12*N4y*detJ*wij + D16*N4x*detJ*wij) + N1y*(D26*N4y*detJ*wij + D66*N4x*detJ*wij)
    K[4+c4, 4+c1] += E44*N1*N4*detJ*wij + N1x*(D26*N4y*detJ*wij + D66*N4x*detJ*wij) + N1y*(D22*N4y*detJ*wij + D26*N4x*detJ*wij)
    K[4+c4, 0+c2] += N2x*(B12*N4y*detJ*wij + B16*N4x*detJ*wij) + N2y*(B26*N4y*detJ*wij + B66*N4x*detJ*wij)
    K[4+c4, 1+c2] += N2x*(B26*N4y*detJ*wij + B66*N4x*detJ*wij) + N2y*(B22*N4y*detJ*wij + B26*N4x*detJ*wij)
    K[4+c4, 2+c2] += E44*N2y*N4*detJ*wij + E45*N2x*N4*detJ*wij
    K[4+c4, 3+c2] += E45*N2*N4*detJ*wij + N2x*(D12*N4y*detJ*wij + D16*N4x*detJ*wij) + N2y*(D26*N4y*detJ*wij + D66*N4x*detJ*wij)
    K[4+c4, 4+c2] += E44*N2*N4*detJ*wij + N2x*(D26*N4y*detJ*wij + D66*N4x*detJ*wij) + N2y*(D22*N4y*detJ*wij + D26*N4x*detJ*wij)
    K[4+c4, 0+c3] += N3x*(B12*N4y*detJ*wij + B16*N4x*detJ*wij) + N3y*(B26*N4y*detJ*wij + B66*N4x*detJ*wij)
    K[4+c4, 1+c3] += N3x*(B26*N4y*detJ*wij + B66*N4x*detJ*wij) + N3y*(B22*N4y*detJ*wij + B26*N4x*detJ*wij)
    K[4+c4, 2+c3] += E44*N3y*N4*detJ*wij + E45*N3x*N4*detJ*wij
    K[4+c4, 3+c3] += E45*N3*N4*detJ*wij + N3x*(D12*N4y*detJ*wij + D16*N4x*detJ*wij) + N3y*(D26*N4y*detJ*wij + D66*N4x*detJ*wij)
    K[4+c4, 4+c3] += E44*N3*N4*detJ*wij + N3x*(D26*N4y*detJ*wij + D66*N4x*detJ*wij) + N3y*(D22*N4y*detJ*wij + D26*N4x*detJ*wij)
    K[4+c4, 0+c4] += N4x*(B12*N4y*detJ*wij + B16*N4x*detJ*wij) + N4y*(B26*N4y*detJ*wij + B66*N4x*detJ*wij)
    K[4+c4, 1+c4] += N4x*(B26*N4y*detJ*wij + B66*N4x*detJ*wij) + N4y*(B22*N4y*detJ*wij + B26*N4x*detJ*wij)
    K[4+c4, 2+c4] += E44*N4*N4y*detJ*wij + E45*N4*N4x*detJ*wij
    K[4+c4, 3+c4] += E45*N4**2*detJ*wij + N4x*(D12*N4y*detJ*wij + D16*N4x*detJ*wij) + N4y*(D26*N4y*detJ*wij + D66*N4x*detJ*wij)
    K[4+c4, 4+c4] += E44*N4**2*detJ*wij + N4x*(D26*N4y*detJ*wij + D66*N4x*detJ*wij) + N4y*(D22*N4y*detJ*wij + D26*N4x*detJ*wij)


def update_M(quad, nid_pos, ncoords, M):
    """Update global M with Me from a quad element

    Properties
    ----------
    quad : `.Quad` object
        The quad element being added to M
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    M : np.array
        Global mass matrix

    """
    pos1 = nid_pos[quad.n1]
    pos2 = nid_pos[quad.n2]
    pos3 = nid_pos[quad.n3]
    pos4 = nid_pos[quad.n4]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    x3, y3 = ncoords[pos3]
    x4, y4 = ncoords[pos4]

    rho = quad.rho
    h = quad.h

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3
    c4 = DOF*pos4

    weights_points =[[1., -0.577350269189625764509148780501957455647601751270126876],
                     [1., +0.577350269189625764509148780501957455647601751270126876]]
    weights_points =[[2., 0.]]
    weights_points =[[8/9, 0.],
                     [5/9, -(3/5)**0.5 ],
                     [5/9, +(3/5)**0.5 ],
                     ]

    for wi, xi in weights_points:
        for wj, eta in weights_points:
            wij = wi*wj
            detJ = (-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16 - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16
            N1 = eta*xi/4 - eta/4 - xi/4 + 1/4
            N2 = -eta*xi/4 - eta/4 + xi/4 + 1/4
            N3 = eta*xi/4 + eta/4 + xi/4 + 1/4
            N4 = -eta*xi/4 + eta/4 - xi/4 + 1/4

            M[0+c1, 0+c1] += N1**2*detJ*h*rho*wij
            M[0+c1, 0+c2] += N1*N2*detJ*h*rho*wij
            M[0+c1, 0+c3] += N1*N3*detJ*h*rho*wij
            M[0+c1, 0+c4] += N1*N4*detJ*h*rho*wij
            M[1+c1, 1+c1] += N1**2*detJ*h*rho*wij
            M[1+c1, 1+c2] += N1*N2*detJ*h*rho*wij
            M[1+c1, 1+c3] += N1*N3*detJ*h*rho*wij
            M[1+c1, 1+c4] += N1*N4*detJ*h*rho*wij
            M[2+c1, 2+c1] += N1**2*detJ*h*rho*wij
            M[2+c1, 2+c2] += N1*N2*detJ*h*rho*wij
            M[2+c1, 2+c3] += N1*N3*detJ*h*rho*wij
            M[2+c1, 2+c4] += N1*N4*detJ*h*rho*wij
            M[3+c1, 3+c1] += N1**2*detJ*h**3*rho*wij/12
            M[3+c1, 3+c2] += N1*N2*detJ*h**3*rho*wij/12
            M[3+c1, 3+c3] += N1*N3*detJ*h**3*rho*wij/12
            M[3+c1, 3+c4] += N1*N4*detJ*h**3*rho*wij/12
            M[4+c1, 4+c1] += N1**2*detJ*h**3*rho*wij/12
            M[4+c1, 4+c2] += N1*N2*detJ*h**3*rho*wij/12
            M[4+c1, 4+c3] += N1*N3*detJ*h**3*rho*wij/12
            M[4+c1, 4+c4] += N1*N4*detJ*h**3*rho*wij/12
            M[0+c2, 0+c1] += N1*N2*detJ*h*rho*wij
            M[0+c2, 0+c2] += N2**2*detJ*h*rho*wij
            M[0+c2, 0+c3] += N2*N3*detJ*h*rho*wij
            M[0+c2, 0+c4] += N2*N4*detJ*h*rho*wij
            M[1+c2, 1+c1] += N1*N2*detJ*h*rho*wij
            M[1+c2, 1+c2] += N2**2*detJ*h*rho*wij
            M[1+c2, 1+c3] += N2*N3*detJ*h*rho*wij
            M[1+c2, 1+c4] += N2*N4*detJ*h*rho*wij
            M[2+c2, 2+c1] += N1*N2*detJ*h*rho*wij
            M[2+c2, 2+c2] += N2**2*detJ*h*rho*wij
            M[2+c2, 2+c3] += N2*N3*detJ*h*rho*wij
            M[2+c2, 2+c4] += N2*N4*detJ*h*rho*wij
            M[3+c2, 3+c1] += N1*N2*detJ*h**3*rho*wij/12
            M[3+c2, 3+c2] += N2**2*detJ*h**3*rho*wij/12
            M[3+c2, 3+c3] += N2*N3*detJ*h**3*rho*wij/12
            M[3+c2, 3+c4] += N2*N4*detJ*h**3*rho*wij/12
            M[4+c2, 4+c1] += N1*N2*detJ*h**3*rho*wij/12
            M[4+c2, 4+c2] += N2**2*detJ*h**3*rho*wij/12
            M[4+c2, 4+c3] += N2*N3*detJ*h**3*rho*wij/12
            M[4+c2, 4+c4] += N2*N4*detJ*h**3*rho*wij/12
            M[0+c3, 0+c1] += N1*N3*detJ*h*rho*wij
            M[0+c3, 0+c2] += N2*N3*detJ*h*rho*wij
            M[0+c3, 0+c3] += N3**2*detJ*h*rho*wij
            M[0+c3, 0+c4] += N3*N4*detJ*h*rho*wij
            M[1+c3, 1+c1] += N1*N3*detJ*h*rho*wij
            M[1+c3, 1+c2] += N2*N3*detJ*h*rho*wij
            M[1+c3, 1+c3] += N3**2*detJ*h*rho*wij
            M[1+c3, 1+c4] += N3*N4*detJ*h*rho*wij
            M[2+c3, 2+c1] += N1*N3*detJ*h*rho*wij
            M[2+c3, 2+c2] += N2*N3*detJ*h*rho*wij
            M[2+c3, 2+c3] += N3**2*detJ*h*rho*wij
            M[2+c3, 2+c4] += N3*N4*detJ*h*rho*wij
            M[3+c3, 3+c1] += N1*N3*detJ*h**3*rho*wij/12
            M[3+c3, 3+c2] += N2*N3*detJ*h**3*rho*wij/12
            M[3+c3, 3+c3] += N3**2*detJ*h**3*rho*wij/12
            M[3+c3, 3+c4] += N3*N4*detJ*h**3*rho*wij/12
            M[4+c3, 4+c1] += N1*N3*detJ*h**3*rho*wij/12
            M[4+c3, 4+c2] += N2*N3*detJ*h**3*rho*wij/12
            M[4+c3, 4+c3] += N3**2*detJ*h**3*rho*wij/12
            M[4+c3, 4+c4] += N3*N4*detJ*h**3*rho*wij/12
            M[0+c4, 0+c1] += N1*N4*detJ*h*rho*wij
            M[0+c4, 0+c2] += N2*N4*detJ*h*rho*wij
            M[0+c4, 0+c3] += N3*N4*detJ*h*rho*wij
            M[0+c4, 0+c4] += N4**2*detJ*h*rho*wij
            M[1+c4, 1+c1] += N1*N4*detJ*h*rho*wij
            M[1+c4, 1+c2] += N2*N4*detJ*h*rho*wij
            M[1+c4, 1+c3] += N3*N4*detJ*h*rho*wij
            M[1+c4, 1+c4] += N4**2*detJ*h*rho*wij
            M[2+c4, 2+c1] += N1*N4*detJ*h*rho*wij
            M[2+c4, 2+c2] += N2*N4*detJ*h*rho*wij
            M[2+c4, 2+c3] += N3*N4*detJ*h*rho*wij
            M[2+c4, 2+c4] += N4**2*detJ*h*rho*wij
            M[3+c4, 3+c1] += N1*N4*detJ*h**3*rho*wij/12
            M[3+c4, 3+c2] += N2*N4*detJ*h**3*rho*wij/12
            M[3+c4, 3+c3] += N3*N4*detJ*h**3*rho*wij/12
            M[3+c4, 3+c4] += N4**2*detJ*h**3*rho*wij/12
            M[4+c4, 4+c1] += N1*N4*detJ*h**3*rho*wij/12
            M[4+c4, 4+c2] += N2*N4*detJ*h**3*rho*wij/12
            M[4+c4, 4+c3] += N3*N4*detJ*h**3*rho*wij/12
            M[4+c4, 4+c4] += N4**2*detJ*h**3*rho*wij/12

