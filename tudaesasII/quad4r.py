import numpy as np

DOF = 5

class Quad4R(object):
    """Reissner-Mindlin plate element with reduced integration

    Formulated based on the first-order shear deformation theory for plates

    Reduced integration is achieved by having only 1 integration point at the
    center, while integrating the stiffness matrix. This removes shear locking.

    An artificial hour-glass stiffness is used, according to Brockman 1987

    https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620241208

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

    A = (np.cross([x2 - x1, y2 - y1], [x4 - x1, y4 - y1])/2 +
         np.cross([x4 - x3, y4 - y3], [x2 - x3, y2 - y3])/2)

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3
    c4 = DOF*pos4

    h = quad.h

    #NOTE reduced integration to remove shear locking
    xi = eta = 0
    wi = wj = 2.

    wij = wi*wj
    detJ = (-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16 - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16
    j11 = 2*(-xi*y1 + xi*y2 - xi*y3 + xi*y4 + y1 + y2 - y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
    j12 = 4*(-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
    j21 = 4*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
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
    N1xy = j11*j22/4 + j12*j21/4
    N2xy = -j11*j22/4 - j12*j21/4
    N3xy = j11*j22/4 + j12*j21/4
    N4xy = -j11*j22/4 - j12*j21/4
    gamma1 = N1xy
    gamma2 = N2xy
    gamma3 = N3xy
    gamma4 = N4xy
    Eu = 0.1*A11/(1 + 1/A)
    Ev = 0.1*A11/(1 + 1/A)
    Ephix = 12*0.1*D11/(1 + 1/A)
    Ephiy = 12*0.1*D22/(1 + 1/A)
    Ew = (Ephix + Ephiy)/2

    K[0+c1, 0+c1] += detJ*wij*(Eu*gamma1**2 + N1x*(A11*N1x + A16*N1y) + N1y*(A16*N1x + A66*N1y))
    K[0+c1, 1+c1] += detJ*wij*(N1x*(A16*N1x + A66*N1y) + N1y*(A12*N1x + A26*N1y))
    K[0+c1, 3+c1] += detJ*wij*(N1x*(B11*N1x + B16*N1y) + N1y*(B16*N1x + B66*N1y))
    K[0+c1, 4+c1] += detJ*wij*(N1x*(B16*N1x + B66*N1y) + N1y*(B12*N1x + B26*N1y))
    K[0+c1, 0+c2] += detJ*wij*(Eu*gamma1*gamma2 + N2x*(A11*N1x + A16*N1y) + N2y*(A16*N1x + A66*N1y))
    K[0+c1, 1+c2] += detJ*wij*(N2x*(A16*N1x + A66*N1y) + N2y*(A12*N1x + A26*N1y))
    K[0+c1, 3+c2] += detJ*wij*(N2x*(B11*N1x + B16*N1y) + N2y*(B16*N1x + B66*N1y))
    K[0+c1, 4+c2] += detJ*wij*(N2x*(B16*N1x + B66*N1y) + N2y*(B12*N1x + B26*N1y))
    K[0+c1, 0+c3] += detJ*wij*(Eu*gamma1*gamma3 + N3x*(A11*N1x + A16*N1y) + N3y*(A16*N1x + A66*N1y))
    K[0+c1, 1+c3] += detJ*wij*(N3x*(A16*N1x + A66*N1y) + N3y*(A12*N1x + A26*N1y))
    K[0+c1, 3+c3] += detJ*wij*(N3x*(B11*N1x + B16*N1y) + N3y*(B16*N1x + B66*N1y))
    K[0+c1, 4+c3] += detJ*wij*(N3x*(B16*N1x + B66*N1y) + N3y*(B12*N1x + B26*N1y))
    K[0+c1, 0+c4] += detJ*wij*(Eu*gamma1*gamma4 + N4x*(A11*N1x + A16*N1y) + N4y*(A16*N1x + A66*N1y))
    K[0+c1, 1+c4] += detJ*wij*(N4x*(A16*N1x + A66*N1y) + N4y*(A12*N1x + A26*N1y))
    K[0+c1, 3+c4] += detJ*wij*(N4x*(B11*N1x + B16*N1y) + N4y*(B16*N1x + B66*N1y))
    K[0+c1, 4+c4] += detJ*wij*(N4x*(B16*N1x + B66*N1y) + N4y*(B12*N1x + B26*N1y))
    K[1+c1, 0+c1] += detJ*wij*(N1x*(A12*N1y + A16*N1x) + N1y*(A26*N1y + A66*N1x))
    K[1+c1, 1+c1] += detJ*wij*(Ev*gamma1**2 + N1x*(A26*N1y + A66*N1x) + N1y*(A22*N1y + A26*N1x))
    K[1+c1, 3+c1] += detJ*wij*(N1x*(B12*N1y + B16*N1x) + N1y*(B26*N1y + B66*N1x))
    K[1+c1, 4+c1] += detJ*wij*(N1x*(B26*N1y + B66*N1x) + N1y*(B22*N1y + B26*N1x))
    K[1+c1, 0+c2] += detJ*wij*(N2x*(A12*N1y + A16*N1x) + N2y*(A26*N1y + A66*N1x))
    K[1+c1, 1+c2] += detJ*wij*(Ev*gamma1*gamma2 + N2x*(A26*N1y + A66*N1x) + N2y*(A22*N1y + A26*N1x))
    K[1+c1, 3+c2] += detJ*wij*(N2x*(B12*N1y + B16*N1x) + N2y*(B26*N1y + B66*N1x))
    K[1+c1, 4+c2] += detJ*wij*(N2x*(B26*N1y + B66*N1x) + N2y*(B22*N1y + B26*N1x))
    K[1+c1, 0+c3] += detJ*wij*(N3x*(A12*N1y + A16*N1x) + N3y*(A26*N1y + A66*N1x))
    K[1+c1, 1+c3] += detJ*wij*(Ev*gamma1*gamma3 + N3x*(A26*N1y + A66*N1x) + N3y*(A22*N1y + A26*N1x))
    K[1+c1, 3+c3] += detJ*wij*(N3x*(B12*N1y + B16*N1x) + N3y*(B26*N1y + B66*N1x))
    K[1+c1, 4+c3] += detJ*wij*(N3x*(B26*N1y + B66*N1x) + N3y*(B22*N1y + B26*N1x))
    K[1+c1, 0+c4] += detJ*wij*(N4x*(A12*N1y + A16*N1x) + N4y*(A26*N1y + A66*N1x))
    K[1+c1, 1+c4] += detJ*wij*(Ev*gamma1*gamma4 + N4x*(A26*N1y + A66*N1x) + N4y*(A22*N1y + A26*N1x))
    K[1+c1, 3+c4] += detJ*wij*(N4x*(B12*N1y + B16*N1x) + N4y*(B26*N1y + B66*N1x))
    K[1+c1, 4+c4] += detJ*wij*(N4x*(B26*N1y + B66*N1x) + N4y*(B22*N1y + B26*N1x))
    K[2+c1, 2+c1] += detJ*wij*(Ew*gamma1**2 + N1x*(E45*N1y + E55*N1x) + N1y*(E44*N1y + E45*N1x))
    K[2+c1, 3+c1] += N1*detJ*wij*(E45*N1y + E55*N1x)
    K[2+c1, 4+c1] += N1*detJ*wij*(E44*N1y + E45*N1x)
    K[2+c1, 2+c2] += detJ*wij*(Ew*gamma1*gamma2 + N2x*(E45*N1y + E55*N1x) + N2y*(E44*N1y + E45*N1x))
    K[2+c1, 3+c2] += N2*detJ*wij*(E45*N1y + E55*N1x)
    K[2+c1, 4+c2] += N2*detJ*wij*(E44*N1y + E45*N1x)
    K[2+c1, 2+c3] += detJ*wij*(Ew*gamma1*gamma3 + N3x*(E45*N1y + E55*N1x) + N3y*(E44*N1y + E45*N1x))
    K[2+c1, 3+c3] += N3*detJ*wij*(E45*N1y + E55*N1x)
    K[2+c1, 4+c3] += N3*detJ*wij*(E44*N1y + E45*N1x)
    K[2+c1, 2+c4] += detJ*wij*(Ew*gamma1*gamma4 + N4x*(E45*N1y + E55*N1x) + N4y*(E44*N1y + E45*N1x))
    K[2+c1, 3+c4] += N4*detJ*wij*(E45*N1y + E55*N1x)
    K[2+c1, 4+c4] += N4*detJ*wij*(E44*N1y + E45*N1x)
    K[3+c1, 0+c1] += detJ*wij*(N1x*(B11*N1x + B16*N1y) + N1y*(B16*N1x + B66*N1y))
    K[3+c1, 1+c1] += detJ*wij*(N1x*(B16*N1x + B66*N1y) + N1y*(B12*N1x + B26*N1y))
    K[3+c1, 2+c1] += detJ*wij*(E45*N1*N1y + E55*N1*N1x)
    K[3+c1, 3+c1] += detJ*wij*(E55*N1**2 + Ephix*gamma1**2 + N1x*(D11*N1x + D16*N1y) + N1y*(D16*N1x + D66*N1y))
    K[3+c1, 4+c1] += detJ*wij*(E45*N1**2 + N1x*(D16*N1x + D66*N1y) + N1y*(D12*N1x + D26*N1y))
    K[3+c1, 0+c2] += detJ*wij*(N2x*(B11*N1x + B16*N1y) + N2y*(B16*N1x + B66*N1y))
    K[3+c1, 1+c2] += detJ*wij*(N2x*(B16*N1x + B66*N1y) + N2y*(B12*N1x + B26*N1y))
    K[3+c1, 2+c2] += detJ*wij*(E45*N1*N2y + E55*N1*N2x)
    K[3+c1, 3+c2] += detJ*wij*(E55*N1*N2 + Ephix*gamma1*gamma2 + N2x*(D11*N1x + D16*N1y) + N2y*(D16*N1x + D66*N1y))
    K[3+c1, 4+c2] += detJ*wij*(E45*N1*N2 + N2x*(D16*N1x + D66*N1y) + N2y*(D12*N1x + D26*N1y))
    K[3+c1, 0+c3] += detJ*wij*(N3x*(B11*N1x + B16*N1y) + N3y*(B16*N1x + B66*N1y))
    K[3+c1, 1+c3] += detJ*wij*(N3x*(B16*N1x + B66*N1y) + N3y*(B12*N1x + B26*N1y))
    K[3+c1, 2+c3] += detJ*wij*(E45*N1*N3y + E55*N1*N3x)
    K[3+c1, 3+c3] += detJ*wij*(E55*N1*N3 + Ephix*gamma1*gamma3 + N3x*(D11*N1x + D16*N1y) + N3y*(D16*N1x + D66*N1y))
    K[3+c1, 4+c3] += detJ*wij*(E45*N1*N3 + N3x*(D16*N1x + D66*N1y) + N3y*(D12*N1x + D26*N1y))
    K[3+c1, 0+c4] += detJ*wij*(N4x*(B11*N1x + B16*N1y) + N4y*(B16*N1x + B66*N1y))
    K[3+c1, 1+c4] += detJ*wij*(N4x*(B16*N1x + B66*N1y) + N4y*(B12*N1x + B26*N1y))
    K[3+c1, 2+c4] += detJ*wij*(E45*N1*N4y + E55*N1*N4x)
    K[3+c1, 3+c4] += detJ*wij*(E55*N1*N4 + Ephix*gamma1*gamma4 + N4x*(D11*N1x + D16*N1y) + N4y*(D16*N1x + D66*N1y))
    K[3+c1, 4+c4] += detJ*wij*(E45*N1*N4 + N4x*(D16*N1x + D66*N1y) + N4y*(D12*N1x + D26*N1y))
    K[4+c1, 0+c1] += detJ*wij*(N1x*(B12*N1y + B16*N1x) + N1y*(B26*N1y + B66*N1x))
    K[4+c1, 1+c1] += detJ*wij*(N1x*(B26*N1y + B66*N1x) + N1y*(B22*N1y + B26*N1x))
    K[4+c1, 2+c1] += detJ*wij*(E44*N1*N1y + E45*N1*N1x)
    K[4+c1, 3+c1] += detJ*wij*(E45*N1**2 + N1x*(D12*N1y + D16*N1x) + N1y*(D26*N1y + D66*N1x))
    K[4+c1, 4+c1] += detJ*wij*(E44*N1**2 + Ephiy*gamma1**2 + N1x*(D26*N1y + D66*N1x) + N1y*(D22*N1y + D26*N1x))
    K[4+c1, 0+c2] += detJ*wij*(N2x*(B12*N1y + B16*N1x) + N2y*(B26*N1y + B66*N1x))
    K[4+c1, 1+c2] += detJ*wij*(N2x*(B26*N1y + B66*N1x) + N2y*(B22*N1y + B26*N1x))
    K[4+c1, 2+c2] += detJ*wij*(E44*N1*N2y + E45*N1*N2x)
    K[4+c1, 3+c2] += detJ*wij*(E45*N1*N2 + N2x*(D12*N1y + D16*N1x) + N2y*(D26*N1y + D66*N1x))
    K[4+c1, 4+c2] += detJ*wij*(E44*N1*N2 + Ephiy*gamma1*gamma2 + N2x*(D26*N1y + D66*N1x) + N2y*(D22*N1y + D26*N1x))
    K[4+c1, 0+c3] += detJ*wij*(N3x*(B12*N1y + B16*N1x) + N3y*(B26*N1y + B66*N1x))
    K[4+c1, 1+c3] += detJ*wij*(N3x*(B26*N1y + B66*N1x) + N3y*(B22*N1y + B26*N1x))
    K[4+c1, 2+c3] += detJ*wij*(E44*N1*N3y + E45*N1*N3x)
    K[4+c1, 3+c3] += detJ*wij*(E45*N1*N3 + N3x*(D12*N1y + D16*N1x) + N3y*(D26*N1y + D66*N1x))
    K[4+c1, 4+c3] += detJ*wij*(E44*N1*N3 + Ephiy*gamma1*gamma3 + N3x*(D26*N1y + D66*N1x) + N3y*(D22*N1y + D26*N1x))
    K[4+c1, 0+c4] += detJ*wij*(N4x*(B12*N1y + B16*N1x) + N4y*(B26*N1y + B66*N1x))
    K[4+c1, 1+c4] += detJ*wij*(N4x*(B26*N1y + B66*N1x) + N4y*(B22*N1y + B26*N1x))
    K[4+c1, 2+c4] += detJ*wij*(E44*N1*N4y + E45*N1*N4x)
    K[4+c1, 3+c4] += detJ*wij*(E45*N1*N4 + N4x*(D12*N1y + D16*N1x) + N4y*(D26*N1y + D66*N1x))
    K[4+c1, 4+c4] += detJ*wij*(E44*N1*N4 + Ephiy*gamma1*gamma4 + N4x*(D26*N1y + D66*N1x) + N4y*(D22*N1y + D26*N1x))
    K[0+c2, 0+c1] += detJ*wij*(Eu*gamma1*gamma2 + N1x*(A11*N2x + A16*N2y) + N1y*(A16*N2x + A66*N2y))
    K[0+c2, 1+c1] += detJ*wij*(N1x*(A16*N2x + A66*N2y) + N1y*(A12*N2x + A26*N2y))
    K[0+c2, 3+c1] += detJ*wij*(N1x*(B11*N2x + B16*N2y) + N1y*(B16*N2x + B66*N2y))
    K[0+c2, 4+c1] += detJ*wij*(N1x*(B16*N2x + B66*N2y) + N1y*(B12*N2x + B26*N2y))
    K[0+c2, 0+c2] += detJ*wij*(Eu*gamma2**2 + N2x*(A11*N2x + A16*N2y) + N2y*(A16*N2x + A66*N2y))
    K[0+c2, 1+c2] += detJ*wij*(N2x*(A16*N2x + A66*N2y) + N2y*(A12*N2x + A26*N2y))
    K[0+c2, 3+c2] += detJ*wij*(N2x*(B11*N2x + B16*N2y) + N2y*(B16*N2x + B66*N2y))
    K[0+c2, 4+c2] += detJ*wij*(N2x*(B16*N2x + B66*N2y) + N2y*(B12*N2x + B26*N2y))
    K[0+c2, 0+c3] += detJ*wij*(Eu*gamma2*gamma3 + N3x*(A11*N2x + A16*N2y) + N3y*(A16*N2x + A66*N2y))
    K[0+c2, 1+c3] += detJ*wij*(N3x*(A16*N2x + A66*N2y) + N3y*(A12*N2x + A26*N2y))
    K[0+c2, 3+c3] += detJ*wij*(N3x*(B11*N2x + B16*N2y) + N3y*(B16*N2x + B66*N2y))
    K[0+c2, 4+c3] += detJ*wij*(N3x*(B16*N2x + B66*N2y) + N3y*(B12*N2x + B26*N2y))
    K[0+c2, 0+c4] += detJ*wij*(Eu*gamma2*gamma4 + N4x*(A11*N2x + A16*N2y) + N4y*(A16*N2x + A66*N2y))
    K[0+c2, 1+c4] += detJ*wij*(N4x*(A16*N2x + A66*N2y) + N4y*(A12*N2x + A26*N2y))
    K[0+c2, 3+c4] += detJ*wij*(N4x*(B11*N2x + B16*N2y) + N4y*(B16*N2x + B66*N2y))
    K[0+c2, 4+c4] += detJ*wij*(N4x*(B16*N2x + B66*N2y) + N4y*(B12*N2x + B26*N2y))
    K[1+c2, 0+c1] += detJ*wij*(N1x*(A12*N2y + A16*N2x) + N1y*(A26*N2y + A66*N2x))
    K[1+c2, 1+c1] += detJ*wij*(Ev*gamma1*gamma2 + N1x*(A26*N2y + A66*N2x) + N1y*(A22*N2y + A26*N2x))
    K[1+c2, 3+c1] += detJ*wij*(N1x*(B12*N2y + B16*N2x) + N1y*(B26*N2y + B66*N2x))
    K[1+c2, 4+c1] += detJ*wij*(N1x*(B26*N2y + B66*N2x) + N1y*(B22*N2y + B26*N2x))
    K[1+c2, 0+c2] += detJ*wij*(N2x*(A12*N2y + A16*N2x) + N2y*(A26*N2y + A66*N2x))
    K[1+c2, 1+c2] += detJ*wij*(Ev*gamma2**2 + N2x*(A26*N2y + A66*N2x) + N2y*(A22*N2y + A26*N2x))
    K[1+c2, 3+c2] += detJ*wij*(N2x*(B12*N2y + B16*N2x) + N2y*(B26*N2y + B66*N2x))
    K[1+c2, 4+c2] += detJ*wij*(N2x*(B26*N2y + B66*N2x) + N2y*(B22*N2y + B26*N2x))
    K[1+c2, 0+c3] += detJ*wij*(N3x*(A12*N2y + A16*N2x) + N3y*(A26*N2y + A66*N2x))
    K[1+c2, 1+c3] += detJ*wij*(Ev*gamma2*gamma3 + N3x*(A26*N2y + A66*N2x) + N3y*(A22*N2y + A26*N2x))
    K[1+c2, 3+c3] += detJ*wij*(N3x*(B12*N2y + B16*N2x) + N3y*(B26*N2y + B66*N2x))
    K[1+c2, 4+c3] += detJ*wij*(N3x*(B26*N2y + B66*N2x) + N3y*(B22*N2y + B26*N2x))
    K[1+c2, 0+c4] += detJ*wij*(N4x*(A12*N2y + A16*N2x) + N4y*(A26*N2y + A66*N2x))
    K[1+c2, 1+c4] += detJ*wij*(Ev*gamma2*gamma4 + N4x*(A26*N2y + A66*N2x) + N4y*(A22*N2y + A26*N2x))
    K[1+c2, 3+c4] += detJ*wij*(N4x*(B12*N2y + B16*N2x) + N4y*(B26*N2y + B66*N2x))
    K[1+c2, 4+c4] += detJ*wij*(N4x*(B26*N2y + B66*N2x) + N4y*(B22*N2y + B26*N2x))
    K[2+c2, 2+c1] += detJ*wij*(Ew*gamma1*gamma2 + N1x*(E45*N2y + E55*N2x) + N1y*(E44*N2y + E45*N2x))
    K[2+c2, 3+c1] += N1*detJ*wij*(E45*N2y + E55*N2x)
    K[2+c2, 4+c1] += N1*detJ*wij*(E44*N2y + E45*N2x)
    K[2+c2, 2+c2] += detJ*wij*(Ew*gamma2**2 + N2x*(E45*N2y + E55*N2x) + N2y*(E44*N2y + E45*N2x))
    K[2+c2, 3+c2] += N2*detJ*wij*(E45*N2y + E55*N2x)
    K[2+c2, 4+c2] += N2*detJ*wij*(E44*N2y + E45*N2x)
    K[2+c2, 2+c3] += detJ*wij*(Ew*gamma2*gamma3 + N3x*(E45*N2y + E55*N2x) + N3y*(E44*N2y + E45*N2x))
    K[2+c2, 3+c3] += N3*detJ*wij*(E45*N2y + E55*N2x)
    K[2+c2, 4+c3] += N3*detJ*wij*(E44*N2y + E45*N2x)
    K[2+c2, 2+c4] += detJ*wij*(Ew*gamma2*gamma4 + N4x*(E45*N2y + E55*N2x) + N4y*(E44*N2y + E45*N2x))
    K[2+c2, 3+c4] += N4*detJ*wij*(E45*N2y + E55*N2x)
    K[2+c2, 4+c4] += N4*detJ*wij*(E44*N2y + E45*N2x)
    K[3+c2, 0+c1] += detJ*wij*(N1x*(B11*N2x + B16*N2y) + N1y*(B16*N2x + B66*N2y))
    K[3+c2, 1+c1] += detJ*wij*(N1x*(B16*N2x + B66*N2y) + N1y*(B12*N2x + B26*N2y))
    K[3+c2, 2+c1] += detJ*wij*(E45*N1y*N2 + E55*N1x*N2)
    K[3+c2, 3+c1] += detJ*wij*(E55*N1*N2 + Ephix*gamma1*gamma2 + N1x*(D11*N2x + D16*N2y) + N1y*(D16*N2x + D66*N2y))
    K[3+c2, 4+c1] += detJ*wij*(E45*N1*N2 + N1x*(D16*N2x + D66*N2y) + N1y*(D12*N2x + D26*N2y))
    K[3+c2, 0+c2] += detJ*wij*(N2x*(B11*N2x + B16*N2y) + N2y*(B16*N2x + B66*N2y))
    K[3+c2, 1+c2] += detJ*wij*(N2x*(B16*N2x + B66*N2y) + N2y*(B12*N2x + B26*N2y))
    K[3+c2, 2+c2] += detJ*wij*(E45*N2*N2y + E55*N2*N2x)
    K[3+c2, 3+c2] += detJ*wij*(E55*N2**2 + Ephix*gamma2**2 + N2x*(D11*N2x + D16*N2y) + N2y*(D16*N2x + D66*N2y))
    K[3+c2, 4+c2] += detJ*wij*(E45*N2**2 + N2x*(D16*N2x + D66*N2y) + N2y*(D12*N2x + D26*N2y))
    K[3+c2, 0+c3] += detJ*wij*(N3x*(B11*N2x + B16*N2y) + N3y*(B16*N2x + B66*N2y))
    K[3+c2, 1+c3] += detJ*wij*(N3x*(B16*N2x + B66*N2y) + N3y*(B12*N2x + B26*N2y))
    K[3+c2, 2+c3] += detJ*wij*(E45*N2*N3y + E55*N2*N3x)
    K[3+c2, 3+c3] += detJ*wij*(E55*N2*N3 + Ephix*gamma2*gamma3 + N3x*(D11*N2x + D16*N2y) + N3y*(D16*N2x + D66*N2y))
    K[3+c2, 4+c3] += detJ*wij*(E45*N2*N3 + N3x*(D16*N2x + D66*N2y) + N3y*(D12*N2x + D26*N2y))
    K[3+c2, 0+c4] += detJ*wij*(N4x*(B11*N2x + B16*N2y) + N4y*(B16*N2x + B66*N2y))
    K[3+c2, 1+c4] += detJ*wij*(N4x*(B16*N2x + B66*N2y) + N4y*(B12*N2x + B26*N2y))
    K[3+c2, 2+c4] += detJ*wij*(E45*N2*N4y + E55*N2*N4x)
    K[3+c2, 3+c4] += detJ*wij*(E55*N2*N4 + Ephix*gamma2*gamma4 + N4x*(D11*N2x + D16*N2y) + N4y*(D16*N2x + D66*N2y))
    K[3+c2, 4+c4] += detJ*wij*(E45*N2*N4 + N4x*(D16*N2x + D66*N2y) + N4y*(D12*N2x + D26*N2y))
    K[4+c2, 0+c1] += detJ*wij*(N1x*(B12*N2y + B16*N2x) + N1y*(B26*N2y + B66*N2x))
    K[4+c2, 1+c1] += detJ*wij*(N1x*(B26*N2y + B66*N2x) + N1y*(B22*N2y + B26*N2x))
    K[4+c2, 2+c1] += detJ*wij*(E44*N1y*N2 + E45*N1x*N2)
    K[4+c2, 3+c1] += detJ*wij*(E45*N1*N2 + N1x*(D12*N2y + D16*N2x) + N1y*(D26*N2y + D66*N2x))
    K[4+c2, 4+c1] += detJ*wij*(E44*N1*N2 + Ephiy*gamma1*gamma2 + N1x*(D26*N2y + D66*N2x) + N1y*(D22*N2y + D26*N2x))
    K[4+c2, 0+c2] += detJ*wij*(N2x*(B12*N2y + B16*N2x) + N2y*(B26*N2y + B66*N2x))
    K[4+c2, 1+c2] += detJ*wij*(N2x*(B26*N2y + B66*N2x) + N2y*(B22*N2y + B26*N2x))
    K[4+c2, 2+c2] += detJ*wij*(E44*N2*N2y + E45*N2*N2x)
    K[4+c2, 3+c2] += detJ*wij*(E45*N2**2 + N2x*(D12*N2y + D16*N2x) + N2y*(D26*N2y + D66*N2x))
    K[4+c2, 4+c2] += detJ*wij*(E44*N2**2 + Ephiy*gamma2**2 + N2x*(D26*N2y + D66*N2x) + N2y*(D22*N2y + D26*N2x))
    K[4+c2, 0+c3] += detJ*wij*(N3x*(B12*N2y + B16*N2x) + N3y*(B26*N2y + B66*N2x))
    K[4+c2, 1+c3] += detJ*wij*(N3x*(B26*N2y + B66*N2x) + N3y*(B22*N2y + B26*N2x))
    K[4+c2, 2+c3] += detJ*wij*(E44*N2*N3y + E45*N2*N3x)
    K[4+c2, 3+c3] += detJ*wij*(E45*N2*N3 + N3x*(D12*N2y + D16*N2x) + N3y*(D26*N2y + D66*N2x))
    K[4+c2, 4+c3] += detJ*wij*(E44*N2*N3 + Ephiy*gamma2*gamma3 + N3x*(D26*N2y + D66*N2x) + N3y*(D22*N2y + D26*N2x))
    K[4+c2, 0+c4] += detJ*wij*(N4x*(B12*N2y + B16*N2x) + N4y*(B26*N2y + B66*N2x))
    K[4+c2, 1+c4] += detJ*wij*(N4x*(B26*N2y + B66*N2x) + N4y*(B22*N2y + B26*N2x))
    K[4+c2, 2+c4] += detJ*wij*(E44*N2*N4y + E45*N2*N4x)
    K[4+c2, 3+c4] += detJ*wij*(E45*N2*N4 + N4x*(D12*N2y + D16*N2x) + N4y*(D26*N2y + D66*N2x))
    K[4+c2, 4+c4] += detJ*wij*(E44*N2*N4 + Ephiy*gamma2*gamma4 + N4x*(D26*N2y + D66*N2x) + N4y*(D22*N2y + D26*N2x))
    K[0+c3, 0+c1] += detJ*wij*(Eu*gamma1*gamma3 + N1x*(A11*N3x + A16*N3y) + N1y*(A16*N3x + A66*N3y))
    K[0+c3, 1+c1] += detJ*wij*(N1x*(A16*N3x + A66*N3y) + N1y*(A12*N3x + A26*N3y))
    K[0+c3, 3+c1] += detJ*wij*(N1x*(B11*N3x + B16*N3y) + N1y*(B16*N3x + B66*N3y))
    K[0+c3, 4+c1] += detJ*wij*(N1x*(B16*N3x + B66*N3y) + N1y*(B12*N3x + B26*N3y))
    K[0+c3, 0+c2] += detJ*wij*(Eu*gamma2*gamma3 + N2x*(A11*N3x + A16*N3y) + N2y*(A16*N3x + A66*N3y))
    K[0+c3, 1+c2] += detJ*wij*(N2x*(A16*N3x + A66*N3y) + N2y*(A12*N3x + A26*N3y))
    K[0+c3, 3+c2] += detJ*wij*(N2x*(B11*N3x + B16*N3y) + N2y*(B16*N3x + B66*N3y))
    K[0+c3, 4+c2] += detJ*wij*(N2x*(B16*N3x + B66*N3y) + N2y*(B12*N3x + B26*N3y))
    K[0+c3, 0+c3] += detJ*wij*(Eu*gamma3**2 + N3x*(A11*N3x + A16*N3y) + N3y*(A16*N3x + A66*N3y))
    K[0+c3, 1+c3] += detJ*wij*(N3x*(A16*N3x + A66*N3y) + N3y*(A12*N3x + A26*N3y))
    K[0+c3, 3+c3] += detJ*wij*(N3x*(B11*N3x + B16*N3y) + N3y*(B16*N3x + B66*N3y))
    K[0+c3, 4+c3] += detJ*wij*(N3x*(B16*N3x + B66*N3y) + N3y*(B12*N3x + B26*N3y))
    K[0+c3, 0+c4] += detJ*wij*(Eu*gamma3*gamma4 + N4x*(A11*N3x + A16*N3y) + N4y*(A16*N3x + A66*N3y))
    K[0+c3, 1+c4] += detJ*wij*(N4x*(A16*N3x + A66*N3y) + N4y*(A12*N3x + A26*N3y))
    K[0+c3, 3+c4] += detJ*wij*(N4x*(B11*N3x + B16*N3y) + N4y*(B16*N3x + B66*N3y))
    K[0+c3, 4+c4] += detJ*wij*(N4x*(B16*N3x + B66*N3y) + N4y*(B12*N3x + B26*N3y))
    K[1+c3, 0+c1] += detJ*wij*(N1x*(A12*N3y + A16*N3x) + N1y*(A26*N3y + A66*N3x))
    K[1+c3, 1+c1] += detJ*wij*(Ev*gamma1*gamma3 + N1x*(A26*N3y + A66*N3x) + N1y*(A22*N3y + A26*N3x))
    K[1+c3, 3+c1] += detJ*wij*(N1x*(B12*N3y + B16*N3x) + N1y*(B26*N3y + B66*N3x))
    K[1+c3, 4+c1] += detJ*wij*(N1x*(B26*N3y + B66*N3x) + N1y*(B22*N3y + B26*N3x))
    K[1+c3, 0+c2] += detJ*wij*(N2x*(A12*N3y + A16*N3x) + N2y*(A26*N3y + A66*N3x))
    K[1+c3, 1+c2] += detJ*wij*(Ev*gamma2*gamma3 + N2x*(A26*N3y + A66*N3x) + N2y*(A22*N3y + A26*N3x))
    K[1+c3, 3+c2] += detJ*wij*(N2x*(B12*N3y + B16*N3x) + N2y*(B26*N3y + B66*N3x))
    K[1+c3, 4+c2] += detJ*wij*(N2x*(B26*N3y + B66*N3x) + N2y*(B22*N3y + B26*N3x))
    K[1+c3, 0+c3] += detJ*wij*(N3x*(A12*N3y + A16*N3x) + N3y*(A26*N3y + A66*N3x))
    K[1+c3, 1+c3] += detJ*wij*(Ev*gamma3**2 + N3x*(A26*N3y + A66*N3x) + N3y*(A22*N3y + A26*N3x))
    K[1+c3, 3+c3] += detJ*wij*(N3x*(B12*N3y + B16*N3x) + N3y*(B26*N3y + B66*N3x))
    K[1+c3, 4+c3] += detJ*wij*(N3x*(B26*N3y + B66*N3x) + N3y*(B22*N3y + B26*N3x))
    K[1+c3, 0+c4] += detJ*wij*(N4x*(A12*N3y + A16*N3x) + N4y*(A26*N3y + A66*N3x))
    K[1+c3, 1+c4] += detJ*wij*(Ev*gamma3*gamma4 + N4x*(A26*N3y + A66*N3x) + N4y*(A22*N3y + A26*N3x))
    K[1+c3, 3+c4] += detJ*wij*(N4x*(B12*N3y + B16*N3x) + N4y*(B26*N3y + B66*N3x))
    K[1+c3, 4+c4] += detJ*wij*(N4x*(B26*N3y + B66*N3x) + N4y*(B22*N3y + B26*N3x))
    K[2+c3, 2+c1] += detJ*wij*(Ew*gamma1*gamma3 + N1x*(E45*N3y + E55*N3x) + N1y*(E44*N3y + E45*N3x))
    K[2+c3, 3+c1] += N1*detJ*wij*(E45*N3y + E55*N3x)
    K[2+c3, 4+c1] += N1*detJ*wij*(E44*N3y + E45*N3x)
    K[2+c3, 2+c2] += detJ*wij*(Ew*gamma2*gamma3 + N2x*(E45*N3y + E55*N3x) + N2y*(E44*N3y + E45*N3x))
    K[2+c3, 3+c2] += N2*detJ*wij*(E45*N3y + E55*N3x)
    K[2+c3, 4+c2] += N2*detJ*wij*(E44*N3y + E45*N3x)
    K[2+c3, 2+c3] += detJ*wij*(Ew*gamma3**2 + N3x*(E45*N3y + E55*N3x) + N3y*(E44*N3y + E45*N3x))
    K[2+c3, 3+c3] += N3*detJ*wij*(E45*N3y + E55*N3x)
    K[2+c3, 4+c3] += N3*detJ*wij*(E44*N3y + E45*N3x)
    K[2+c3, 2+c4] += detJ*wij*(Ew*gamma3*gamma4 + N4x*(E45*N3y + E55*N3x) + N4y*(E44*N3y + E45*N3x))
    K[2+c3, 3+c4] += N4*detJ*wij*(E45*N3y + E55*N3x)
    K[2+c3, 4+c4] += N4*detJ*wij*(E44*N3y + E45*N3x)
    K[3+c3, 0+c1] += detJ*wij*(N1x*(B11*N3x + B16*N3y) + N1y*(B16*N3x + B66*N3y))
    K[3+c3, 1+c1] += detJ*wij*(N1x*(B16*N3x + B66*N3y) + N1y*(B12*N3x + B26*N3y))
    K[3+c3, 2+c1] += detJ*wij*(E45*N1y*N3 + E55*N1x*N3)
    K[3+c3, 3+c1] += detJ*wij*(E55*N1*N3 + Ephix*gamma1*gamma3 + N1x*(D11*N3x + D16*N3y) + N1y*(D16*N3x + D66*N3y))
    K[3+c3, 4+c1] += detJ*wij*(E45*N1*N3 + N1x*(D16*N3x + D66*N3y) + N1y*(D12*N3x + D26*N3y))
    K[3+c3, 0+c2] += detJ*wij*(N2x*(B11*N3x + B16*N3y) + N2y*(B16*N3x + B66*N3y))
    K[3+c3, 1+c2] += detJ*wij*(N2x*(B16*N3x + B66*N3y) + N2y*(B12*N3x + B26*N3y))
    K[3+c3, 2+c2] += detJ*wij*(E45*N2y*N3 + E55*N2x*N3)
    K[3+c3, 3+c2] += detJ*wij*(E55*N2*N3 + Ephix*gamma2*gamma3 + N2x*(D11*N3x + D16*N3y) + N2y*(D16*N3x + D66*N3y))
    K[3+c3, 4+c2] += detJ*wij*(E45*N2*N3 + N2x*(D16*N3x + D66*N3y) + N2y*(D12*N3x + D26*N3y))
    K[3+c3, 0+c3] += detJ*wij*(N3x*(B11*N3x + B16*N3y) + N3y*(B16*N3x + B66*N3y))
    K[3+c3, 1+c3] += detJ*wij*(N3x*(B16*N3x + B66*N3y) + N3y*(B12*N3x + B26*N3y))
    K[3+c3, 2+c3] += detJ*wij*(E45*N3*N3y + E55*N3*N3x)
    K[3+c3, 3+c3] += detJ*wij*(E55*N3**2 + Ephix*gamma3**2 + N3x*(D11*N3x + D16*N3y) + N3y*(D16*N3x + D66*N3y))
    K[3+c3, 4+c3] += detJ*wij*(E45*N3**2 + N3x*(D16*N3x + D66*N3y) + N3y*(D12*N3x + D26*N3y))
    K[3+c3, 0+c4] += detJ*wij*(N4x*(B11*N3x + B16*N3y) + N4y*(B16*N3x + B66*N3y))
    K[3+c3, 1+c4] += detJ*wij*(N4x*(B16*N3x + B66*N3y) + N4y*(B12*N3x + B26*N3y))
    K[3+c3, 2+c4] += detJ*wij*(E45*N3*N4y + E55*N3*N4x)
    K[3+c3, 3+c4] += detJ*wij*(E55*N3*N4 + Ephix*gamma3*gamma4 + N4x*(D11*N3x + D16*N3y) + N4y*(D16*N3x + D66*N3y))
    K[3+c3, 4+c4] += detJ*wij*(E45*N3*N4 + N4x*(D16*N3x + D66*N3y) + N4y*(D12*N3x + D26*N3y))
    K[4+c3, 0+c1] += detJ*wij*(N1x*(B12*N3y + B16*N3x) + N1y*(B26*N3y + B66*N3x))
    K[4+c3, 1+c1] += detJ*wij*(N1x*(B26*N3y + B66*N3x) + N1y*(B22*N3y + B26*N3x))
    K[4+c3, 2+c1] += detJ*wij*(E44*N1y*N3 + E45*N1x*N3)
    K[4+c3, 3+c1] += detJ*wij*(E45*N1*N3 + N1x*(D12*N3y + D16*N3x) + N1y*(D26*N3y + D66*N3x))
    K[4+c3, 4+c1] += detJ*wij*(E44*N1*N3 + Ephiy*gamma1*gamma3 + N1x*(D26*N3y + D66*N3x) + N1y*(D22*N3y + D26*N3x))
    K[4+c3, 0+c2] += detJ*wij*(N2x*(B12*N3y + B16*N3x) + N2y*(B26*N3y + B66*N3x))
    K[4+c3, 1+c2] += detJ*wij*(N2x*(B26*N3y + B66*N3x) + N2y*(B22*N3y + B26*N3x))
    K[4+c3, 2+c2] += detJ*wij*(E44*N2y*N3 + E45*N2x*N3)
    K[4+c3, 3+c2] += detJ*wij*(E45*N2*N3 + N2x*(D12*N3y + D16*N3x) + N2y*(D26*N3y + D66*N3x))
    K[4+c3, 4+c2] += detJ*wij*(E44*N2*N3 + Ephiy*gamma2*gamma3 + N2x*(D26*N3y + D66*N3x) + N2y*(D22*N3y + D26*N3x))
    K[4+c3, 0+c3] += detJ*wij*(N3x*(B12*N3y + B16*N3x) + N3y*(B26*N3y + B66*N3x))
    K[4+c3, 1+c3] += detJ*wij*(N3x*(B26*N3y + B66*N3x) + N3y*(B22*N3y + B26*N3x))
    K[4+c3, 2+c3] += detJ*wij*(E44*N3*N3y + E45*N3*N3x)
    K[4+c3, 3+c3] += detJ*wij*(E45*N3**2 + N3x*(D12*N3y + D16*N3x) + N3y*(D26*N3y + D66*N3x))
    K[4+c3, 4+c3] += detJ*wij*(E44*N3**2 + Ephiy*gamma3**2 + N3x*(D26*N3y + D66*N3x) + N3y*(D22*N3y + D26*N3x))
    K[4+c3, 0+c4] += detJ*wij*(N4x*(B12*N3y + B16*N3x) + N4y*(B26*N3y + B66*N3x))
    K[4+c3, 1+c4] += detJ*wij*(N4x*(B26*N3y + B66*N3x) + N4y*(B22*N3y + B26*N3x))
    K[4+c3, 2+c4] += detJ*wij*(E44*N3*N4y + E45*N3*N4x)
    K[4+c3, 3+c4] += detJ*wij*(E45*N3*N4 + N4x*(D12*N3y + D16*N3x) + N4y*(D26*N3y + D66*N3x))
    K[4+c3, 4+c4] += detJ*wij*(E44*N3*N4 + Ephiy*gamma3*gamma4 + N4x*(D26*N3y + D66*N3x) + N4y*(D22*N3y + D26*N3x))
    K[0+c4, 0+c1] += detJ*wij*(Eu*gamma1*gamma4 + N1x*(A11*N4x + A16*N4y) + N1y*(A16*N4x + A66*N4y))
    K[0+c4, 1+c1] += detJ*wij*(N1x*(A16*N4x + A66*N4y) + N1y*(A12*N4x + A26*N4y))
    K[0+c4, 3+c1] += detJ*wij*(N1x*(B11*N4x + B16*N4y) + N1y*(B16*N4x + B66*N4y))
    K[0+c4, 4+c1] += detJ*wij*(N1x*(B16*N4x + B66*N4y) + N1y*(B12*N4x + B26*N4y))
    K[0+c4, 0+c2] += detJ*wij*(Eu*gamma2*gamma4 + N2x*(A11*N4x + A16*N4y) + N2y*(A16*N4x + A66*N4y))
    K[0+c4, 1+c2] += detJ*wij*(N2x*(A16*N4x + A66*N4y) + N2y*(A12*N4x + A26*N4y))
    K[0+c4, 3+c2] += detJ*wij*(N2x*(B11*N4x + B16*N4y) + N2y*(B16*N4x + B66*N4y))
    K[0+c4, 4+c2] += detJ*wij*(N2x*(B16*N4x + B66*N4y) + N2y*(B12*N4x + B26*N4y))
    K[0+c4, 0+c3] += detJ*wij*(Eu*gamma3*gamma4 + N3x*(A11*N4x + A16*N4y) + N3y*(A16*N4x + A66*N4y))
    K[0+c4, 1+c3] += detJ*wij*(N3x*(A16*N4x + A66*N4y) + N3y*(A12*N4x + A26*N4y))
    K[0+c4, 3+c3] += detJ*wij*(N3x*(B11*N4x + B16*N4y) + N3y*(B16*N4x + B66*N4y))
    K[0+c4, 4+c3] += detJ*wij*(N3x*(B16*N4x + B66*N4y) + N3y*(B12*N4x + B26*N4y))
    K[0+c4, 0+c4] += detJ*wij*(Eu*gamma4**2 + N4x*(A11*N4x + A16*N4y) + N4y*(A16*N4x + A66*N4y))
    K[0+c4, 1+c4] += detJ*wij*(N4x*(A16*N4x + A66*N4y) + N4y*(A12*N4x + A26*N4y))
    K[0+c4, 3+c4] += detJ*wij*(N4x*(B11*N4x + B16*N4y) + N4y*(B16*N4x + B66*N4y))
    K[0+c4, 4+c4] += detJ*wij*(N4x*(B16*N4x + B66*N4y) + N4y*(B12*N4x + B26*N4y))
    K[1+c4, 0+c1] += detJ*wij*(N1x*(A12*N4y + A16*N4x) + N1y*(A26*N4y + A66*N4x))
    K[1+c4, 1+c1] += detJ*wij*(Ev*gamma1*gamma4 + N1x*(A26*N4y + A66*N4x) + N1y*(A22*N4y + A26*N4x))
    K[1+c4, 3+c1] += detJ*wij*(N1x*(B12*N4y + B16*N4x) + N1y*(B26*N4y + B66*N4x))
    K[1+c4, 4+c1] += detJ*wij*(N1x*(B26*N4y + B66*N4x) + N1y*(B22*N4y + B26*N4x))
    K[1+c4, 0+c2] += detJ*wij*(N2x*(A12*N4y + A16*N4x) + N2y*(A26*N4y + A66*N4x))
    K[1+c4, 1+c2] += detJ*wij*(Ev*gamma2*gamma4 + N2x*(A26*N4y + A66*N4x) + N2y*(A22*N4y + A26*N4x))
    K[1+c4, 3+c2] += detJ*wij*(N2x*(B12*N4y + B16*N4x) + N2y*(B26*N4y + B66*N4x))
    K[1+c4, 4+c2] += detJ*wij*(N2x*(B26*N4y + B66*N4x) + N2y*(B22*N4y + B26*N4x))
    K[1+c4, 0+c3] += detJ*wij*(N3x*(A12*N4y + A16*N4x) + N3y*(A26*N4y + A66*N4x))
    K[1+c4, 1+c3] += detJ*wij*(Ev*gamma3*gamma4 + N3x*(A26*N4y + A66*N4x) + N3y*(A22*N4y + A26*N4x))
    K[1+c4, 3+c3] += detJ*wij*(N3x*(B12*N4y + B16*N4x) + N3y*(B26*N4y + B66*N4x))
    K[1+c4, 4+c3] += detJ*wij*(N3x*(B26*N4y + B66*N4x) + N3y*(B22*N4y + B26*N4x))
    K[1+c4, 0+c4] += detJ*wij*(N4x*(A12*N4y + A16*N4x) + N4y*(A26*N4y + A66*N4x))
    K[1+c4, 1+c4] += detJ*wij*(Ev*gamma4**2 + N4x*(A26*N4y + A66*N4x) + N4y*(A22*N4y + A26*N4x))
    K[1+c4, 3+c4] += detJ*wij*(N4x*(B12*N4y + B16*N4x) + N4y*(B26*N4y + B66*N4x))
    K[1+c4, 4+c4] += detJ*wij*(N4x*(B26*N4y + B66*N4x) + N4y*(B22*N4y + B26*N4x))
    K[2+c4, 2+c1] += detJ*wij*(Ew*gamma1*gamma4 + N1x*(E45*N4y + E55*N4x) + N1y*(E44*N4y + E45*N4x))
    K[2+c4, 3+c1] += N1*detJ*wij*(E45*N4y + E55*N4x)
    K[2+c4, 4+c1] += N1*detJ*wij*(E44*N4y + E45*N4x)
    K[2+c4, 2+c2] += detJ*wij*(Ew*gamma2*gamma4 + N2x*(E45*N4y + E55*N4x) + N2y*(E44*N4y + E45*N4x))
    K[2+c4, 3+c2] += N2*detJ*wij*(E45*N4y + E55*N4x)
    K[2+c4, 4+c2] += N2*detJ*wij*(E44*N4y + E45*N4x)
    K[2+c4, 2+c3] += detJ*wij*(Ew*gamma3*gamma4 + N3x*(E45*N4y + E55*N4x) + N3y*(E44*N4y + E45*N4x))
    K[2+c4, 3+c3] += N3*detJ*wij*(E45*N4y + E55*N4x)
    K[2+c4, 4+c3] += N3*detJ*wij*(E44*N4y + E45*N4x)
    K[2+c4, 2+c4] += detJ*wij*(Ew*gamma4**2 + N4x*(E45*N4y + E55*N4x) + N4y*(E44*N4y + E45*N4x))
    K[2+c4, 3+c4] += N4*detJ*wij*(E45*N4y + E55*N4x)
    K[2+c4, 4+c4] += N4*detJ*wij*(E44*N4y + E45*N4x)
    K[3+c4, 0+c1] += detJ*wij*(N1x*(B11*N4x + B16*N4y) + N1y*(B16*N4x + B66*N4y))
    K[3+c4, 1+c1] += detJ*wij*(N1x*(B16*N4x + B66*N4y) + N1y*(B12*N4x + B26*N4y))
    K[3+c4, 2+c1] += detJ*wij*(E45*N1y*N4 + E55*N1x*N4)
    K[3+c4, 3+c1] += detJ*wij*(E55*N1*N4 + Ephix*gamma1*gamma4 + N1x*(D11*N4x + D16*N4y) + N1y*(D16*N4x + D66*N4y))
    K[3+c4, 4+c1] += detJ*wij*(E45*N1*N4 + N1x*(D16*N4x + D66*N4y) + N1y*(D12*N4x + D26*N4y))
    K[3+c4, 0+c2] += detJ*wij*(N2x*(B11*N4x + B16*N4y) + N2y*(B16*N4x + B66*N4y))
    K[3+c4, 1+c2] += detJ*wij*(N2x*(B16*N4x + B66*N4y) + N2y*(B12*N4x + B26*N4y))
    K[3+c4, 2+c2] += detJ*wij*(E45*N2y*N4 + E55*N2x*N4)
    K[3+c4, 3+c2] += detJ*wij*(E55*N2*N4 + Ephix*gamma2*gamma4 + N2x*(D11*N4x + D16*N4y) + N2y*(D16*N4x + D66*N4y))
    K[3+c4, 4+c2] += detJ*wij*(E45*N2*N4 + N2x*(D16*N4x + D66*N4y) + N2y*(D12*N4x + D26*N4y))
    K[3+c4, 0+c3] += detJ*wij*(N3x*(B11*N4x + B16*N4y) + N3y*(B16*N4x + B66*N4y))
    K[3+c4, 1+c3] += detJ*wij*(N3x*(B16*N4x + B66*N4y) + N3y*(B12*N4x + B26*N4y))
    K[3+c4, 2+c3] += detJ*wij*(E45*N3y*N4 + E55*N3x*N4)
    K[3+c4, 3+c3] += detJ*wij*(E55*N3*N4 + Ephix*gamma3*gamma4 + N3x*(D11*N4x + D16*N4y) + N3y*(D16*N4x + D66*N4y))
    K[3+c4, 4+c3] += detJ*wij*(E45*N3*N4 + N3x*(D16*N4x + D66*N4y) + N3y*(D12*N4x + D26*N4y))
    K[3+c4, 0+c4] += detJ*wij*(N4x*(B11*N4x + B16*N4y) + N4y*(B16*N4x + B66*N4y))
    K[3+c4, 1+c4] += detJ*wij*(N4x*(B16*N4x + B66*N4y) + N4y*(B12*N4x + B26*N4y))
    K[3+c4, 2+c4] += detJ*wij*(E45*N4*N4y + E55*N4*N4x)
    K[3+c4, 3+c4] += detJ*wij*(E55*N4**2 + Ephix*gamma4**2 + N4x*(D11*N4x + D16*N4y) + N4y*(D16*N4x + D66*N4y))
    K[3+c4, 4+c4] += detJ*wij*(E45*N4**2 + N4x*(D16*N4x + D66*N4y) + N4y*(D12*N4x + D26*N4y))
    K[4+c4, 0+c1] += detJ*wij*(N1x*(B12*N4y + B16*N4x) + N1y*(B26*N4y + B66*N4x))
    K[4+c4, 1+c1] += detJ*wij*(N1x*(B26*N4y + B66*N4x) + N1y*(B22*N4y + B26*N4x))
    K[4+c4, 2+c1] += detJ*wij*(E44*N1y*N4 + E45*N1x*N4)
    K[4+c4, 3+c1] += detJ*wij*(E45*N1*N4 + N1x*(D12*N4y + D16*N4x) + N1y*(D26*N4y + D66*N4x))
    K[4+c4, 4+c1] += detJ*wij*(E44*N1*N4 + Ephiy*gamma1*gamma4 + N1x*(D26*N4y + D66*N4x) + N1y*(D22*N4y + D26*N4x))
    K[4+c4, 0+c2] += detJ*wij*(N2x*(B12*N4y + B16*N4x) + N2y*(B26*N4y + B66*N4x))
    K[4+c4, 1+c2] += detJ*wij*(N2x*(B26*N4y + B66*N4x) + N2y*(B22*N4y + B26*N4x))
    K[4+c4, 2+c2] += detJ*wij*(E44*N2y*N4 + E45*N2x*N4)
    K[4+c4, 3+c2] += detJ*wij*(E45*N2*N4 + N2x*(D12*N4y + D16*N4x) + N2y*(D26*N4y + D66*N4x))
    K[4+c4, 4+c2] += detJ*wij*(E44*N2*N4 + Ephiy*gamma2*gamma4 + N2x*(D26*N4y + D66*N4x) + N2y*(D22*N4y + D26*N4x))
    K[4+c4, 0+c3] += detJ*wij*(N3x*(B12*N4y + B16*N4x) + N3y*(B26*N4y + B66*N4x))
    K[4+c4, 1+c3] += detJ*wij*(N3x*(B26*N4y + B66*N4x) + N3y*(B22*N4y + B26*N4x))
    K[4+c4, 2+c3] += detJ*wij*(E44*N3y*N4 + E45*N3x*N4)
    K[4+c4, 3+c3] += detJ*wij*(E45*N3*N4 + N3x*(D12*N4y + D16*N4x) + N3y*(D26*N4y + D66*N4x))
    K[4+c4, 4+c3] += detJ*wij*(E44*N3*N4 + Ephiy*gamma3*gamma4 + N3x*(D26*N4y + D66*N4x) + N3y*(D22*N4y + D26*N4x))
    K[4+c4, 0+c4] += detJ*wij*(N4x*(B12*N4y + B16*N4x) + N4y*(B26*N4y + B66*N4x))
    K[4+c4, 1+c4] += detJ*wij*(N4x*(B26*N4y + B66*N4x) + N4y*(B22*N4y + B26*N4x))
    K[4+c4, 2+c4] += detJ*wij*(E44*N4*N4y + E45*N4*N4x)
    K[4+c4, 3+c4] += detJ*wij*(E45*N4**2 + N4x*(D12*N4y + D16*N4x) + N4y*(D26*N4y + D66*N4x))
    K[4+c4, 4+c4] += detJ*wij*(E44*N4**2 + Ephiy*gamma4**2 + N4x*(D26*N4y + D66*N4x) + N4y*(D22*N4y + D26*N4x))


def update_M(quad, nid_pos, ncoords, M, lumped=False):
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
    lumped : bool, optional
        If lumped mass should be used

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

    if lumped:
        M[0+c1, 0+c1] += h*rho*(3*x1*y2 - 3*x1*y4 - 3*x2*y1 + x2*y3 + 2*x2*y4 - x3*y2 + x3*y4 + 3*x4*y1 - 2*x4*y2 - x4*y3)/16
        M[1+c1, 1+c1] += h*rho*(3*x1*y2 - 3*x1*y4 - 3*x2*y1 + x2*y3 + 2*x2*y4 - x3*y2 + x3*y4 + 3*x4*y1 - 2*x4*y2 - x4*y3)/16
        M[2+c1, 2+c1] += h*rho*(3*x1*y2 - 3*x1*y4 - 3*x2*y1 + x2*y3 + 2*x2*y4 - x3*y2 + x3*y4 + 3*x4*y1 - 2*x4*y2 - x4*y3)/16
        M[3+c1, 3+c1] += h*rho*(57*x1**3*y2 - 57*x1**3*y4 - 57*x1**2*x2*y1 - 52*x1**2*x2*y2 + 33*x1**2*x2*y3 + 76*x1**2*x2*y4 - 43*x1**2*x3*y2 + 43*x1**2*x3*y4 + 57*x1**2*x4*y1 - 76*x1**2*x4*y2 - 33*x1**2*x4*y3 + 52*x1**2*x4*y4 + 52*x1*x2**2*y1 + 15*x1*x2**2*y2 - 28*x1*x2**2*y3 - 39*x1*x2**2*y4 + 10*x1*x2*x3*y1 + 32*x1*x2*x3*y2 - 10*x1*x2*x3*y3 - 32*x1*x2*x3*y4 + 42*x1*x2*x4*y2 - 42*x1*x2*x4*y4 + 11*x1*x3**2*y2 - 11*x1*x3**2*y4 - 10*x1*x3*x4*y1 + 32*x1*x3*x4*y2 + 10*x1*x3*x4*y3 - 32*x1*x3*x4*y4 - 52*x1*x4**2*y1 + 39*x1*x4**2*y2 + 28*x1*x4**2*y3 - 15*x1*x4**2*y4 - 15*x2**3*y1 + 7*x2**3*y3 + 8*x2**3*y4 - 4*x2**2*x3*y1 - 7*x2**2*x3*y2 + 4*x2**2*x3*y3 + 7*x2**2*x3*y4 - 3*x2**2*x4*y1 - 8*x2**2*x4*y2 + 3*x2**2*x4*y3 + 8*x2**2*x4*y4 - x2*x3**2*y1 - 4*x2*x3**2*y2 + x2*x3**2*y3 + 4*x2*x3**2*y4 - 10*x2*x3*x4*y2 + 10*x2*x3*x4*y4 + 3*x2*x4**2*y1 - 8*x2*x4**2*y2 - 3*x2*x4**2*y3 + 8*x2*x4**2*y4 - x3**3*y2 + x3**3*y4 + x3**2*x4*y1 - 4*x3**2*x4*y2 - x3**2*x4*y3 + 4*x3**2*x4*y4 + 4*x3*x4**2*y1 - 7*x3*x4**2*y2 - 4*x3*x4**2*y3 + 7*x3*x4**2*y4 + 15*x4**3*y1 - 8*x4**3*y2 - 7*x4**3*y3)/1536
        M[4+c1, 4+c1] += h*rho*(57*x1*y1**2*y2 - 57*x1*y1**2*y4 - 52*x1*y1*y2**2 - 10*x1*y1*y2*y3 + 10*x1*y1*y3*y4 + 52*x1*y1*y4**2 + 15*x1*y2**3 + 4*x1*y2**2*y3 + 3*x1*y2**2*y4 + x1*y2*y3**2 - 3*x1*y2*y4**2 - x1*y3**2*y4 - 4*x1*y3*y4**2 - 15*x1*y4**3 - 57*x2*y1**3 + 52*x2*y1**2*y2 + 43*x2*y1**2*y3 + 76*x2*y1**2*y4 - 15*x2*y1*y2**2 - 32*x2*y1*y2*y3 - 42*x2*y1*y2*y4 - 11*x2*y1*y3**2 - 32*x2*y1*y3*y4 - 39*x2*y1*y4**2 + 7*x2*y2**2*y3 + 8*x2*y2**2*y4 + 4*x2*y2*y3**2 + 10*x2*y2*y3*y4 + 8*x2*y2*y4**2 + x2*y3**3 + 4*x2*y3**2*y4 + 7*x2*y3*y4**2 + 8*x2*y4**3 - 33*x3*y1**2*y2 + 33*x3*y1**2*y4 + 28*x3*y1*y2**2 + 10*x3*y1*y2*y3 - 10*x3*y1*y3*y4 - 28*x3*y1*y4**2 - 7*x3*y2**3 - 4*x3*y2**2*y3 - 3*x3*y2**2*y4 - x3*y2*y3**2 + 3*x3*y2*y4**2 + x3*y3**2*y4 + 4*x3*y3*y4**2 + 7*x3*y4**3 + 57*x4*y1**3 - 76*x4*y1**2*y2 - 43*x4*y1**2*y3 - 52*x4*y1**2*y4 + 39*x4*y1*y2**2 + 32*x4*y1*y2*y3 + 42*x4*y1*y2*y4 + 11*x4*y1*y3**2 + 32*x4*y1*y3*y4 + 15*x4*y1*y4**2 - 8*x4*y2**3 - 7*x4*y2**2*y3 - 8*x4*y2**2*y4 - 4*x4*y2*y3**2 - 10*x4*y2*y3*y4 - 8*x4*y2*y4**2 - x4*y3**3 - 4*x4*y3**2*y4 - 7*x4*y3*y4**2)/1536
        M[0+c2, 0+c2] += h*rho*(3*x1*y2 - 2*x1*y3 - x1*y4 - 3*x2*y1 + 3*x2*y3 + 2*x3*y1 - 3*x3*y2 + x3*y4 + x4*y1 - x4*y3)/16
        M[1+c2, 1+c2] += h*rho*(3*x1*y2 - 2*x1*y3 - x1*y4 - 3*x2*y1 + 3*x2*y3 + 2*x3*y1 - 3*x3*y2 + x3*y4 + x4*y1 - x4*y3)/16
        M[2+c2, 2+c2] += h*rho*(3*x1*y2 - 2*x1*y3 - x1*y4 - 3*x2*y1 + 3*x2*y3 + 2*x3*y1 - 3*x3*y2 + x3*y4 + x4*y1 - x4*y3)/16
        M[3+c2, 3+c2] += h*rho*(15*x1**3*y2 - 8*x1**3*y3 - 7*x1**3*y4 - 15*x1**2*x2*y1 - 52*x1**2*x2*y2 + 39*x1**2*x2*y3 + 28*x1**2*x2*y4 + 8*x1**2*x3*y1 + 3*x1**2*x3*y2 - 8*x1**2*x3*y3 - 3*x1**2*x3*y4 + 7*x1**2*x4*y1 + 4*x1**2*x4*y2 - 7*x1**2*x4*y3 - 4*x1**2*x4*y4 + 52*x1*x2**2*y1 + 57*x1*x2**2*y2 - 76*x1*x2**2*y3 - 33*x1*x2**2*y4 - 42*x1*x2*x3*y1 + 42*x1*x2*x3*y3 - 32*x1*x2*x4*y1 - 10*x1*x2*x4*y2 + 32*x1*x2*x4*y3 + 10*x1*x2*x4*y4 + 8*x1*x3**2*y1 - 3*x1*x3**2*y2 - 8*x1*x3**2*y3 + 3*x1*x3**2*y4 + 10*x1*x3*x4*y1 - 10*x1*x3*x4*y3 + 4*x1*x4**2*y1 + x1*x4**2*y2 - 4*x1*x4**2*y3 - x1*x4**2*y4 - 57*x2**3*y1 + 57*x2**3*y3 + 76*x2**2*x3*y1 - 57*x2**2*x3*y2 - 52*x2**2*x3*y3 + 33*x2**2*x3*y4 + 43*x2**2*x4*y1 - 43*x2**2*x4*y3 - 39*x2*x3**2*y1 + 52*x2*x3**2*y2 + 15*x2*x3**2*y3 - 28*x2*x3**2*y4 - 32*x2*x3*x4*y1 + 10*x2*x3*x4*y2 + 32*x2*x3*x4*y3 - 10*x2*x3*x4*y4 - 11*x2*x4**2*y1 + 11*x2*x4**2*y3 + 8*x3**3*y1 - 15*x3**3*y2 + 7*x3**3*y4 + 7*x3**2*x4*y1 - 4*x3**2*x4*y2 - 7*x3**2*x4*y3 + 4*x3**2*x4*y4 + 4*x3*x4**2*y1 - x3*x4**2*y2 - 4*x3*x4**2*y3 + x3*x4**2*y4 + x4**3*y1 - x4**3*y3)/1536
        M[4+c2, 4+c2] += h*rho*(15*x1*y1**2*y2 - 8*x1*y1**2*y3 - 7*x1*y1**2*y4 - 52*x1*y1*y2**2 + 42*x1*y1*y2*y3 + 32*x1*y1*y2*y4 - 8*x1*y1*y3**2 - 10*x1*y1*y3*y4 - 4*x1*y1*y4**2 + 57*x1*y2**3 - 76*x1*y2**2*y3 - 43*x1*y2**2*y4 + 39*x1*y2*y3**2 + 32*x1*y2*y3*y4 + 11*x1*y2*y4**2 - 8*x1*y3**3 - 7*x1*y3**2*y4 - 4*x1*y3*y4**2 - x1*y4**3 - 15*x2*y1**3 + 52*x2*y1**2*y2 - 3*x2*y1**2*y3 - 4*x2*y1**2*y4 - 57*x2*y1*y2**2 + 10*x2*y1*y2*y4 + 3*x2*y1*y3**2 - x2*y1*y4**2 + 57*x2*y2**2*y3 - 52*x2*y2*y3**2 - 10*x2*y2*y3*y4 + 15*x2*y3**3 + 4*x2*y3**2*y4 + x2*y3*y4**2 + 8*x3*y1**3 - 39*x3*y1**2*y2 + 8*x3*y1**2*y3 + 7*x3*y1**2*y4 + 76*x3*y1*y2**2 - 42*x3*y1*y2*y3 - 32*x3*y1*y2*y4 + 8*x3*y1*y3**2 + 10*x3*y1*y3*y4 + 4*x3*y1*y4**2 - 57*x3*y2**3 + 52*x3*y2**2*y3 + 43*x3*y2**2*y4 - 15*x3*y2*y3**2 - 32*x3*y2*y3*y4 - 11*x3*y2*y4**2 + 7*x3*y3**2*y4 + 4*x3*y3*y4**2 + x3*y4**3 + 7*x4*y1**3 - 28*x4*y1**2*y2 + 3*x4*y1**2*y3 + 4*x4*y1**2*y4 + 33*x4*y1*y2**2 - 10*x4*y1*y2*y4 - 3*x4*y1*y3**2 + x4*y1*y4**2 - 33*x4*y2**2*y3 + 28*x4*y2*y3**2 + 10*x4*y2*y3*y4 - 7*x4*y3**3 - 4*x4*y3**2*y4 - x4*y3*y4**2)/1536
        M[0+c3, 0+c3] += h*rho*(x1*y2 - x1*y4 - x2*y1 + 3*x2*y3 - 2*x2*y4 - 3*x3*y2 + 3*x3*y4 + x4*y1 + 2*x4*y2 - 3*x4*y3)/16
        M[1+c3, 1+c3] += h*rho*(x1*y2 - x1*y4 - x2*y1 + 3*x2*y3 - 2*x2*y4 - 3*x3*y2 + 3*x3*y4 + x4*y1 + 2*x4*y2 - 3*x4*y3)/16
        M[2+c3, 2+c3] += h*rho*(x1*y2 - x1*y4 - x2*y1 + 3*x2*y3 - 2*x2*y4 - 3*x3*y2 + 3*x3*y4 + x4*y1 + 2*x4*y2 - 3*x4*y3)/16
        M[3+c3, 3+c3] += h*rho*(x1**3*y2 - x1**3*y4 - x1**2*x2*y1 + 4*x1**2*x2*y2 + x1**2*x2*y3 - 4*x1**2*x2*y4 - 11*x1**2*x3*y2 + 11*x1**2*x3*y4 + x1**2*x4*y1 + 4*x1**2*x4*y2 - x1**2*x4*y3 - 4*x1**2*x4*y4 - 4*x1*x2**2*y1 + 7*x1*x2**2*y2 + 4*x1*x2**2*y3 - 7*x1*x2**2*y4 + 10*x1*x2*x3*y1 - 32*x1*x2*x3*y2 - 10*x1*x2*x3*y3 + 32*x1*x2*x3*y4 + 10*x1*x2*x4*y2 - 10*x1*x2*x4*y4 + 43*x1*x3**2*y2 - 43*x1*x3**2*y4 - 10*x1*x3*x4*y1 - 32*x1*x3*x4*y2 + 10*x1*x3*x4*y3 + 32*x1*x3*x4*y4 + 4*x1*x4**2*y1 + 7*x1*x4**2*y2 - 4*x1*x4**2*y3 - 7*x1*x4**2*y4 - 7*x2**3*y1 + 15*x2**3*y3 - 8*x2**3*y4 + 28*x2**2*x3*y1 - 15*x2**2*x3*y2 - 52*x2**2*x3*y3 + 39*x2**2*x3*y4 - 3*x2**2*x4*y1 + 8*x2**2*x4*y2 + 3*x2**2*x4*y3 - 8*x2**2*x4*y4 - 33*x2*x3**2*y1 + 52*x2*x3**2*y2 + 57*x2*x3**2*y3 - 76*x2*x3**2*y4 - 42*x2*x3*x4*y2 + 42*x2*x3*x4*y4 + 3*x2*x4**2*y1 + 8*x2*x4**2*y2 - 3*x2*x4**2*y3 - 8*x2*x4**2*y4 - 57*x3**3*y2 + 57*x3**3*y4 + 33*x3**2*x4*y1 + 76*x3**2*x4*y2 - 57*x3**2*x4*y3 - 52*x3**2*x4*y4 - 28*x3*x4**2*y1 - 39*x3*x4**2*y2 + 52*x3*x4**2*y3 + 15*x3*x4**2*y4 + 7*x4**3*y1 + 8*x4**3*y2 - 15*x4**3*y3)/1536
        M[4+c3, 4+c3] += h*rho*(x1*y1**2*y2 - x1*y1**2*y4 + 4*x1*y1*y2**2 - 10*x1*y1*y2*y3 + 10*x1*y1*y3*y4 - 4*x1*y1*y4**2 + 7*x1*y2**3 - 28*x1*y2**2*y3 + 3*x1*y2**2*y4 + 33*x1*y2*y3**2 - 3*x1*y2*y4**2 - 33*x1*y3**2*y4 + 28*x1*y3*y4**2 - 7*x1*y4**3 - x2*y1**3 - 4*x2*y1**2*y2 + 11*x2*y1**2*y3 - 4*x2*y1**2*y4 - 7*x2*y1*y2**2 + 32*x2*y1*y2*y3 - 10*x2*y1*y2*y4 - 43*x2*y1*y3**2 + 32*x2*y1*y3*y4 - 7*x2*y1*y4**2 + 15*x2*y2**2*y3 - 8*x2*y2**2*y4 - 52*x2*y2*y3**2 + 42*x2*y2*y3*y4 - 8*x2*y2*y4**2 + 57*x2*y3**3 - 76*x2*y3**2*y4 + 39*x2*y3*y4**2 - 8*x2*y4**3 - x3*y1**2*y2 + x3*y1**2*y4 - 4*x3*y1*y2**2 + 10*x3*y1*y2*y3 - 10*x3*y1*y3*y4 + 4*x3*y1*y4**2 - 15*x3*y2**3 + 52*x3*y2**2*y3 - 3*x3*y2**2*y4 - 57*x3*y2*y3**2 + 3*x3*y2*y4**2 + 57*x3*y3**2*y4 - 52*x3*y3*y4**2 + 15*x3*y4**3 + x4*y1**3 + 4*x4*y1**2*y2 - 11*x4*y1**2*y3 + 4*x4*y1**2*y4 + 7*x4*y1*y2**2 - 32*x4*y1*y2*y3 + 10*x4*y1*y2*y4 + 43*x4*y1*y3**2 - 32*x4*y1*y3*y4 + 7*x4*y1*y4**2 + 8*x4*y2**3 - 39*x4*y2**2*y3 + 8*x4*y2**2*y4 + 76*x4*y2*y3**2 - 42*x4*y2*y3*y4 + 8*x4*y2*y4**2 - 57*x4*y3**3 + 52*x4*y3**2*y4 - 15*x4*y3*y4**2)/1536
        M[0+c4, 0+c4] += h*rho*(x1*y2 + 2*x1*y3 - 3*x1*y4 - x2*y1 + x2*y3 - 2*x3*y1 - x3*y2 + 3*x3*y4 + 3*x4*y1 - 3*x4*y3)/16
        M[1+c4, 1+c4] += h*rho*(x1*y2 + 2*x1*y3 - 3*x1*y4 - x2*y1 + x2*y3 - 2*x3*y1 - x3*y2 + 3*x3*y4 + 3*x4*y1 - 3*x4*y3)/16
        M[2+c4, 2+c4] += h*rho*(x1*y2 + 2*x1*y3 - 3*x1*y4 - x2*y1 + x2*y3 - 2*x3*y1 - x3*y2 + 3*x3*y4 + 3*x4*y1 - 3*x4*y3)/16
        M[3+c4, 3+c4] += h*rho*(7*x1**3*y2 + 8*x1**3*y3 - 15*x1**3*y4 - 7*x1**2*x2*y1 + 4*x1**2*x2*y2 + 7*x1**2*x2*y3 - 4*x1**2*x2*y4 - 8*x1**2*x3*y1 + 3*x1**2*x3*y2 + 8*x1**2*x3*y3 - 3*x1**2*x3*y4 + 15*x1**2*x4*y1 - 28*x1**2*x4*y2 - 39*x1**2*x4*y3 + 52*x1**2*x4*y4 - 4*x1*x2**2*y1 + x1*x2**2*y2 + 4*x1*x2**2*y3 - x1*x2**2*y4 - 10*x1*x2*x3*y1 + 10*x1*x2*x3*y3 + 32*x1*x2*x4*y1 - 10*x1*x2*x4*y2 - 32*x1*x2*x4*y3 + 10*x1*x2*x4*y4 - 8*x1*x3**2*y1 - 3*x1*x3**2*y2 + 8*x1*x3**2*y3 + 3*x1*x3**2*y4 + 42*x1*x3*x4*y1 - 42*x1*x3*x4*y3 - 52*x1*x4**2*y1 + 33*x1*x4**2*y2 + 76*x1*x4**2*y3 - 57*x1*x4**2*y4 - x2**3*y1 + x2**3*y3 - 4*x2**2*x3*y1 - x2**2*x3*y2 + 4*x2**2*x3*y3 + x2**2*x3*y4 + 11*x2**2*x4*y1 - 11*x2**2*x4*y3 - 7*x2*x3**2*y1 - 4*x2*x3**2*y2 + 7*x2*x3**2*y3 + 4*x2*x3**2*y4 + 32*x2*x3*x4*y1 + 10*x2*x3*x4*y2 - 32*x2*x3*x4*y3 - 10*x2*x3*x4*y4 - 43*x2*x4**2*y1 + 43*x2*x4**2*y3 - 8*x3**3*y1 - 7*x3**3*y2 + 15*x3**3*y4 + 39*x3**2*x4*y1 + 28*x3**2*x4*y2 - 15*x3**2*x4*y3 - 52*x3**2*x4*y4 - 76*x3*x4**2*y1 - 33*x3*x4**2*y2 + 52*x3*x4**2*y3 + 57*x3*x4**2*y4 + 57*x4**3*y1 - 57*x4**3*y3)/1536
        M[4+c4, 4+c4] += h*rho*(7*x1*y1**2*y2 + 8*x1*y1**2*y3 - 15*x1*y1**2*y4 + 4*x1*y1*y2**2 + 10*x1*y1*y2*y3 - 32*x1*y1*y2*y4 + 8*x1*y1*y3**2 - 42*x1*y1*y3*y4 + 52*x1*y1*y4**2 + x1*y2**3 + 4*x1*y2**2*y3 - 11*x1*y2**2*y4 + 7*x1*y2*y3**2 - 32*x1*y2*y3*y4 + 43*x1*y2*y4**2 + 8*x1*y3**3 - 39*x1*y3**2*y4 + 76*x1*y3*y4**2 - 57*x1*y4**3 - 7*x2*y1**3 - 4*x2*y1**2*y2 - 3*x2*y1**2*y3 + 28*x2*y1**2*y4 - x2*y1*y2**2 + 10*x2*y1*y2*y4 + 3*x2*y1*y3**2 - 33*x2*y1*y4**2 + x2*y2**2*y3 + 4*x2*y2*y3**2 - 10*x2*y2*y3*y4 + 7*x2*y3**3 - 28*x2*y3**2*y4 + 33*x2*y3*y4**2 - 8*x3*y1**3 - 7*x3*y1**2*y2 - 8*x3*y1**2*y3 + 39*x3*y1**2*y4 - 4*x3*y1*y2**2 - 10*x3*y1*y2*y3 + 32*x3*y1*y2*y4 - 8*x3*y1*y3**2 + 42*x3*y1*y3*y4 - 76*x3*y1*y4**2 - x3*y2**3 - 4*x3*y2**2*y3 + 11*x3*y2**2*y4 - 7*x3*y2*y3**2 + 32*x3*y2*y3*y4 - 43*x3*y2*y4**2 + 15*x3*y3**2*y4 - 52*x3*y3*y4**2 + 57*x3*y4**3 + 15*x4*y1**3 + 4*x4*y1**2*y2 + 3*x4*y1**2*y3 - 52*x4*y1**2*y4 + x4*y1*y2**2 - 10*x4*y1*y2*y4 - 3*x4*y1*y3**2 + 57*x4*y1*y4**2 - x4*y2**2*y3 - 4*x4*y2*y3**2 + 10*x4*y2*y3*y4 - 15*x4*y3**3 + 52*x4*y3**2*y4 - 57*x4*y3*y4**2)/1536

        return


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


def update_KA(quad, nid_pos, ncoords, KA):
    """Update global KA with KAe from a quad element

    Properties
    ----------
    quad : `.Quad`object
        The quad element being added to KA
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    KA : np.array
        Aerodynamic matrix

    """
    pos1 = nid_pos[quad.n1]
    pos2 = nid_pos[quad.n2]
    pos3 = nid_pos[quad.n3]
    pos4 = nid_pos[quad.n4]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    x3, y3 = ncoords[pos3]
    x4, y4 = ncoords[pos4]

    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    x3, y3 = ncoords[pos3]
    x4, y4 = ncoords[pos4]

    A = (np.cross([x2 - x1, y2 - y1], [x4 - x1, y4 - y1])/2 +
         np.cross([x4 - x3, y4 - y3], [x2 - x3, y2 - y3])/2)

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3
    c4 = DOF*pos4

    h = quad.h
    rho = quad.rho

    weights_points =[[8/9, 0.],
                     [5/9, -(3/5)**0.5 ],
                     [5/9, +(3/5)**0.5 ],
                     ]

    for wi, xi in weights_points:
        for wj, eta in weights_points:
            wij = wi*wj
            detJ = (-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16 - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16
            j11 = 2*(-xi*y1 + xi*y2 - xi*y3 + xi*y4 + y1 + y2 - y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
            j12 = 4*(-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
            N1 = eta*xi/4 - eta/4 - xi/4 + 1/4
            N2 = -eta*xi/4 - eta/4 + xi/4 + 1/4
            N3 = eta*xi/4 + eta/4 + xi/4 + 1/4
            N4 = -eta*xi/4 + eta/4 - xi/4 + 1/4
            N1x = j11*(eta - 1)/4 + j12*(xi - 1)/4
            N2x = -eta*j11/4 + j11/4 - j12*xi/4 - j12/4
            N3x = j11*(eta + 1)/4 + j12*(xi + 1)/4
            N4x = -eta*j11/4 - j11/4 - j12*xi/4 + j12/4

            KA[2+c1, 2+c1] += N1*N1x*detJ*wij
            KA[2+c1, 2+c2] += N1*N2x*detJ*wij
            KA[2+c1, 2+c3] += N1*N3x*detJ*wij
            KA[2+c1, 2+c4] += N1*N4x*detJ*wij
            KA[2+c2, 2+c1] += N1x*N2*detJ*wij
            KA[2+c2, 2+c2] += N2*N2x*detJ*wij
            KA[2+c2, 2+c3] += N2*N3x*detJ*wij
            KA[2+c2, 2+c4] += N2*N4x*detJ*wij
            KA[2+c3, 2+c1] += N1x*N3*detJ*wij
            KA[2+c3, 2+c2] += N2x*N3*detJ*wij
            KA[2+c3, 2+c3] += N3*N3x*detJ*wij
            KA[2+c3, 2+c4] += N3*N4x*detJ*wij
            KA[2+c4, 2+c1] += N1x*N4*detJ*wij
            KA[2+c4, 2+c2] += N2x*N4*detJ*wij
            KA[2+c4, 2+c3] += N3x*N4*detJ*wij
            KA[2+c4, 2+c4] += N4*N4x*detJ*wij
