import numpy as np

DOF = 5

class Quad4R(object):
    """Reissner-Mindlin plate element with reduced integration

    Formulated based on the first-order shear deformation theory for plates

    Reduced integration is achieved by having only 1 integration point at the
    center, while integrating the stiffness matrix. This removes shear locking.

    An artificial hour-glass stiffness is used, according to Brockman 1987:

    - the hourglass control in the in-plane direction was increased by 10 times
      with respect to what would be the equivalent from Brockman 1987
    - the hourglass control in the w direction by 10 times with respect to what
      would be the equivalent from Brockman 1987

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
    quad : `.Quad4R`object
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

    #NOTE hourglass stiffnesses
    #NOTE increasing in-plane by 10 times with respect to what would be the equivalent from Brockman 1987
    #NOTE increasing in w direction by 10 times with respect to what would be the equivalent from Brockman 1987
    Eu = 10*0.1*A11/(1 + 1/A)
    Ev = 10*0.1*A22/(1 + 1/A)
    Ephix = 12*0.1*D11/(1 + 1/A)
    Ephiy = 12*0.1*D22/(1 + 1/A)
    Ew = 10*(Ephix + Ephiy)/2

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


def update_KG(quad, u0, nid_pos, ncoords, KG):
    """Update geometric stiffness matrix KG with quad element

    Properties
    ----------
    quad : `.Quad4R`object
        The quad element being added to KG
    u0: array-like
        A displacement state ``u0`` in global coordinates.
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    KG : np.array
        Global geometric stiffness matrix

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

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3
    c4 = DOF*pos4

    u = np.concatenate((u0[c1:c1+DOF], u0[c2:c2+DOF], u0[c3:c3+DOF], u0[c4:c4+DOF]))

    #NOTE full 2-point Gauss-Legendre quadrature integration for KG
    weights_points =[[1., -0.577350269189625764509148780501957455647601751270126876],
                     [1., +0.577350269189625764509148780501957455647601751270126876]]

    #NOTE reduced integration with 1 point at center
    #NOTE this seems to work the same as using the full integration
    weights_points =[[2., 0]]

    for wi, xi in weights_points:
        for wj, eta in weights_points:
            wij = wi*wj
            detJ = (-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16 - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16
            j11 = 2*(-xi*y1 + xi*y2 - xi*y3 + xi*y4 + y1 + y2 - y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
            j12 = 4*(-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
            j21 = 4*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
            j22 = 4*(2*x1 - 2*x2 - (eta + 1)*(x1 - x2 + x3 - x4))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
            N1x = j11*(eta - 1)/4 + j12*(xi - 1)/4
            N2x = -eta*j11/4 + j11/4 - j12*xi/4 - j12/4
            N3x = j11*(eta + 1)/4 + j12*(xi + 1)/4
            N4x = -eta*j11/4 - j11/4 - j12*xi/4 + j12/4
            N1y = j21*(eta - 1)/4 + j22*(xi - 1)/4
            N2y = -eta*j21/4 + j21/4 - j22*xi/4 - j22/4
            N3y = j21*(eta + 1)/4 + j22*(xi + 1)/4
            N4y = -eta*j21/4 - j21/4 - j22*xi/4 + j22/4
            ux = N1x*u[0] + N2x*u[5] + N3x*u[10] + N4x*u[15]
            uy = N1y*u[0] + N2y*u[5] + N3y*u[10] + N4y*u[15]
            vx = N1x*u[1] + N2x*u[6] + N3x*u[11] + N4x*u[16]
            vy = N1y*u[1] + N2y*u[6] + N3y*u[11] + N4y*u[16]
            wx = N1x*u[2] + N2x*u[7] + N3x*u[12] + N4x*u[17]
            wy = N1y*u[2] + N2y*u[7] + N3y*u[12] + N4y*u[17]
            Nxx = u[0]*(A11*(N1x*ux/2 + N1x) + A12*N1y*uy/2 + A16*(N1x*uy/2 + N1y*ux/2 + N1y)) + u[10]*(A11*(N3x*ux/2 + N3x) + A12*N3y*uy/2 + A16*(N3x*uy/2 + N3y*ux/2 + N3y)) + u[11]*(A11*N3x*vx/2 + A12*(N3y*vy/2 + N3y) + A16*(N3x*vy/2 + N3x + N3y*vx/2)) + u[12]*(A11*N3x*wx/2 + A12*N3y*wy/2 + A16*(N3x*wy/2 + N3y*wx/2)) + u[13]*(B11*N3x + B16*N3y) + u[14]*(B12*N3y + B16*N3x) + u[15]*(A11*(N4x*ux/2 + N4x) + A12*N4y*uy/2 + A16*(N4x*uy/2 + N4y*ux/2 + N4y)) + u[16]*(A11*N4x*vx/2 + A12*(N4y*vy/2 + N4y) + A16*(N4x*vy/2 + N4x + N4y*vx/2)) + u[17]*(A11*N4x*wx/2 + A12*N4y*wy/2 + A16*(N4x*wy/2 + N4y*wx/2)) + u[18]*(B11*N4x + B16*N4y) + u[19]*(B12*N4y + B16*N4x) + u[1]*(A11*N1x*vx/2 + A12*(N1y*vy/2 + N1y) + A16*(N1x*vy/2 + N1x + N1y*vx/2)) + u[2]*(A11*N1x*wx/2 + A12*N1y*wy/2 + A16*(N1x*wy/2 + N1y*wx/2)) + u[3]*(B11*N1x + B16*N1y) + u[4]*(B12*N1y + B16*N1x) + u[5]*(A11*(N2x*ux/2 + N2x) + A12*N2y*uy/2 + A16*(N2x*uy/2 + N2y*ux/2 + N2y)) + u[6]*(A11*N2x*vx/2 + A12*(N2y*vy/2 + N2y) + A16*(N2x*vy/2 + N2x + N2y*vx/2)) + u[7]*(A11*N2x*wx/2 + A12*N2y*wy/2 + A16*(N2x*wy/2 + N2y*wx/2)) + u[8]*(B11*N2x + B16*N2y) + u[9]*(B12*N2y + B16*N2x)
            Nyy = u[0]*(A12*(N1x*ux/2 + N1x) + A22*N1y*uy/2 + A26*(N1x*uy/2 + N1y*ux/2 + N1y)) + u[10]*(A12*(N3x*ux/2 + N3x) + A22*N3y*uy/2 + A26*(N3x*uy/2 + N3y*ux/2 + N3y)) + u[11]*(A12*N3x*vx/2 + A22*(N3y*vy/2 + N3y) + A26*(N3x*vy/2 + N3x + N3y*vx/2)) + u[12]*(A12*N3x*wx/2 + A22*N3y*wy/2 + A26*(N3x*wy/2 + N3y*wx/2)) + u[13]*(B12*N3x + B26*N3y) + u[14]*(B22*N3y + B26*N3x) + u[15]*(A12*(N4x*ux/2 + N4x) + A22*N4y*uy/2 + A26*(N4x*uy/2 + N4y*ux/2 + N4y)) + u[16]*(A12*N4x*vx/2 + A22*(N4y*vy/2 + N4y) + A26*(N4x*vy/2 + N4x + N4y*vx/2)) + u[17]*(A12*N4x*wx/2 + A22*N4y*wy/2 + A26*(N4x*wy/2 + N4y*wx/2)) + u[18]*(B12*N4x + B26*N4y) + u[19]*(B22*N4y + B26*N4x) + u[1]*(A12*N1x*vx/2 + A22*(N1y*vy/2 + N1y) + A26*(N1x*vy/2 + N1x + N1y*vx/2)) + u[2]*(A12*N1x*wx/2 + A22*N1y*wy/2 + A26*(N1x*wy/2 + N1y*wx/2)) + u[3]*(B12*N1x + B26*N1y) + u[4]*(B22*N1y + B26*N1x) + u[5]*(A12*(N2x*ux/2 + N2x) + A22*N2y*uy/2 + A26*(N2x*uy/2 + N2y*ux/2 + N2y)) + u[6]*(A12*N2x*vx/2 + A22*(N2y*vy/2 + N2y) + A26*(N2x*vy/2 + N2x + N2y*vx/2)) + u[7]*(A12*N2x*wx/2 + A22*N2y*wy/2 + A26*(N2x*wy/2 + N2y*wx/2)) + u[8]*(B12*N2x + B26*N2y) + u[9]*(B22*N2y + B26*N2x)
            Nxy = u[0]*(A16*(N1x*ux/2 + N1x) + A26*N1y*uy/2 + A66*(N1x*uy/2 + N1y*ux/2 + N1y)) + u[10]*(A16*(N3x*ux/2 + N3x) + A26*N3y*uy/2 + A66*(N3x*uy/2 + N3y*ux/2 + N3y)) + u[11]*(A16*N3x*vx/2 + A26*(N3y*vy/2 + N3y) + A66*(N3x*vy/2 + N3x + N3y*vx/2)) + u[12]*(A16*N3x*wx/2 + A26*N3y*wy/2 + A66*(N3x*wy/2 + N3y*wx/2)) + u[13]*(B16*N3x + B66*N3y) + u[14]*(B26*N3y + B66*N3x) + u[15]*(A16*(N4x*ux/2 + N4x) + A26*N4y*uy/2 + A66*(N4x*uy/2 + N4y*ux/2 + N4y)) + u[16]*(A16*N4x*vx/2 + A26*(N4y*vy/2 + N4y) + A66*(N4x*vy/2 + N4x + N4y*vx/2)) + u[17]*(A16*N4x*wx/2 + A26*N4y*wy/2 + A66*(N4x*wy/2 + N4y*wx/2)) + u[18]*(B16*N4x + B66*N4y) + u[19]*(B26*N4y + B66*N4x) + u[1]*(A16*N1x*vx/2 + A26*(N1y*vy/2 + N1y) + A66*(N1x*vy/2 + N1x + N1y*vx/2)) + u[2]*(A16*N1x*wx/2 + A26*N1y*wy/2 + A66*(N1x*wy/2 + N1y*wx/2)) + u[3]*(B16*N1x + B66*N1y) + u[4]*(B26*N1y + B66*N1x) + u[5]*(A16*(N2x*ux/2 + N2x) + A26*N2y*uy/2 + A66*(N2x*uy/2 + N2y*ux/2 + N2y)) + u[6]*(A16*N2x*vx/2 + A26*(N2y*vy/2 + N2y) + A66*(N2x*vy/2 + N2x + N2y*vx/2)) + u[7]*(A16*N2x*wx/2 + A26*N2y*wy/2 + A66*(N2x*wy/2 + N2y*wx/2)) + u[8]*(B16*N2x + B66*N2y) + u[9]*(B26*N2y + B66*N2x)

            KG[0+c1, 0+c1] += detJ*wij*(N1x**2*Nxx + 2*N1x*N1y*Nxy + N1y**2*Nyy)
            KG[0+c1, 0+c2] += detJ*wij*(N1x*N2x*Nxx + N1y*N2y*Nyy + Nxy*(N1x*N2y + N1y*N2x))
            KG[0+c1, 0+c3] += detJ*wij*(N1x*N3x*Nxx + N1y*N3y*Nyy + Nxy*(N1x*N3y + N1y*N3x))
            KG[0+c1, 0+c4] += detJ*wij*(N1x*N4x*Nxx + N1y*N4y*Nyy + Nxy*(N1x*N4y + N1y*N4x))
            KG[1+c1, 1+c1] += detJ*wij*(N1x**2*Nxx + 2*N1x*N1y*Nxy + N1y**2*Nyy)
            KG[1+c1, 1+c2] += detJ*wij*(N1x*N2x*Nxx + N1y*N2y*Nyy + Nxy*(N1x*N2y + N1y*N2x))
            KG[1+c1, 1+c3] += detJ*wij*(N1x*N3x*Nxx + N1y*N3y*Nyy + Nxy*(N1x*N3y + N1y*N3x))
            KG[1+c1, 1+c4] += detJ*wij*(N1x*N4x*Nxx + N1y*N4y*Nyy + Nxy*(N1x*N4y + N1y*N4x))
            KG[2+c1, 2+c1] += detJ*wij*(N1x**2*Nxx + 2*N1x*N1y*Nxy + N1y**2*Nyy)
            KG[2+c1, 2+c2] += detJ*wij*(N1x*N2x*Nxx + N1y*N2y*Nyy + Nxy*(N1x*N2y + N1y*N2x))
            KG[2+c1, 2+c3] += detJ*wij*(N1x*N3x*Nxx + N1y*N3y*Nyy + Nxy*(N1x*N3y + N1y*N3x))
            KG[2+c1, 2+c4] += detJ*wij*(N1x*N4x*Nxx + N1y*N4y*Nyy + Nxy*(N1x*N4y + N1y*N4x))
            KG[0+c2, 0+c1] += detJ*wij*(N1x*N2x*Nxx + N1y*N2y*Nyy + Nxy*(N1x*N2y + N1y*N2x))
            KG[0+c2, 0+c2] += detJ*wij*(N2x**2*Nxx + 2*N2x*N2y*Nxy + N2y**2*Nyy)
            KG[0+c2, 0+c3] += detJ*wij*(N2x*N3x*Nxx + N2y*N3y*Nyy + Nxy*(N2x*N3y + N2y*N3x))
            KG[0+c2, 0+c4] += detJ*wij*(N2x*N4x*Nxx + N2y*N4y*Nyy + Nxy*(N2x*N4y + N2y*N4x))
            KG[1+c2, 1+c1] += detJ*wij*(N1x*N2x*Nxx + N1y*N2y*Nyy + Nxy*(N1x*N2y + N1y*N2x))
            KG[1+c2, 1+c2] += detJ*wij*(N2x**2*Nxx + 2*N2x*N2y*Nxy + N2y**2*Nyy)
            KG[1+c2, 1+c3] += detJ*wij*(N2x*N3x*Nxx + N2y*N3y*Nyy + Nxy*(N2x*N3y + N2y*N3x))
            KG[1+c2, 1+c4] += detJ*wij*(N2x*N4x*Nxx + N2y*N4y*Nyy + Nxy*(N2x*N4y + N2y*N4x))
            KG[2+c2, 2+c1] += detJ*wij*(N1x*N2x*Nxx + N1y*N2y*Nyy + Nxy*(N1x*N2y + N1y*N2x))
            KG[2+c2, 2+c2] += detJ*wij*(N2x**2*Nxx + 2*N2x*N2y*Nxy + N2y**2*Nyy)
            KG[2+c2, 2+c3] += detJ*wij*(N2x*N3x*Nxx + N2y*N3y*Nyy + Nxy*(N2x*N3y + N2y*N3x))
            KG[2+c2, 2+c4] += detJ*wij*(N2x*N4x*Nxx + N2y*N4y*Nyy + Nxy*(N2x*N4y + N2y*N4x))
            KG[0+c3, 0+c1] += detJ*wij*(N1x*N3x*Nxx + N1y*N3y*Nyy + Nxy*(N1x*N3y + N1y*N3x))
            KG[0+c3, 0+c2] += detJ*wij*(N2x*N3x*Nxx + N2y*N3y*Nyy + Nxy*(N2x*N3y + N2y*N3x))
            KG[0+c3, 0+c3] += detJ*wij*(N3x**2*Nxx + 2*N3x*N3y*Nxy + N3y**2*Nyy)
            KG[0+c3, 0+c4] += detJ*wij*(N3x*N4x*Nxx + N3y*N4y*Nyy + Nxy*(N3x*N4y + N3y*N4x))
            KG[1+c3, 1+c1] += detJ*wij*(N1x*N3x*Nxx + N1y*N3y*Nyy + Nxy*(N1x*N3y + N1y*N3x))
            KG[1+c3, 1+c2] += detJ*wij*(N2x*N3x*Nxx + N2y*N3y*Nyy + Nxy*(N2x*N3y + N2y*N3x))
            KG[1+c3, 1+c3] += detJ*wij*(N3x**2*Nxx + 2*N3x*N3y*Nxy + N3y**2*Nyy)
            KG[1+c3, 1+c4] += detJ*wij*(N3x*N4x*Nxx + N3y*N4y*Nyy + Nxy*(N3x*N4y + N3y*N4x))
            KG[2+c3, 2+c1] += detJ*wij*(N1x*N3x*Nxx + N1y*N3y*Nyy + Nxy*(N1x*N3y + N1y*N3x))
            KG[2+c3, 2+c2] += detJ*wij*(N2x*N3x*Nxx + N2y*N3y*Nyy + Nxy*(N2x*N3y + N2y*N3x))
            KG[2+c3, 2+c3] += detJ*wij*(N3x**2*Nxx + 2*N3x*N3y*Nxy + N3y**2*Nyy)
            KG[2+c3, 2+c4] += detJ*wij*(N3x*N4x*Nxx + N3y*N4y*Nyy + Nxy*(N3x*N4y + N3y*N4x))
            KG[0+c4, 0+c1] += detJ*wij*(N1x*N4x*Nxx + N1y*N4y*Nyy + Nxy*(N1x*N4y + N1y*N4x))
            KG[0+c4, 0+c2] += detJ*wij*(N2x*N4x*Nxx + N2y*N4y*Nyy + Nxy*(N2x*N4y + N2y*N4x))
            KG[0+c4, 0+c3] += detJ*wij*(N3x*N4x*Nxx + N3y*N4y*Nyy + Nxy*(N3x*N4y + N3y*N4x))
            KG[0+c4, 0+c4] += detJ*wij*(N4x**2*Nxx + 2*N4x*N4y*Nxy + N4y**2*Nyy)
            KG[1+c4, 1+c1] += detJ*wij*(N1x*N4x*Nxx + N1y*N4y*Nyy + Nxy*(N1x*N4y + N1y*N4x))
            KG[1+c4, 1+c2] += detJ*wij*(N2x*N4x*Nxx + N2y*N4y*Nyy + Nxy*(N2x*N4y + N2y*N4x))
            KG[1+c4, 1+c3] += detJ*wij*(N3x*N4x*Nxx + N3y*N4y*Nyy + Nxy*(N3x*N4y + N3y*N4x))
            KG[1+c4, 1+c4] += detJ*wij*(N4x**2*Nxx + 2*N4x*N4y*Nxy + N4y**2*Nyy)
            KG[2+c4, 2+c1] += detJ*wij*(N1x*N4x*Nxx + N1y*N4y*Nyy + Nxy*(N1x*N4y + N1y*N4x))
            KG[2+c4, 2+c2] += detJ*wij*(N2x*N4x*Nxx + N2y*N4y*Nyy + Nxy*(N2x*N4y + N2y*N4x))
            KG[2+c4, 2+c3] += detJ*wij*(N3x*N4x*Nxx + N3y*N4y*Nyy + Nxy*(N3x*N4y + N3y*N4x))
            KG[2+c4, 2+c4] += detJ*wij*(N4x**2*Nxx + 2*N4x*N4y*Nxy + N4y**2*Nyy)


def update_KNL(quad, u0, nid_pos, ncoords, KNL):
    """Update the nonlinear part of global constitutive stiffness KNL with quad element

    Properties
    ----------
    quad : `.Quad4R`object
        The quad element being added to KNL
    u0: array-like
        A displacement state ``u0`` in global coordinates.
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    KNL : np.array
        Nonlinear part of global constitutive stiffness matrix

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

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3
    c4 = DOF*pos4

    u = np.concatenate((u0[c1:c1+DOF], u0[c2:c2+DOF], u0[c3:c3+DOF], u0[c4:c4+DOF]))

    #NOTE full 2-point Gauss-Legendre quadrature integration for KNL
    weights_points =[[1., -0.577350269189625764509148780501957455647601751270126876],
                     [1., +0.577350269189625764509148780501957455647601751270126876]]

    #NOTE reduced integration with 1 point at center
    #NOTE this seems to work the same as using the full integration
    weights_points =[[2., 0]]

    for wi, xi in weights_points:
        for wj, eta in weights_points:
            wij = wi*wj
            detJ = (-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4))/16 - (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/16
            j11 = 2*(-xi*y1 + xi*y2 - xi*y3 + xi*y4 + y1 + y2 - y3 - y4)/(eta*x1*y2 - eta*x1*y3 - eta*x2*y1 + eta*x2*y4 + eta*x3*y1 - eta*x3*y4 - eta*x4*y2 + eta*x4*y3 + x1*xi*y3 - x1*xi*y4 - x1*y2 + x1*y4 - x2*xi*y3 + x2*xi*y4 + x2*y1 - x2*y3 - x3*xi*y1 + x3*xi*y2 + x3*y2 - x3*y4 + x4*xi*y1 - x4*xi*y2 - x4*y1 + x4*y3)
            j12 = 4*(-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
            j21 = 4*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
            j22 = 4*(2*x1 - 2*x2 - (eta + 1)*(x1 - x2 + x3 - x4))/(-(-2*x1 + 2*x2 + (eta + 1)*(x1 - x2 + x3 - x4))*(-2*y1 + 2*y4 + (xi + 1)*(y1 - y2) + (xi + 1)*(y3 - y4)) + (-2*y1 + 2*y2 + (eta + 1)*(y1 - y2 + y3 - y4))*(-2*x1 + 2*x4 + (x1 - x2)*(xi + 1) + (x3 - x4)*(xi + 1)))
            N1x = j11*(eta - 1)/4 + j12*(xi - 1)/4
            N2x = -eta*j11/4 + j11/4 - j12*xi/4 - j12/4
            N3x = j11*(eta + 1)/4 + j12*(xi + 1)/4
            N4x = -eta*j11/4 - j11/4 - j12*xi/4 + j12/4
            N1y = j21*(eta - 1)/4 + j22*(xi - 1)/4
            N2y = -eta*j21/4 + j21/4 - j22*xi/4 - j22/4
            N3y = j21*(eta + 1)/4 + j22*(xi + 1)/4
            N4y = -eta*j21/4 - j21/4 - j22*xi/4 + j22/4
            ux = N1x*u[0] + N2x*u[5] + N3x*u[10] + N4x*u[15]
            uy = N1y*u[0] + N2y*u[5] + N3y*u[10] + N4y*u[15]
            vx = N1x*u[1] + N2x*u[6] + N3x*u[11] + N4x*u[16]
            vy = N1y*u[1] + N2y*u[6] + N3y*u[11] + N4y*u[16]
            wx = N1x*u[2] + N2x*u[7] + N3x*u[12] + N4x*u[17]
            wy = N1y*u[2] + N2y*u[7] + N3y*u[12] + N4y*u[17]

            KNL[0+c1, 0+c1] += detJ*wij*(N1x*ux*(A11*N1x + A16*N1y) + N1x*ux*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N1x*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N1y*uy*(A12*N1x + A26*N1y) + N1y*uy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + N1y*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N1x*uy + N1y*ux) + (N1x*uy + N1y*ux)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 1+c1] += detJ*wij*(N1x*vx*(A11*N1x + A16*N1y) + N1x*vx*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N1x*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)) + N1y*vy*(A12*N1x + A26*N1y) + N1y*vy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + N1y*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N1x*vy + N1y*vx) + (N1x*vy + N1y*vx)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 2+c1] += detJ*wij*(N1x*wx*(A11*N1x + A16*N1y) + N1x*wx*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N1y*wy*(A12*N1x + A26*N1y) + N1y*wy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N1x*wy + N1y*wx) + (N1x*wy + N1y*wx)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 3+c1] += detJ*wij*(N1x*(B11*N1x*ux + B12*N1y*uy + B16*(N1x*uy + N1y*ux)) + N1y*(B16*N1x*ux + B26*N1y*uy + B66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 4+c1] += detJ*wij*(N1x*(B16*N1x*ux + B26*N1y*uy + B66*(N1x*uy + N1y*ux)) + N1y*(B12*N1x*ux + B22*N1y*uy + B26*(N1x*uy + N1y*ux)))
            KNL[0+c1, 0+c2] += detJ*wij*(N2x*ux*(A11*N1x + A16*N1y) + N2x*ux*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N2x*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N2y*uy*(A12*N1x + A26*N1y) + N2y*uy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + N2y*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N2x*uy + N2y*ux) + (N2x*uy + N2y*ux)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 1+c2] += detJ*wij*(N2x*vx*(A11*N1x + A16*N1y) + N2x*vx*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N2x*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)) + N2y*vy*(A12*N1x + A26*N1y) + N2y*vy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + N2y*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N2x*vy + N2y*vx) + (N2x*vy + N2y*vx)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 2+c2] += detJ*wij*(N2x*wx*(A11*N1x + A16*N1y) + N2x*wx*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N2y*wy*(A12*N1x + A26*N1y) + N2y*wy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N2x*wy + N2y*wx) + (N2x*wy + N2y*wx)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 3+c2] += detJ*wij*(N2x*(B11*N1x*ux + B12*N1y*uy + B16*(N1x*uy + N1y*ux)) + N2y*(B16*N1x*ux + B26*N1y*uy + B66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 4+c2] += detJ*wij*(N2x*(B16*N1x*ux + B26*N1y*uy + B66*(N1x*uy + N1y*ux)) + N2y*(B12*N1x*ux + B22*N1y*uy + B26*(N1x*uy + N1y*ux)))
            KNL[0+c1, 0+c3] += detJ*wij*(N3x*ux*(A11*N1x + A16*N1y) + N3x*ux*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N3x*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N3y*uy*(A12*N1x + A26*N1y) + N3y*uy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + N3y*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N3x*uy + N3y*ux) + (N3x*uy + N3y*ux)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 1+c3] += detJ*wij*(N3x*vx*(A11*N1x + A16*N1y) + N3x*vx*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N3x*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)) + N3y*vy*(A12*N1x + A26*N1y) + N3y*vy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + N3y*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N3x*vy + N3y*vx) + (N3x*vy + N3y*vx)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 2+c3] += detJ*wij*(N3x*wx*(A11*N1x + A16*N1y) + N3x*wx*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N3y*wy*(A12*N1x + A26*N1y) + N3y*wy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N3x*wy + N3y*wx) + (N3x*wy + N3y*wx)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 3+c3] += detJ*wij*(N3x*(B11*N1x*ux + B12*N1y*uy + B16*(N1x*uy + N1y*ux)) + N3y*(B16*N1x*ux + B26*N1y*uy + B66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 4+c3] += detJ*wij*(N3x*(B16*N1x*ux + B26*N1y*uy + B66*(N1x*uy + N1y*ux)) + N3y*(B12*N1x*ux + B22*N1y*uy + B26*(N1x*uy + N1y*ux)))
            KNL[0+c1, 0+c4] += detJ*wij*(N4x*ux*(A11*N1x + A16*N1y) + N4x*ux*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N4x*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N4y*uy*(A12*N1x + A26*N1y) + N4y*uy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + N4y*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N4x*uy + N4y*ux) + (N4x*uy + N4y*ux)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 1+c4] += detJ*wij*(N4x*vx*(A11*N1x + A16*N1y) + N4x*vx*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N4x*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)) + N4y*vy*(A12*N1x + A26*N1y) + N4y*vy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + N4y*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N4x*vy + N4y*vx) + (N4x*vy + N4y*vx)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 2+c4] += detJ*wij*(N4x*wx*(A11*N1x + A16*N1y) + N4x*wx*(A11*N1x*ux + A12*N1y*uy + A16*(N1x*uy + N1y*ux)) + N4y*wy*(A12*N1x + A26*N1y) + N4y*wy*(A12*N1x*ux + A22*N1y*uy + A26*(N1x*uy + N1y*ux)) + (A16*N1x + A66*N1y)*(N4x*wy + N4y*wx) + (N4x*wy + N4y*wx)*(A16*N1x*ux + A26*N1y*uy + A66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 3+c4] += detJ*wij*(N4x*(B11*N1x*ux + B12*N1y*uy + B16*(N1x*uy + N1y*ux)) + N4y*(B16*N1x*ux + B26*N1y*uy + B66*(N1x*uy + N1y*ux)))
            KNL[0+c1, 4+c4] += detJ*wij*(N4x*(B16*N1x*ux + B26*N1y*uy + B66*(N1x*uy + N1y*ux)) + N4y*(B12*N1x*ux + B22*N1y*uy + B26*(N1x*uy + N1y*ux)))
            KNL[1+c1, 0+c1] += detJ*wij*(N1x*ux*(A12*N1y + A16*N1x) + N1x*ux*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N1x*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N1y*uy*(A22*N1y + A26*N1x) + N1y*uy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + N1y*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N1x*uy + N1y*ux) + (N1x*uy + N1y*ux)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 1+c1] += detJ*wij*(N1x*vx*(A12*N1y + A16*N1x) + N1x*vx*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N1x*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)) + N1y*vy*(A22*N1y + A26*N1x) + N1y*vy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + N1y*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N1x*vy + N1y*vx) + (N1x*vy + N1y*vx)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 2+c1] += detJ*wij*(N1x*wx*(A12*N1y + A16*N1x) + N1x*wx*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N1y*wy*(A22*N1y + A26*N1x) + N1y*wy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N1x*wy + N1y*wx) + (N1x*wy + N1y*wx)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 3+c1] += detJ*wij*(N1x*(B11*N1x*vx + B12*N1y*vy + B16*(N1x*vy + N1y*vx)) + N1y*(B16*N1x*vx + B26*N1y*vy + B66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 4+c1] += detJ*wij*(N1x*(B16*N1x*vx + B26*N1y*vy + B66*(N1x*vy + N1y*vx)) + N1y*(B12*N1x*vx + B22*N1y*vy + B26*(N1x*vy + N1y*vx)))
            KNL[1+c1, 0+c2] += detJ*wij*(N2x*ux*(A12*N1y + A16*N1x) + N2x*ux*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N2x*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N2y*uy*(A22*N1y + A26*N1x) + N2y*uy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + N2y*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N2x*uy + N2y*ux) + (N2x*uy + N2y*ux)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 1+c2] += detJ*wij*(N2x*vx*(A12*N1y + A16*N1x) + N2x*vx*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N2x*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)) + N2y*vy*(A22*N1y + A26*N1x) + N2y*vy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + N2y*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N2x*vy + N2y*vx) + (N2x*vy + N2y*vx)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 2+c2] += detJ*wij*(N2x*wx*(A12*N1y + A16*N1x) + N2x*wx*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N2y*wy*(A22*N1y + A26*N1x) + N2y*wy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N2x*wy + N2y*wx) + (N2x*wy + N2y*wx)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 3+c2] += detJ*wij*(N2x*(B11*N1x*vx + B12*N1y*vy + B16*(N1x*vy + N1y*vx)) + N2y*(B16*N1x*vx + B26*N1y*vy + B66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 4+c2] += detJ*wij*(N2x*(B16*N1x*vx + B26*N1y*vy + B66*(N1x*vy + N1y*vx)) + N2y*(B12*N1x*vx + B22*N1y*vy + B26*(N1x*vy + N1y*vx)))
            KNL[1+c1, 0+c3] += detJ*wij*(N3x*ux*(A12*N1y + A16*N1x) + N3x*ux*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N3x*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N3y*uy*(A22*N1y + A26*N1x) + N3y*uy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + N3y*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N3x*uy + N3y*ux) + (N3x*uy + N3y*ux)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 1+c3] += detJ*wij*(N3x*vx*(A12*N1y + A16*N1x) + N3x*vx*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N3x*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)) + N3y*vy*(A22*N1y + A26*N1x) + N3y*vy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + N3y*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N3x*vy + N3y*vx) + (N3x*vy + N3y*vx)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 2+c3] += detJ*wij*(N3x*wx*(A12*N1y + A16*N1x) + N3x*wx*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N3y*wy*(A22*N1y + A26*N1x) + N3y*wy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N3x*wy + N3y*wx) + (N3x*wy + N3y*wx)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 3+c3] += detJ*wij*(N3x*(B11*N1x*vx + B12*N1y*vy + B16*(N1x*vy + N1y*vx)) + N3y*(B16*N1x*vx + B26*N1y*vy + B66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 4+c3] += detJ*wij*(N3x*(B16*N1x*vx + B26*N1y*vy + B66*(N1x*vy + N1y*vx)) + N3y*(B12*N1x*vx + B22*N1y*vy + B26*(N1x*vy + N1y*vx)))
            KNL[1+c1, 0+c4] += detJ*wij*(N4x*ux*(A12*N1y + A16*N1x) + N4x*ux*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N4x*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N4y*uy*(A22*N1y + A26*N1x) + N4y*uy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + N4y*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N4x*uy + N4y*ux) + (N4x*uy + N4y*ux)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 1+c4] += detJ*wij*(N4x*vx*(A12*N1y + A16*N1x) + N4x*vx*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N4x*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)) + N4y*vy*(A22*N1y + A26*N1x) + N4y*vy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + N4y*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N4x*vy + N4y*vx) + (N4x*vy + N4y*vx)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 2+c4] += detJ*wij*(N4x*wx*(A12*N1y + A16*N1x) + N4x*wx*(A11*N1x*vx + A12*N1y*vy + A16*(N1x*vy + N1y*vx)) + N4y*wy*(A22*N1y + A26*N1x) + N4y*wy*(A12*N1x*vx + A22*N1y*vy + A26*(N1x*vy + N1y*vx)) + (A26*N1y + A66*N1x)*(N4x*wy + N4y*wx) + (N4x*wy + N4y*wx)*(A16*N1x*vx + A26*N1y*vy + A66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 3+c4] += detJ*wij*(N4x*(B11*N1x*vx + B12*N1y*vy + B16*(N1x*vy + N1y*vx)) + N4y*(B16*N1x*vx + B26*N1y*vy + B66*(N1x*vy + N1y*vx)))
            KNL[1+c1, 4+c4] += detJ*wij*(N4x*(B16*N1x*vx + B26*N1y*vy + B66*(N1x*vy + N1y*vx)) + N4y*(B12*N1x*vx + B22*N1y*vy + B26*(N1x*vy + N1y*vx)))
            KNL[2+c1, 0+c1] += detJ*wij*(N1x*ux*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N1x*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N1y*uy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + N1y*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)) + (N1x*uy + N1y*ux)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 1+c1] += detJ*wij*(N1x*vx*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N1x*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)) + N1y*vy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + N1y*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + (N1x*vy + N1y*vx)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 2+c1] += detJ*wij*(N1x*wx*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N1y*wy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + (N1x*wy + N1y*wx)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 3+c1] += detJ*wij*(N1x*(B11*N1x*wx + B12*N1y*wy + B16*(N1x*wy + N1y*wx)) + N1y*(B16*N1x*wx + B26*N1y*wy + B66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 4+c1] += detJ*wij*(N1x*(B16*N1x*wx + B26*N1y*wy + B66*(N1x*wy + N1y*wx)) + N1y*(B12*N1x*wx + B22*N1y*wy + B26*(N1x*wy + N1y*wx)))
            KNL[2+c1, 0+c2] += detJ*wij*(N2x*ux*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N2x*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N2y*uy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + N2y*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)) + (N2x*uy + N2y*ux)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 1+c2] += detJ*wij*(N2x*vx*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N2x*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)) + N2y*vy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + N2y*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + (N2x*vy + N2y*vx)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 2+c2] += detJ*wij*(N2x*wx*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N2y*wy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + (N2x*wy + N2y*wx)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 3+c2] += detJ*wij*(N2x*(B11*N1x*wx + B12*N1y*wy + B16*(N1x*wy + N1y*wx)) + N2y*(B16*N1x*wx + B26*N1y*wy + B66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 4+c2] += detJ*wij*(N2x*(B16*N1x*wx + B26*N1y*wy + B66*(N1x*wy + N1y*wx)) + N2y*(B12*N1x*wx + B22*N1y*wy + B26*(N1x*wy + N1y*wx)))
            KNL[2+c1, 0+c3] += detJ*wij*(N3x*ux*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N3x*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N3y*uy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + N3y*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)) + (N3x*uy + N3y*ux)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 1+c3] += detJ*wij*(N3x*vx*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N3x*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)) + N3y*vy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + N3y*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + (N3x*vy + N3y*vx)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 2+c3] += detJ*wij*(N3x*wx*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N3y*wy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + (N3x*wy + N3y*wx)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 3+c3] += detJ*wij*(N3x*(B11*N1x*wx + B12*N1y*wy + B16*(N1x*wy + N1y*wx)) + N3y*(B16*N1x*wx + B26*N1y*wy + B66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 4+c3] += detJ*wij*(N3x*(B16*N1x*wx + B26*N1y*wy + B66*(N1x*wy + N1y*wx)) + N3y*(B12*N1x*wx + B22*N1y*wy + B26*(N1x*wy + N1y*wx)))
            KNL[2+c1, 0+c4] += detJ*wij*(N4x*ux*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N4x*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N4y*uy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + N4y*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)) + (N4x*uy + N4y*ux)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 1+c4] += detJ*wij*(N4x*vx*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N4x*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)) + N4y*vy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + N4y*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + (N4x*vy + N4y*vx)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 2+c4] += detJ*wij*(N4x*wx*(A11*N1x*wx + A12*N1y*wy + A16*(N1x*wy + N1y*wx)) + N4y*wy*(A12*N1x*wx + A22*N1y*wy + A26*(N1x*wy + N1y*wx)) + (N4x*wy + N4y*wx)*(A16*N1x*wx + A26*N1y*wy + A66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 3+c4] += detJ*wij*(N4x*(B11*N1x*wx + B12*N1y*wy + B16*(N1x*wy + N1y*wx)) + N4y*(B16*N1x*wx + B26*N1y*wy + B66*(N1x*wy + N1y*wx)))
            KNL[2+c1, 4+c4] += detJ*wij*(N4x*(B16*N1x*wx + B26*N1y*wy + B66*(N1x*wy + N1y*wx)) + N4y*(B12*N1x*wx + B22*N1y*wy + B26*(N1x*wy + N1y*wx)))
            KNL[3+c1, 0+c1] += detJ*wij*(N1x*ux*(B11*N1x + B16*N1y) + N1y*uy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N1x*uy + N1y*ux))
            KNL[3+c1, 1+c1] += detJ*wij*(N1x*vx*(B11*N1x + B16*N1y) + N1y*vy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N1x*vy + N1y*vx))
            KNL[3+c1, 2+c1] += detJ*wij*(N1x*wx*(B11*N1x + B16*N1y) + N1y*wy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N1x*wy + N1y*wx))
            KNL[3+c1, 0+c2] += detJ*wij*(N2x*ux*(B11*N1x + B16*N1y) + N2y*uy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N2x*uy + N2y*ux))
            KNL[3+c1, 1+c2] += detJ*wij*(N2x*vx*(B11*N1x + B16*N1y) + N2y*vy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N2x*vy + N2y*vx))
            KNL[3+c1, 2+c2] += detJ*wij*(N2x*wx*(B11*N1x + B16*N1y) + N2y*wy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N2x*wy + N2y*wx))
            KNL[3+c1, 0+c3] += detJ*wij*(N3x*ux*(B11*N1x + B16*N1y) + N3y*uy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N3x*uy + N3y*ux))
            KNL[3+c1, 1+c3] += detJ*wij*(N3x*vx*(B11*N1x + B16*N1y) + N3y*vy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N3x*vy + N3y*vx))
            KNL[3+c1, 2+c3] += detJ*wij*(N3x*wx*(B11*N1x + B16*N1y) + N3y*wy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N3x*wy + N3y*wx))
            KNL[3+c1, 0+c4] += detJ*wij*(N4x*ux*(B11*N1x + B16*N1y) + N4y*uy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N4x*uy + N4y*ux))
            KNL[3+c1, 1+c4] += detJ*wij*(N4x*vx*(B11*N1x + B16*N1y) + N4y*vy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N4x*vy + N4y*vx))
            KNL[3+c1, 2+c4] += detJ*wij*(N4x*wx*(B11*N1x + B16*N1y) + N4y*wy*(B12*N1x + B26*N1y) + (B16*N1x + B66*N1y)*(N4x*wy + N4y*wx))
            KNL[4+c1, 0+c1] += detJ*wij*(N1x*ux*(B12*N1y + B16*N1x) + N1y*uy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N1x*uy + N1y*ux))
            KNL[4+c1, 1+c1] += detJ*wij*(N1x*vx*(B12*N1y + B16*N1x) + N1y*vy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N1x*vy + N1y*vx))
            KNL[4+c1, 2+c1] += detJ*wij*(N1x*wx*(B12*N1y + B16*N1x) + N1y*wy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N1x*wy + N1y*wx))
            KNL[4+c1, 0+c2] += detJ*wij*(N2x*ux*(B12*N1y + B16*N1x) + N2y*uy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N2x*uy + N2y*ux))
            KNL[4+c1, 1+c2] += detJ*wij*(N2x*vx*(B12*N1y + B16*N1x) + N2y*vy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N2x*vy + N2y*vx))
            KNL[4+c1, 2+c2] += detJ*wij*(N2x*wx*(B12*N1y + B16*N1x) + N2y*wy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N2x*wy + N2y*wx))
            KNL[4+c1, 0+c3] += detJ*wij*(N3x*ux*(B12*N1y + B16*N1x) + N3y*uy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N3x*uy + N3y*ux))
            KNL[4+c1, 1+c3] += detJ*wij*(N3x*vx*(B12*N1y + B16*N1x) + N3y*vy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N3x*vy + N3y*vx))
            KNL[4+c1, 2+c3] += detJ*wij*(N3x*wx*(B12*N1y + B16*N1x) + N3y*wy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N3x*wy + N3y*wx))
            KNL[4+c1, 0+c4] += detJ*wij*(N4x*ux*(B12*N1y + B16*N1x) + N4y*uy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N4x*uy + N4y*ux))
            KNL[4+c1, 1+c4] += detJ*wij*(N4x*vx*(B12*N1y + B16*N1x) + N4y*vy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N4x*vy + N4y*vx))
            KNL[4+c1, 2+c4] += detJ*wij*(N4x*wx*(B12*N1y + B16*N1x) + N4y*wy*(B22*N1y + B26*N1x) + (B26*N1y + B66*N1x)*(N4x*wy + N4y*wx))
            KNL[0+c2, 0+c1] += detJ*wij*(N1x*ux*(A11*N2x + A16*N2y) + N1x*ux*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N1x*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N1y*uy*(A12*N2x + A26*N2y) + N1y*uy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + N1y*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N1x*uy + N1y*ux) + (N1x*uy + N1y*ux)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 1+c1] += detJ*wij*(N1x*vx*(A11*N2x + A16*N2y) + N1x*vx*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N1x*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)) + N1y*vy*(A12*N2x + A26*N2y) + N1y*vy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + N1y*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N1x*vy + N1y*vx) + (N1x*vy + N1y*vx)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 2+c1] += detJ*wij*(N1x*wx*(A11*N2x + A16*N2y) + N1x*wx*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N1y*wy*(A12*N2x + A26*N2y) + N1y*wy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N1x*wy + N1y*wx) + (N1x*wy + N1y*wx)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 3+c1] += detJ*wij*(N1x*(B11*N2x*ux + B12*N2y*uy + B16*(N2x*uy + N2y*ux)) + N1y*(B16*N2x*ux + B26*N2y*uy + B66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 4+c1] += detJ*wij*(N1x*(B16*N2x*ux + B26*N2y*uy + B66*(N2x*uy + N2y*ux)) + N1y*(B12*N2x*ux + B22*N2y*uy + B26*(N2x*uy + N2y*ux)))
            KNL[0+c2, 0+c2] += detJ*wij*(N2x*ux*(A11*N2x + A16*N2y) + N2x*ux*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N2x*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N2y*uy*(A12*N2x + A26*N2y) + N2y*uy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + N2y*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N2x*uy + N2y*ux) + (N2x*uy + N2y*ux)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 1+c2] += detJ*wij*(N2x*vx*(A11*N2x + A16*N2y) + N2x*vx*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N2x*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)) + N2y*vy*(A12*N2x + A26*N2y) + N2y*vy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + N2y*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N2x*vy + N2y*vx) + (N2x*vy + N2y*vx)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 2+c2] += detJ*wij*(N2x*wx*(A11*N2x + A16*N2y) + N2x*wx*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N2y*wy*(A12*N2x + A26*N2y) + N2y*wy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N2x*wy + N2y*wx) + (N2x*wy + N2y*wx)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 3+c2] += detJ*wij*(N2x*(B11*N2x*ux + B12*N2y*uy + B16*(N2x*uy + N2y*ux)) + N2y*(B16*N2x*ux + B26*N2y*uy + B66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 4+c2] += detJ*wij*(N2x*(B16*N2x*ux + B26*N2y*uy + B66*(N2x*uy + N2y*ux)) + N2y*(B12*N2x*ux + B22*N2y*uy + B26*(N2x*uy + N2y*ux)))
            KNL[0+c2, 0+c3] += detJ*wij*(N3x*ux*(A11*N2x + A16*N2y) + N3x*ux*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N3x*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N3y*uy*(A12*N2x + A26*N2y) + N3y*uy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + N3y*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N3x*uy + N3y*ux) + (N3x*uy + N3y*ux)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 1+c3] += detJ*wij*(N3x*vx*(A11*N2x + A16*N2y) + N3x*vx*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N3x*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)) + N3y*vy*(A12*N2x + A26*N2y) + N3y*vy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + N3y*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N3x*vy + N3y*vx) + (N3x*vy + N3y*vx)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 2+c3] += detJ*wij*(N3x*wx*(A11*N2x + A16*N2y) + N3x*wx*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N3y*wy*(A12*N2x + A26*N2y) + N3y*wy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N3x*wy + N3y*wx) + (N3x*wy + N3y*wx)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 3+c3] += detJ*wij*(N3x*(B11*N2x*ux + B12*N2y*uy + B16*(N2x*uy + N2y*ux)) + N3y*(B16*N2x*ux + B26*N2y*uy + B66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 4+c3] += detJ*wij*(N3x*(B16*N2x*ux + B26*N2y*uy + B66*(N2x*uy + N2y*ux)) + N3y*(B12*N2x*ux + B22*N2y*uy + B26*(N2x*uy + N2y*ux)))
            KNL[0+c2, 0+c4] += detJ*wij*(N4x*ux*(A11*N2x + A16*N2y) + N4x*ux*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N4x*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N4y*uy*(A12*N2x + A26*N2y) + N4y*uy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + N4y*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N4x*uy + N4y*ux) + (N4x*uy + N4y*ux)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 1+c4] += detJ*wij*(N4x*vx*(A11*N2x + A16*N2y) + N4x*vx*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N4x*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)) + N4y*vy*(A12*N2x + A26*N2y) + N4y*vy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + N4y*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N4x*vy + N4y*vx) + (N4x*vy + N4y*vx)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 2+c4] += detJ*wij*(N4x*wx*(A11*N2x + A16*N2y) + N4x*wx*(A11*N2x*ux + A12*N2y*uy + A16*(N2x*uy + N2y*ux)) + N4y*wy*(A12*N2x + A26*N2y) + N4y*wy*(A12*N2x*ux + A22*N2y*uy + A26*(N2x*uy + N2y*ux)) + (A16*N2x + A66*N2y)*(N4x*wy + N4y*wx) + (N4x*wy + N4y*wx)*(A16*N2x*ux + A26*N2y*uy + A66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 3+c4] += detJ*wij*(N4x*(B11*N2x*ux + B12*N2y*uy + B16*(N2x*uy + N2y*ux)) + N4y*(B16*N2x*ux + B26*N2y*uy + B66*(N2x*uy + N2y*ux)))
            KNL[0+c2, 4+c4] += detJ*wij*(N4x*(B16*N2x*ux + B26*N2y*uy + B66*(N2x*uy + N2y*ux)) + N4y*(B12*N2x*ux + B22*N2y*uy + B26*(N2x*uy + N2y*ux)))
            KNL[1+c2, 0+c1] += detJ*wij*(N1x*ux*(A12*N2y + A16*N2x) + N1x*ux*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N1x*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N1y*uy*(A22*N2y + A26*N2x) + N1y*uy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + N1y*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N1x*uy + N1y*ux) + (N1x*uy + N1y*ux)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 1+c1] += detJ*wij*(N1x*vx*(A12*N2y + A16*N2x) + N1x*vx*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N1x*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)) + N1y*vy*(A22*N2y + A26*N2x) + N1y*vy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + N1y*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N1x*vy + N1y*vx) + (N1x*vy + N1y*vx)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 2+c1] += detJ*wij*(N1x*wx*(A12*N2y + A16*N2x) + N1x*wx*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N1y*wy*(A22*N2y + A26*N2x) + N1y*wy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N1x*wy + N1y*wx) + (N1x*wy + N1y*wx)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 3+c1] += detJ*wij*(N1x*(B11*N2x*vx + B12*N2y*vy + B16*(N2x*vy + N2y*vx)) + N1y*(B16*N2x*vx + B26*N2y*vy + B66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 4+c1] += detJ*wij*(N1x*(B16*N2x*vx + B26*N2y*vy + B66*(N2x*vy + N2y*vx)) + N1y*(B12*N2x*vx + B22*N2y*vy + B26*(N2x*vy + N2y*vx)))
            KNL[1+c2, 0+c2] += detJ*wij*(N2x*ux*(A12*N2y + A16*N2x) + N2x*ux*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N2x*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N2y*uy*(A22*N2y + A26*N2x) + N2y*uy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + N2y*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N2x*uy + N2y*ux) + (N2x*uy + N2y*ux)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 1+c2] += detJ*wij*(N2x*vx*(A12*N2y + A16*N2x) + N2x*vx*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N2x*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)) + N2y*vy*(A22*N2y + A26*N2x) + N2y*vy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + N2y*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N2x*vy + N2y*vx) + (N2x*vy + N2y*vx)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 2+c2] += detJ*wij*(N2x*wx*(A12*N2y + A16*N2x) + N2x*wx*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N2y*wy*(A22*N2y + A26*N2x) + N2y*wy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N2x*wy + N2y*wx) + (N2x*wy + N2y*wx)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 3+c2] += detJ*wij*(N2x*(B11*N2x*vx + B12*N2y*vy + B16*(N2x*vy + N2y*vx)) + N2y*(B16*N2x*vx + B26*N2y*vy + B66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 4+c2] += detJ*wij*(N2x*(B16*N2x*vx + B26*N2y*vy + B66*(N2x*vy + N2y*vx)) + N2y*(B12*N2x*vx + B22*N2y*vy + B26*(N2x*vy + N2y*vx)))
            KNL[1+c2, 0+c3] += detJ*wij*(N3x*ux*(A12*N2y + A16*N2x) + N3x*ux*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N3x*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N3y*uy*(A22*N2y + A26*N2x) + N3y*uy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + N3y*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N3x*uy + N3y*ux) + (N3x*uy + N3y*ux)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 1+c3] += detJ*wij*(N3x*vx*(A12*N2y + A16*N2x) + N3x*vx*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N3x*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)) + N3y*vy*(A22*N2y + A26*N2x) + N3y*vy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + N3y*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N3x*vy + N3y*vx) + (N3x*vy + N3y*vx)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 2+c3] += detJ*wij*(N3x*wx*(A12*N2y + A16*N2x) + N3x*wx*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N3y*wy*(A22*N2y + A26*N2x) + N3y*wy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N3x*wy + N3y*wx) + (N3x*wy + N3y*wx)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 3+c3] += detJ*wij*(N3x*(B11*N2x*vx + B12*N2y*vy + B16*(N2x*vy + N2y*vx)) + N3y*(B16*N2x*vx + B26*N2y*vy + B66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 4+c3] += detJ*wij*(N3x*(B16*N2x*vx + B26*N2y*vy + B66*(N2x*vy + N2y*vx)) + N3y*(B12*N2x*vx + B22*N2y*vy + B26*(N2x*vy + N2y*vx)))
            KNL[1+c2, 0+c4] += detJ*wij*(N4x*ux*(A12*N2y + A16*N2x) + N4x*ux*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N4x*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N4y*uy*(A22*N2y + A26*N2x) + N4y*uy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + N4y*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N4x*uy + N4y*ux) + (N4x*uy + N4y*ux)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 1+c4] += detJ*wij*(N4x*vx*(A12*N2y + A16*N2x) + N4x*vx*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N4x*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)) + N4y*vy*(A22*N2y + A26*N2x) + N4y*vy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + N4y*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N4x*vy + N4y*vx) + (N4x*vy + N4y*vx)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 2+c4] += detJ*wij*(N4x*wx*(A12*N2y + A16*N2x) + N4x*wx*(A11*N2x*vx + A12*N2y*vy + A16*(N2x*vy + N2y*vx)) + N4y*wy*(A22*N2y + A26*N2x) + N4y*wy*(A12*N2x*vx + A22*N2y*vy + A26*(N2x*vy + N2y*vx)) + (A26*N2y + A66*N2x)*(N4x*wy + N4y*wx) + (N4x*wy + N4y*wx)*(A16*N2x*vx + A26*N2y*vy + A66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 3+c4] += detJ*wij*(N4x*(B11*N2x*vx + B12*N2y*vy + B16*(N2x*vy + N2y*vx)) + N4y*(B16*N2x*vx + B26*N2y*vy + B66*(N2x*vy + N2y*vx)))
            KNL[1+c2, 4+c4] += detJ*wij*(N4x*(B16*N2x*vx + B26*N2y*vy + B66*(N2x*vy + N2y*vx)) + N4y*(B12*N2x*vx + B22*N2y*vy + B26*(N2x*vy + N2y*vx)))
            KNL[2+c2, 0+c1] += detJ*wij*(N1x*ux*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N1x*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N1y*uy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + N1y*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)) + (N1x*uy + N1y*ux)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 1+c1] += detJ*wij*(N1x*vx*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N1x*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)) + N1y*vy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + N1y*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + (N1x*vy + N1y*vx)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 2+c1] += detJ*wij*(N1x*wx*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N1y*wy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + (N1x*wy + N1y*wx)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 3+c1] += detJ*wij*(N1x*(B11*N2x*wx + B12*N2y*wy + B16*(N2x*wy + N2y*wx)) + N1y*(B16*N2x*wx + B26*N2y*wy + B66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 4+c1] += detJ*wij*(N1x*(B16*N2x*wx + B26*N2y*wy + B66*(N2x*wy + N2y*wx)) + N1y*(B12*N2x*wx + B22*N2y*wy + B26*(N2x*wy + N2y*wx)))
            KNL[2+c2, 0+c2] += detJ*wij*(N2x*ux*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N2x*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N2y*uy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + N2y*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)) + (N2x*uy + N2y*ux)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 1+c2] += detJ*wij*(N2x*vx*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N2x*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)) + N2y*vy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + N2y*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + (N2x*vy + N2y*vx)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 2+c2] += detJ*wij*(N2x*wx*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N2y*wy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + (N2x*wy + N2y*wx)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 3+c2] += detJ*wij*(N2x*(B11*N2x*wx + B12*N2y*wy + B16*(N2x*wy + N2y*wx)) + N2y*(B16*N2x*wx + B26*N2y*wy + B66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 4+c2] += detJ*wij*(N2x*(B16*N2x*wx + B26*N2y*wy + B66*(N2x*wy + N2y*wx)) + N2y*(B12*N2x*wx + B22*N2y*wy + B26*(N2x*wy + N2y*wx)))
            KNL[2+c2, 0+c3] += detJ*wij*(N3x*ux*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N3x*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N3y*uy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + N3y*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)) + (N3x*uy + N3y*ux)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 1+c3] += detJ*wij*(N3x*vx*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N3x*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)) + N3y*vy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + N3y*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + (N3x*vy + N3y*vx)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 2+c3] += detJ*wij*(N3x*wx*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N3y*wy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + (N3x*wy + N3y*wx)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 3+c3] += detJ*wij*(N3x*(B11*N2x*wx + B12*N2y*wy + B16*(N2x*wy + N2y*wx)) + N3y*(B16*N2x*wx + B26*N2y*wy + B66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 4+c3] += detJ*wij*(N3x*(B16*N2x*wx + B26*N2y*wy + B66*(N2x*wy + N2y*wx)) + N3y*(B12*N2x*wx + B22*N2y*wy + B26*(N2x*wy + N2y*wx)))
            KNL[2+c2, 0+c4] += detJ*wij*(N4x*ux*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N4x*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N4y*uy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + N4y*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)) + (N4x*uy + N4y*ux)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 1+c4] += detJ*wij*(N4x*vx*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N4x*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)) + N4y*vy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + N4y*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + (N4x*vy + N4y*vx)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 2+c4] += detJ*wij*(N4x*wx*(A11*N2x*wx + A12*N2y*wy + A16*(N2x*wy + N2y*wx)) + N4y*wy*(A12*N2x*wx + A22*N2y*wy + A26*(N2x*wy + N2y*wx)) + (N4x*wy + N4y*wx)*(A16*N2x*wx + A26*N2y*wy + A66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 3+c4] += detJ*wij*(N4x*(B11*N2x*wx + B12*N2y*wy + B16*(N2x*wy + N2y*wx)) + N4y*(B16*N2x*wx + B26*N2y*wy + B66*(N2x*wy + N2y*wx)))
            KNL[2+c2, 4+c4] += detJ*wij*(N4x*(B16*N2x*wx + B26*N2y*wy + B66*(N2x*wy + N2y*wx)) + N4y*(B12*N2x*wx + B22*N2y*wy + B26*(N2x*wy + N2y*wx)))
            KNL[3+c2, 0+c1] += detJ*wij*(N1x*ux*(B11*N2x + B16*N2y) + N1y*uy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N1x*uy + N1y*ux))
            KNL[3+c2, 1+c1] += detJ*wij*(N1x*vx*(B11*N2x + B16*N2y) + N1y*vy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N1x*vy + N1y*vx))
            KNL[3+c2, 2+c1] += detJ*wij*(N1x*wx*(B11*N2x + B16*N2y) + N1y*wy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N1x*wy + N1y*wx))
            KNL[3+c2, 0+c2] += detJ*wij*(N2x*ux*(B11*N2x + B16*N2y) + N2y*uy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N2x*uy + N2y*ux))
            KNL[3+c2, 1+c2] += detJ*wij*(N2x*vx*(B11*N2x + B16*N2y) + N2y*vy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N2x*vy + N2y*vx))
            KNL[3+c2, 2+c2] += detJ*wij*(N2x*wx*(B11*N2x + B16*N2y) + N2y*wy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N2x*wy + N2y*wx))
            KNL[3+c2, 0+c3] += detJ*wij*(N3x*ux*(B11*N2x + B16*N2y) + N3y*uy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N3x*uy + N3y*ux))
            KNL[3+c2, 1+c3] += detJ*wij*(N3x*vx*(B11*N2x + B16*N2y) + N3y*vy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N3x*vy + N3y*vx))
            KNL[3+c2, 2+c3] += detJ*wij*(N3x*wx*(B11*N2x + B16*N2y) + N3y*wy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N3x*wy + N3y*wx))
            KNL[3+c2, 0+c4] += detJ*wij*(N4x*ux*(B11*N2x + B16*N2y) + N4y*uy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N4x*uy + N4y*ux))
            KNL[3+c2, 1+c4] += detJ*wij*(N4x*vx*(B11*N2x + B16*N2y) + N4y*vy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N4x*vy + N4y*vx))
            KNL[3+c2, 2+c4] += detJ*wij*(N4x*wx*(B11*N2x + B16*N2y) + N4y*wy*(B12*N2x + B26*N2y) + (B16*N2x + B66*N2y)*(N4x*wy + N4y*wx))
            KNL[4+c2, 0+c1] += detJ*wij*(N1x*ux*(B12*N2y + B16*N2x) + N1y*uy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N1x*uy + N1y*ux))
            KNL[4+c2, 1+c1] += detJ*wij*(N1x*vx*(B12*N2y + B16*N2x) + N1y*vy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N1x*vy + N1y*vx))
            KNL[4+c2, 2+c1] += detJ*wij*(N1x*wx*(B12*N2y + B16*N2x) + N1y*wy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N1x*wy + N1y*wx))
            KNL[4+c2, 0+c2] += detJ*wij*(N2x*ux*(B12*N2y + B16*N2x) + N2y*uy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N2x*uy + N2y*ux))
            KNL[4+c2, 1+c2] += detJ*wij*(N2x*vx*(B12*N2y + B16*N2x) + N2y*vy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N2x*vy + N2y*vx))
            KNL[4+c2, 2+c2] += detJ*wij*(N2x*wx*(B12*N2y + B16*N2x) + N2y*wy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N2x*wy + N2y*wx))
            KNL[4+c2, 0+c3] += detJ*wij*(N3x*ux*(B12*N2y + B16*N2x) + N3y*uy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N3x*uy + N3y*ux))
            KNL[4+c2, 1+c3] += detJ*wij*(N3x*vx*(B12*N2y + B16*N2x) + N3y*vy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N3x*vy + N3y*vx))
            KNL[4+c2, 2+c3] += detJ*wij*(N3x*wx*(B12*N2y + B16*N2x) + N3y*wy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N3x*wy + N3y*wx))
            KNL[4+c2, 0+c4] += detJ*wij*(N4x*ux*(B12*N2y + B16*N2x) + N4y*uy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N4x*uy + N4y*ux))
            KNL[4+c2, 1+c4] += detJ*wij*(N4x*vx*(B12*N2y + B16*N2x) + N4y*vy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N4x*vy + N4y*vx))
            KNL[4+c2, 2+c4] += detJ*wij*(N4x*wx*(B12*N2y + B16*N2x) + N4y*wy*(B22*N2y + B26*N2x) + (B26*N2y + B66*N2x)*(N4x*wy + N4y*wx))
            KNL[0+c3, 0+c1] += detJ*wij*(N1x*ux*(A11*N3x + A16*N3y) + N1x*ux*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N1x*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N1y*uy*(A12*N3x + A26*N3y) + N1y*uy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + N1y*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N1x*uy + N1y*ux) + (N1x*uy + N1y*ux)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 1+c1] += detJ*wij*(N1x*vx*(A11*N3x + A16*N3y) + N1x*vx*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N1x*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)) + N1y*vy*(A12*N3x + A26*N3y) + N1y*vy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + N1y*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N1x*vy + N1y*vx) + (N1x*vy + N1y*vx)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 2+c1] += detJ*wij*(N1x*wx*(A11*N3x + A16*N3y) + N1x*wx*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N1y*wy*(A12*N3x + A26*N3y) + N1y*wy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N1x*wy + N1y*wx) + (N1x*wy + N1y*wx)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 3+c1] += detJ*wij*(N1x*(B11*N3x*ux + B12*N3y*uy + B16*(N3x*uy + N3y*ux)) + N1y*(B16*N3x*ux + B26*N3y*uy + B66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 4+c1] += detJ*wij*(N1x*(B16*N3x*ux + B26*N3y*uy + B66*(N3x*uy + N3y*ux)) + N1y*(B12*N3x*ux + B22*N3y*uy + B26*(N3x*uy + N3y*ux)))
            KNL[0+c3, 0+c2] += detJ*wij*(N2x*ux*(A11*N3x + A16*N3y) + N2x*ux*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N2x*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N2y*uy*(A12*N3x + A26*N3y) + N2y*uy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + N2y*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N2x*uy + N2y*ux) + (N2x*uy + N2y*ux)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 1+c2] += detJ*wij*(N2x*vx*(A11*N3x + A16*N3y) + N2x*vx*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N2x*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)) + N2y*vy*(A12*N3x + A26*N3y) + N2y*vy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + N2y*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N2x*vy + N2y*vx) + (N2x*vy + N2y*vx)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 2+c2] += detJ*wij*(N2x*wx*(A11*N3x + A16*N3y) + N2x*wx*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N2y*wy*(A12*N3x + A26*N3y) + N2y*wy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N2x*wy + N2y*wx) + (N2x*wy + N2y*wx)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 3+c2] += detJ*wij*(N2x*(B11*N3x*ux + B12*N3y*uy + B16*(N3x*uy + N3y*ux)) + N2y*(B16*N3x*ux + B26*N3y*uy + B66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 4+c2] += detJ*wij*(N2x*(B16*N3x*ux + B26*N3y*uy + B66*(N3x*uy + N3y*ux)) + N2y*(B12*N3x*ux + B22*N3y*uy + B26*(N3x*uy + N3y*ux)))
            KNL[0+c3, 0+c3] += detJ*wij*(N3x*ux*(A11*N3x + A16*N3y) + N3x*ux*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N3x*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N3y*uy*(A12*N3x + A26*N3y) + N3y*uy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + N3y*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N3x*uy + N3y*ux) + (N3x*uy + N3y*ux)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 1+c3] += detJ*wij*(N3x*vx*(A11*N3x + A16*N3y) + N3x*vx*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N3x*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)) + N3y*vy*(A12*N3x + A26*N3y) + N3y*vy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + N3y*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N3x*vy + N3y*vx) + (N3x*vy + N3y*vx)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 2+c3] += detJ*wij*(N3x*wx*(A11*N3x + A16*N3y) + N3x*wx*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N3y*wy*(A12*N3x + A26*N3y) + N3y*wy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N3x*wy + N3y*wx) + (N3x*wy + N3y*wx)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 3+c3] += detJ*wij*(N3x*(B11*N3x*ux + B12*N3y*uy + B16*(N3x*uy + N3y*ux)) + N3y*(B16*N3x*ux + B26*N3y*uy + B66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 4+c3] += detJ*wij*(N3x*(B16*N3x*ux + B26*N3y*uy + B66*(N3x*uy + N3y*ux)) + N3y*(B12*N3x*ux + B22*N3y*uy + B26*(N3x*uy + N3y*ux)))
            KNL[0+c3, 0+c4] += detJ*wij*(N4x*ux*(A11*N3x + A16*N3y) + N4x*ux*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N4x*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N4y*uy*(A12*N3x + A26*N3y) + N4y*uy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + N4y*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N4x*uy + N4y*ux) + (N4x*uy + N4y*ux)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 1+c4] += detJ*wij*(N4x*vx*(A11*N3x + A16*N3y) + N4x*vx*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N4x*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)) + N4y*vy*(A12*N3x + A26*N3y) + N4y*vy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + N4y*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N4x*vy + N4y*vx) + (N4x*vy + N4y*vx)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 2+c4] += detJ*wij*(N4x*wx*(A11*N3x + A16*N3y) + N4x*wx*(A11*N3x*ux + A12*N3y*uy + A16*(N3x*uy + N3y*ux)) + N4y*wy*(A12*N3x + A26*N3y) + N4y*wy*(A12*N3x*ux + A22*N3y*uy + A26*(N3x*uy + N3y*ux)) + (A16*N3x + A66*N3y)*(N4x*wy + N4y*wx) + (N4x*wy + N4y*wx)*(A16*N3x*ux + A26*N3y*uy + A66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 3+c4] += detJ*wij*(N4x*(B11*N3x*ux + B12*N3y*uy + B16*(N3x*uy + N3y*ux)) + N4y*(B16*N3x*ux + B26*N3y*uy + B66*(N3x*uy + N3y*ux)))
            KNL[0+c3, 4+c4] += detJ*wij*(N4x*(B16*N3x*ux + B26*N3y*uy + B66*(N3x*uy + N3y*ux)) + N4y*(B12*N3x*ux + B22*N3y*uy + B26*(N3x*uy + N3y*ux)))
            KNL[1+c3, 0+c1] += detJ*wij*(N1x*ux*(A12*N3y + A16*N3x) + N1x*ux*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N1x*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N1y*uy*(A22*N3y + A26*N3x) + N1y*uy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + N1y*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N1x*uy + N1y*ux) + (N1x*uy + N1y*ux)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 1+c1] += detJ*wij*(N1x*vx*(A12*N3y + A16*N3x) + N1x*vx*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N1x*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)) + N1y*vy*(A22*N3y + A26*N3x) + N1y*vy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + N1y*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N1x*vy + N1y*vx) + (N1x*vy + N1y*vx)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 2+c1] += detJ*wij*(N1x*wx*(A12*N3y + A16*N3x) + N1x*wx*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N1y*wy*(A22*N3y + A26*N3x) + N1y*wy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N1x*wy + N1y*wx) + (N1x*wy + N1y*wx)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 3+c1] += detJ*wij*(N1x*(B11*N3x*vx + B12*N3y*vy + B16*(N3x*vy + N3y*vx)) + N1y*(B16*N3x*vx + B26*N3y*vy + B66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 4+c1] += detJ*wij*(N1x*(B16*N3x*vx + B26*N3y*vy + B66*(N3x*vy + N3y*vx)) + N1y*(B12*N3x*vx + B22*N3y*vy + B26*(N3x*vy + N3y*vx)))
            KNL[1+c3, 0+c2] += detJ*wij*(N2x*ux*(A12*N3y + A16*N3x) + N2x*ux*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N2x*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N2y*uy*(A22*N3y + A26*N3x) + N2y*uy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + N2y*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N2x*uy + N2y*ux) + (N2x*uy + N2y*ux)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 1+c2] += detJ*wij*(N2x*vx*(A12*N3y + A16*N3x) + N2x*vx*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N2x*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)) + N2y*vy*(A22*N3y + A26*N3x) + N2y*vy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + N2y*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N2x*vy + N2y*vx) + (N2x*vy + N2y*vx)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 2+c2] += detJ*wij*(N2x*wx*(A12*N3y + A16*N3x) + N2x*wx*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N2y*wy*(A22*N3y + A26*N3x) + N2y*wy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N2x*wy + N2y*wx) + (N2x*wy + N2y*wx)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 3+c2] += detJ*wij*(N2x*(B11*N3x*vx + B12*N3y*vy + B16*(N3x*vy + N3y*vx)) + N2y*(B16*N3x*vx + B26*N3y*vy + B66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 4+c2] += detJ*wij*(N2x*(B16*N3x*vx + B26*N3y*vy + B66*(N3x*vy + N3y*vx)) + N2y*(B12*N3x*vx + B22*N3y*vy + B26*(N3x*vy + N3y*vx)))
            KNL[1+c3, 0+c3] += detJ*wij*(N3x*ux*(A12*N3y + A16*N3x) + N3x*ux*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N3x*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N3y*uy*(A22*N3y + A26*N3x) + N3y*uy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + N3y*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N3x*uy + N3y*ux) + (N3x*uy + N3y*ux)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 1+c3] += detJ*wij*(N3x*vx*(A12*N3y + A16*N3x) + N3x*vx*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N3x*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)) + N3y*vy*(A22*N3y + A26*N3x) + N3y*vy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + N3y*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N3x*vy + N3y*vx) + (N3x*vy + N3y*vx)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 2+c3] += detJ*wij*(N3x*wx*(A12*N3y + A16*N3x) + N3x*wx*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N3y*wy*(A22*N3y + A26*N3x) + N3y*wy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N3x*wy + N3y*wx) + (N3x*wy + N3y*wx)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 3+c3] += detJ*wij*(N3x*(B11*N3x*vx + B12*N3y*vy + B16*(N3x*vy + N3y*vx)) + N3y*(B16*N3x*vx + B26*N3y*vy + B66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 4+c3] += detJ*wij*(N3x*(B16*N3x*vx + B26*N3y*vy + B66*(N3x*vy + N3y*vx)) + N3y*(B12*N3x*vx + B22*N3y*vy + B26*(N3x*vy + N3y*vx)))
            KNL[1+c3, 0+c4] += detJ*wij*(N4x*ux*(A12*N3y + A16*N3x) + N4x*ux*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N4x*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N4y*uy*(A22*N3y + A26*N3x) + N4y*uy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + N4y*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N4x*uy + N4y*ux) + (N4x*uy + N4y*ux)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 1+c4] += detJ*wij*(N4x*vx*(A12*N3y + A16*N3x) + N4x*vx*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N4x*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)) + N4y*vy*(A22*N3y + A26*N3x) + N4y*vy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + N4y*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N4x*vy + N4y*vx) + (N4x*vy + N4y*vx)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 2+c4] += detJ*wij*(N4x*wx*(A12*N3y + A16*N3x) + N4x*wx*(A11*N3x*vx + A12*N3y*vy + A16*(N3x*vy + N3y*vx)) + N4y*wy*(A22*N3y + A26*N3x) + N4y*wy*(A12*N3x*vx + A22*N3y*vy + A26*(N3x*vy + N3y*vx)) + (A26*N3y + A66*N3x)*(N4x*wy + N4y*wx) + (N4x*wy + N4y*wx)*(A16*N3x*vx + A26*N3y*vy + A66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 3+c4] += detJ*wij*(N4x*(B11*N3x*vx + B12*N3y*vy + B16*(N3x*vy + N3y*vx)) + N4y*(B16*N3x*vx + B26*N3y*vy + B66*(N3x*vy + N3y*vx)))
            KNL[1+c3, 4+c4] += detJ*wij*(N4x*(B16*N3x*vx + B26*N3y*vy + B66*(N3x*vy + N3y*vx)) + N4y*(B12*N3x*vx + B22*N3y*vy + B26*(N3x*vy + N3y*vx)))
            KNL[2+c3, 0+c1] += detJ*wij*(N1x*ux*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N1x*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N1y*uy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + N1y*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)) + (N1x*uy + N1y*ux)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 1+c1] += detJ*wij*(N1x*vx*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N1x*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)) + N1y*vy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + N1y*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + (N1x*vy + N1y*vx)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 2+c1] += detJ*wij*(N1x*wx*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N1y*wy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + (N1x*wy + N1y*wx)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 3+c1] += detJ*wij*(N1x*(B11*N3x*wx + B12*N3y*wy + B16*(N3x*wy + N3y*wx)) + N1y*(B16*N3x*wx + B26*N3y*wy + B66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 4+c1] += detJ*wij*(N1x*(B16*N3x*wx + B26*N3y*wy + B66*(N3x*wy + N3y*wx)) + N1y*(B12*N3x*wx + B22*N3y*wy + B26*(N3x*wy + N3y*wx)))
            KNL[2+c3, 0+c2] += detJ*wij*(N2x*ux*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N2x*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N2y*uy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + N2y*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)) + (N2x*uy + N2y*ux)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 1+c2] += detJ*wij*(N2x*vx*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N2x*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)) + N2y*vy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + N2y*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + (N2x*vy + N2y*vx)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 2+c2] += detJ*wij*(N2x*wx*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N2y*wy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + (N2x*wy + N2y*wx)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 3+c2] += detJ*wij*(N2x*(B11*N3x*wx + B12*N3y*wy + B16*(N3x*wy + N3y*wx)) + N2y*(B16*N3x*wx + B26*N3y*wy + B66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 4+c2] += detJ*wij*(N2x*(B16*N3x*wx + B26*N3y*wy + B66*(N3x*wy + N3y*wx)) + N2y*(B12*N3x*wx + B22*N3y*wy + B26*(N3x*wy + N3y*wx)))
            KNL[2+c3, 0+c3] += detJ*wij*(N3x*ux*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N3x*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N3y*uy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + N3y*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)) + (N3x*uy + N3y*ux)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 1+c3] += detJ*wij*(N3x*vx*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N3x*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)) + N3y*vy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + N3y*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + (N3x*vy + N3y*vx)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 2+c3] += detJ*wij*(N3x*wx*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N3y*wy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + (N3x*wy + N3y*wx)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 3+c3] += detJ*wij*(N3x*(B11*N3x*wx + B12*N3y*wy + B16*(N3x*wy + N3y*wx)) + N3y*(B16*N3x*wx + B26*N3y*wy + B66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 4+c3] += detJ*wij*(N3x*(B16*N3x*wx + B26*N3y*wy + B66*(N3x*wy + N3y*wx)) + N3y*(B12*N3x*wx + B22*N3y*wy + B26*(N3x*wy + N3y*wx)))
            KNL[2+c3, 0+c4] += detJ*wij*(N4x*ux*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N4x*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N4y*uy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + N4y*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)) + (N4x*uy + N4y*ux)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 1+c4] += detJ*wij*(N4x*vx*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N4x*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)) + N4y*vy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + N4y*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + (N4x*vy + N4y*vx)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 2+c4] += detJ*wij*(N4x*wx*(A11*N3x*wx + A12*N3y*wy + A16*(N3x*wy + N3y*wx)) + N4y*wy*(A12*N3x*wx + A22*N3y*wy + A26*(N3x*wy + N3y*wx)) + (N4x*wy + N4y*wx)*(A16*N3x*wx + A26*N3y*wy + A66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 3+c4] += detJ*wij*(N4x*(B11*N3x*wx + B12*N3y*wy + B16*(N3x*wy + N3y*wx)) + N4y*(B16*N3x*wx + B26*N3y*wy + B66*(N3x*wy + N3y*wx)))
            KNL[2+c3, 4+c4] += detJ*wij*(N4x*(B16*N3x*wx + B26*N3y*wy + B66*(N3x*wy + N3y*wx)) + N4y*(B12*N3x*wx + B22*N3y*wy + B26*(N3x*wy + N3y*wx)))
            KNL[3+c3, 0+c1] += detJ*wij*(N1x*ux*(B11*N3x + B16*N3y) + N1y*uy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N1x*uy + N1y*ux))
            KNL[3+c3, 1+c1] += detJ*wij*(N1x*vx*(B11*N3x + B16*N3y) + N1y*vy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N1x*vy + N1y*vx))
            KNL[3+c3, 2+c1] += detJ*wij*(N1x*wx*(B11*N3x + B16*N3y) + N1y*wy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N1x*wy + N1y*wx))
            KNL[3+c3, 0+c2] += detJ*wij*(N2x*ux*(B11*N3x + B16*N3y) + N2y*uy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N2x*uy + N2y*ux))
            KNL[3+c3, 1+c2] += detJ*wij*(N2x*vx*(B11*N3x + B16*N3y) + N2y*vy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N2x*vy + N2y*vx))
            KNL[3+c3, 2+c2] += detJ*wij*(N2x*wx*(B11*N3x + B16*N3y) + N2y*wy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N2x*wy + N2y*wx))
            KNL[3+c3, 0+c3] += detJ*wij*(N3x*ux*(B11*N3x + B16*N3y) + N3y*uy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N3x*uy + N3y*ux))
            KNL[3+c3, 1+c3] += detJ*wij*(N3x*vx*(B11*N3x + B16*N3y) + N3y*vy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N3x*vy + N3y*vx))
            KNL[3+c3, 2+c3] += detJ*wij*(N3x*wx*(B11*N3x + B16*N3y) + N3y*wy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N3x*wy + N3y*wx))
            KNL[3+c3, 0+c4] += detJ*wij*(N4x*ux*(B11*N3x + B16*N3y) + N4y*uy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N4x*uy + N4y*ux))
            KNL[3+c3, 1+c4] += detJ*wij*(N4x*vx*(B11*N3x + B16*N3y) + N4y*vy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N4x*vy + N4y*vx))
            KNL[3+c3, 2+c4] += detJ*wij*(N4x*wx*(B11*N3x + B16*N3y) + N4y*wy*(B12*N3x + B26*N3y) + (B16*N3x + B66*N3y)*(N4x*wy + N4y*wx))
            KNL[4+c3, 0+c1] += detJ*wij*(N1x*ux*(B12*N3y + B16*N3x) + N1y*uy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N1x*uy + N1y*ux))
            KNL[4+c3, 1+c1] += detJ*wij*(N1x*vx*(B12*N3y + B16*N3x) + N1y*vy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N1x*vy + N1y*vx))
            KNL[4+c3, 2+c1] += detJ*wij*(N1x*wx*(B12*N3y + B16*N3x) + N1y*wy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N1x*wy + N1y*wx))
            KNL[4+c3, 0+c2] += detJ*wij*(N2x*ux*(B12*N3y + B16*N3x) + N2y*uy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N2x*uy + N2y*ux))
            KNL[4+c3, 1+c2] += detJ*wij*(N2x*vx*(B12*N3y + B16*N3x) + N2y*vy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N2x*vy + N2y*vx))
            KNL[4+c3, 2+c2] += detJ*wij*(N2x*wx*(B12*N3y + B16*N3x) + N2y*wy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N2x*wy + N2y*wx))
            KNL[4+c3, 0+c3] += detJ*wij*(N3x*ux*(B12*N3y + B16*N3x) + N3y*uy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N3x*uy + N3y*ux))
            KNL[4+c3, 1+c3] += detJ*wij*(N3x*vx*(B12*N3y + B16*N3x) + N3y*vy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N3x*vy + N3y*vx))
            KNL[4+c3, 2+c3] += detJ*wij*(N3x*wx*(B12*N3y + B16*N3x) + N3y*wy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N3x*wy + N3y*wx))
            KNL[4+c3, 0+c4] += detJ*wij*(N4x*ux*(B12*N3y + B16*N3x) + N4y*uy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N4x*uy + N4y*ux))
            KNL[4+c3, 1+c4] += detJ*wij*(N4x*vx*(B12*N3y + B16*N3x) + N4y*vy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N4x*vy + N4y*vx))
            KNL[4+c3, 2+c4] += detJ*wij*(N4x*wx*(B12*N3y + B16*N3x) + N4y*wy*(B22*N3y + B26*N3x) + (B26*N3y + B66*N3x)*(N4x*wy + N4y*wx))
            KNL[0+c4, 0+c1] += detJ*wij*(N1x*ux*(A11*N4x + A16*N4y) + N1x*ux*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N1x*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N1y*uy*(A12*N4x + A26*N4y) + N1y*uy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + N1y*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N1x*uy + N1y*ux) + (N1x*uy + N1y*ux)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 1+c1] += detJ*wij*(N1x*vx*(A11*N4x + A16*N4y) + N1x*vx*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N1x*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)) + N1y*vy*(A12*N4x + A26*N4y) + N1y*vy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + N1y*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N1x*vy + N1y*vx) + (N1x*vy + N1y*vx)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 2+c1] += detJ*wij*(N1x*wx*(A11*N4x + A16*N4y) + N1x*wx*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N1y*wy*(A12*N4x + A26*N4y) + N1y*wy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N1x*wy + N1y*wx) + (N1x*wy + N1y*wx)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 3+c1] += detJ*wij*(N1x*(B11*N4x*ux + B12*N4y*uy + B16*(N4x*uy + N4y*ux)) + N1y*(B16*N4x*ux + B26*N4y*uy + B66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 4+c1] += detJ*wij*(N1x*(B16*N4x*ux + B26*N4y*uy + B66*(N4x*uy + N4y*ux)) + N1y*(B12*N4x*ux + B22*N4y*uy + B26*(N4x*uy + N4y*ux)))
            KNL[0+c4, 0+c2] += detJ*wij*(N2x*ux*(A11*N4x + A16*N4y) + N2x*ux*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N2x*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N2y*uy*(A12*N4x + A26*N4y) + N2y*uy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + N2y*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N2x*uy + N2y*ux) + (N2x*uy + N2y*ux)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 1+c2] += detJ*wij*(N2x*vx*(A11*N4x + A16*N4y) + N2x*vx*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N2x*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)) + N2y*vy*(A12*N4x + A26*N4y) + N2y*vy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + N2y*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N2x*vy + N2y*vx) + (N2x*vy + N2y*vx)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 2+c2] += detJ*wij*(N2x*wx*(A11*N4x + A16*N4y) + N2x*wx*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N2y*wy*(A12*N4x + A26*N4y) + N2y*wy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N2x*wy + N2y*wx) + (N2x*wy + N2y*wx)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 3+c2] += detJ*wij*(N2x*(B11*N4x*ux + B12*N4y*uy + B16*(N4x*uy + N4y*ux)) + N2y*(B16*N4x*ux + B26*N4y*uy + B66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 4+c2] += detJ*wij*(N2x*(B16*N4x*ux + B26*N4y*uy + B66*(N4x*uy + N4y*ux)) + N2y*(B12*N4x*ux + B22*N4y*uy + B26*(N4x*uy + N4y*ux)))
            KNL[0+c4, 0+c3] += detJ*wij*(N3x*ux*(A11*N4x + A16*N4y) + N3x*ux*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N3x*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N3y*uy*(A12*N4x + A26*N4y) + N3y*uy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + N3y*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N3x*uy + N3y*ux) + (N3x*uy + N3y*ux)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 1+c3] += detJ*wij*(N3x*vx*(A11*N4x + A16*N4y) + N3x*vx*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N3x*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)) + N3y*vy*(A12*N4x + A26*N4y) + N3y*vy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + N3y*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N3x*vy + N3y*vx) + (N3x*vy + N3y*vx)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 2+c3] += detJ*wij*(N3x*wx*(A11*N4x + A16*N4y) + N3x*wx*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N3y*wy*(A12*N4x + A26*N4y) + N3y*wy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N3x*wy + N3y*wx) + (N3x*wy + N3y*wx)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 3+c3] += detJ*wij*(N3x*(B11*N4x*ux + B12*N4y*uy + B16*(N4x*uy + N4y*ux)) + N3y*(B16*N4x*ux + B26*N4y*uy + B66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 4+c3] += detJ*wij*(N3x*(B16*N4x*ux + B26*N4y*uy + B66*(N4x*uy + N4y*ux)) + N3y*(B12*N4x*ux + B22*N4y*uy + B26*(N4x*uy + N4y*ux)))
            KNL[0+c4, 0+c4] += detJ*wij*(N4x*ux*(A11*N4x + A16*N4y) + N4x*ux*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N4x*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N4y*uy*(A12*N4x + A26*N4y) + N4y*uy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + N4y*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N4x*uy + N4y*ux) + (N4x*uy + N4y*ux)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 1+c4] += detJ*wij*(N4x*vx*(A11*N4x + A16*N4y) + N4x*vx*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N4x*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)) + N4y*vy*(A12*N4x + A26*N4y) + N4y*vy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + N4y*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N4x*vy + N4y*vx) + (N4x*vy + N4y*vx)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 2+c4] += detJ*wij*(N4x*wx*(A11*N4x + A16*N4y) + N4x*wx*(A11*N4x*ux + A12*N4y*uy + A16*(N4x*uy + N4y*ux)) + N4y*wy*(A12*N4x + A26*N4y) + N4y*wy*(A12*N4x*ux + A22*N4y*uy + A26*(N4x*uy + N4y*ux)) + (A16*N4x + A66*N4y)*(N4x*wy + N4y*wx) + (N4x*wy + N4y*wx)*(A16*N4x*ux + A26*N4y*uy + A66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 3+c4] += detJ*wij*(N4x*(B11*N4x*ux + B12*N4y*uy + B16*(N4x*uy + N4y*ux)) + N4y*(B16*N4x*ux + B26*N4y*uy + B66*(N4x*uy + N4y*ux)))
            KNL[0+c4, 4+c4] += detJ*wij*(N4x*(B16*N4x*ux + B26*N4y*uy + B66*(N4x*uy + N4y*ux)) + N4y*(B12*N4x*ux + B22*N4y*uy + B26*(N4x*uy + N4y*ux)))
            KNL[1+c4, 0+c1] += detJ*wij*(N1x*ux*(A12*N4y + A16*N4x) + N1x*ux*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N1x*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N1y*uy*(A22*N4y + A26*N4x) + N1y*uy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + N1y*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N1x*uy + N1y*ux) + (N1x*uy + N1y*ux)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 1+c1] += detJ*wij*(N1x*vx*(A12*N4y + A16*N4x) + N1x*vx*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N1x*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)) + N1y*vy*(A22*N4y + A26*N4x) + N1y*vy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + N1y*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N1x*vy + N1y*vx) + (N1x*vy + N1y*vx)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 2+c1] += detJ*wij*(N1x*wx*(A12*N4y + A16*N4x) + N1x*wx*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N1y*wy*(A22*N4y + A26*N4x) + N1y*wy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N1x*wy + N1y*wx) + (N1x*wy + N1y*wx)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 3+c1] += detJ*wij*(N1x*(B11*N4x*vx + B12*N4y*vy + B16*(N4x*vy + N4y*vx)) + N1y*(B16*N4x*vx + B26*N4y*vy + B66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 4+c1] += detJ*wij*(N1x*(B16*N4x*vx + B26*N4y*vy + B66*(N4x*vy + N4y*vx)) + N1y*(B12*N4x*vx + B22*N4y*vy + B26*(N4x*vy + N4y*vx)))
            KNL[1+c4, 0+c2] += detJ*wij*(N2x*ux*(A12*N4y + A16*N4x) + N2x*ux*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N2x*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N2y*uy*(A22*N4y + A26*N4x) + N2y*uy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + N2y*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N2x*uy + N2y*ux) + (N2x*uy + N2y*ux)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 1+c2] += detJ*wij*(N2x*vx*(A12*N4y + A16*N4x) + N2x*vx*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N2x*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)) + N2y*vy*(A22*N4y + A26*N4x) + N2y*vy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + N2y*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N2x*vy + N2y*vx) + (N2x*vy + N2y*vx)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 2+c2] += detJ*wij*(N2x*wx*(A12*N4y + A16*N4x) + N2x*wx*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N2y*wy*(A22*N4y + A26*N4x) + N2y*wy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N2x*wy + N2y*wx) + (N2x*wy + N2y*wx)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 3+c2] += detJ*wij*(N2x*(B11*N4x*vx + B12*N4y*vy + B16*(N4x*vy + N4y*vx)) + N2y*(B16*N4x*vx + B26*N4y*vy + B66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 4+c2] += detJ*wij*(N2x*(B16*N4x*vx + B26*N4y*vy + B66*(N4x*vy + N4y*vx)) + N2y*(B12*N4x*vx + B22*N4y*vy + B26*(N4x*vy + N4y*vx)))
            KNL[1+c4, 0+c3] += detJ*wij*(N3x*ux*(A12*N4y + A16*N4x) + N3x*ux*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N3x*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N3y*uy*(A22*N4y + A26*N4x) + N3y*uy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + N3y*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N3x*uy + N3y*ux) + (N3x*uy + N3y*ux)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 1+c3] += detJ*wij*(N3x*vx*(A12*N4y + A16*N4x) + N3x*vx*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N3x*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)) + N3y*vy*(A22*N4y + A26*N4x) + N3y*vy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + N3y*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N3x*vy + N3y*vx) + (N3x*vy + N3y*vx)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 2+c3] += detJ*wij*(N3x*wx*(A12*N4y + A16*N4x) + N3x*wx*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N3y*wy*(A22*N4y + A26*N4x) + N3y*wy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N3x*wy + N3y*wx) + (N3x*wy + N3y*wx)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 3+c3] += detJ*wij*(N3x*(B11*N4x*vx + B12*N4y*vy + B16*(N4x*vy + N4y*vx)) + N3y*(B16*N4x*vx + B26*N4y*vy + B66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 4+c3] += detJ*wij*(N3x*(B16*N4x*vx + B26*N4y*vy + B66*(N4x*vy + N4y*vx)) + N3y*(B12*N4x*vx + B22*N4y*vy + B26*(N4x*vy + N4y*vx)))
            KNL[1+c4, 0+c4] += detJ*wij*(N4x*ux*(A12*N4y + A16*N4x) + N4x*ux*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N4x*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N4y*uy*(A22*N4y + A26*N4x) + N4y*uy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + N4y*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N4x*uy + N4y*ux) + (N4x*uy + N4y*ux)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 1+c4] += detJ*wij*(N4x*vx*(A12*N4y + A16*N4x) + N4x*vx*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N4x*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)) + N4y*vy*(A22*N4y + A26*N4x) + N4y*vy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + N4y*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N4x*vy + N4y*vx) + (N4x*vy + N4y*vx)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 2+c4] += detJ*wij*(N4x*wx*(A12*N4y + A16*N4x) + N4x*wx*(A11*N4x*vx + A12*N4y*vy + A16*(N4x*vy + N4y*vx)) + N4y*wy*(A22*N4y + A26*N4x) + N4y*wy*(A12*N4x*vx + A22*N4y*vy + A26*(N4x*vy + N4y*vx)) + (A26*N4y + A66*N4x)*(N4x*wy + N4y*wx) + (N4x*wy + N4y*wx)*(A16*N4x*vx + A26*N4y*vy + A66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 3+c4] += detJ*wij*(N4x*(B11*N4x*vx + B12*N4y*vy + B16*(N4x*vy + N4y*vx)) + N4y*(B16*N4x*vx + B26*N4y*vy + B66*(N4x*vy + N4y*vx)))
            KNL[1+c4, 4+c4] += detJ*wij*(N4x*(B16*N4x*vx + B26*N4y*vy + B66*(N4x*vy + N4y*vx)) + N4y*(B12*N4x*vx + B22*N4y*vy + B26*(N4x*vy + N4y*vx)))
            KNL[2+c4, 0+c1] += detJ*wij*(N1x*ux*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N1x*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N1y*uy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + N1y*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)) + (N1x*uy + N1y*ux)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 1+c1] += detJ*wij*(N1x*vx*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N1x*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)) + N1y*vy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + N1y*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + (N1x*vy + N1y*vx)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 2+c1] += detJ*wij*(N1x*wx*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N1y*wy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + (N1x*wy + N1y*wx)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 3+c1] += detJ*wij*(N1x*(B11*N4x*wx + B12*N4y*wy + B16*(N4x*wy + N4y*wx)) + N1y*(B16*N4x*wx + B26*N4y*wy + B66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 4+c1] += detJ*wij*(N1x*(B16*N4x*wx + B26*N4y*wy + B66*(N4x*wy + N4y*wx)) + N1y*(B12*N4x*wx + B22*N4y*wy + B26*(N4x*wy + N4y*wx)))
            KNL[2+c4, 0+c2] += detJ*wij*(N2x*ux*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N2x*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N2y*uy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + N2y*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)) + (N2x*uy + N2y*ux)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 1+c2] += detJ*wij*(N2x*vx*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N2x*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)) + N2y*vy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + N2y*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + (N2x*vy + N2y*vx)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 2+c2] += detJ*wij*(N2x*wx*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N2y*wy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + (N2x*wy + N2y*wx)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 3+c2] += detJ*wij*(N2x*(B11*N4x*wx + B12*N4y*wy + B16*(N4x*wy + N4y*wx)) + N2y*(B16*N4x*wx + B26*N4y*wy + B66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 4+c2] += detJ*wij*(N2x*(B16*N4x*wx + B26*N4y*wy + B66*(N4x*wy + N4y*wx)) + N2y*(B12*N4x*wx + B22*N4y*wy + B26*(N4x*wy + N4y*wx)))
            KNL[2+c4, 0+c3] += detJ*wij*(N3x*ux*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N3x*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N3y*uy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + N3y*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)) + (N3x*uy + N3y*ux)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 1+c3] += detJ*wij*(N3x*vx*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N3x*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)) + N3y*vy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + N3y*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + (N3x*vy + N3y*vx)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 2+c3] += detJ*wij*(N3x*wx*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N3y*wy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + (N3x*wy + N3y*wx)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 3+c3] += detJ*wij*(N3x*(B11*N4x*wx + B12*N4y*wy + B16*(N4x*wy + N4y*wx)) + N3y*(B16*N4x*wx + B26*N4y*wy + B66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 4+c3] += detJ*wij*(N3x*(B16*N4x*wx + B26*N4y*wy + B66*(N4x*wy + N4y*wx)) + N3y*(B12*N4x*wx + B22*N4y*wy + B26*(N4x*wy + N4y*wx)))
            KNL[2+c4, 0+c4] += detJ*wij*(N4x*ux*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N4x*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N4y*uy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + N4y*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)) + (N4x*uy + N4y*ux)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 1+c4] += detJ*wij*(N4x*vx*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N4x*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)) + N4y*vy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + N4y*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + (N4x*vy + N4y*vx)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 2+c4] += detJ*wij*(N4x*wx*(A11*N4x*wx + A12*N4y*wy + A16*(N4x*wy + N4y*wx)) + N4y*wy*(A12*N4x*wx + A22*N4y*wy + A26*(N4x*wy + N4y*wx)) + (N4x*wy + N4y*wx)*(A16*N4x*wx + A26*N4y*wy + A66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 3+c4] += detJ*wij*(N4x*(B11*N4x*wx + B12*N4y*wy + B16*(N4x*wy + N4y*wx)) + N4y*(B16*N4x*wx + B26*N4y*wy + B66*(N4x*wy + N4y*wx)))
            KNL[2+c4, 4+c4] += detJ*wij*(N4x*(B16*N4x*wx + B26*N4y*wy + B66*(N4x*wy + N4y*wx)) + N4y*(B12*N4x*wx + B22*N4y*wy + B26*(N4x*wy + N4y*wx)))
            KNL[3+c4, 0+c1] += detJ*wij*(N1x*ux*(B11*N4x + B16*N4y) + N1y*uy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N1x*uy + N1y*ux))
            KNL[3+c4, 1+c1] += detJ*wij*(N1x*vx*(B11*N4x + B16*N4y) + N1y*vy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N1x*vy + N1y*vx))
            KNL[3+c4, 2+c1] += detJ*wij*(N1x*wx*(B11*N4x + B16*N4y) + N1y*wy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N1x*wy + N1y*wx))
            KNL[3+c4, 0+c2] += detJ*wij*(N2x*ux*(B11*N4x + B16*N4y) + N2y*uy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N2x*uy + N2y*ux))
            KNL[3+c4, 1+c2] += detJ*wij*(N2x*vx*(B11*N4x + B16*N4y) + N2y*vy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N2x*vy + N2y*vx))
            KNL[3+c4, 2+c2] += detJ*wij*(N2x*wx*(B11*N4x + B16*N4y) + N2y*wy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N2x*wy + N2y*wx))
            KNL[3+c4, 0+c3] += detJ*wij*(N3x*ux*(B11*N4x + B16*N4y) + N3y*uy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N3x*uy + N3y*ux))
            KNL[3+c4, 1+c3] += detJ*wij*(N3x*vx*(B11*N4x + B16*N4y) + N3y*vy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N3x*vy + N3y*vx))
            KNL[3+c4, 2+c3] += detJ*wij*(N3x*wx*(B11*N4x + B16*N4y) + N3y*wy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N3x*wy + N3y*wx))
            KNL[3+c4, 0+c4] += detJ*wij*(N4x*ux*(B11*N4x + B16*N4y) + N4y*uy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N4x*uy + N4y*ux))
            KNL[3+c4, 1+c4] += detJ*wij*(N4x*vx*(B11*N4x + B16*N4y) + N4y*vy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N4x*vy + N4y*vx))
            KNL[3+c4, 2+c4] += detJ*wij*(N4x*wx*(B11*N4x + B16*N4y) + N4y*wy*(B12*N4x + B26*N4y) + (B16*N4x + B66*N4y)*(N4x*wy + N4y*wx))
            KNL[4+c4, 0+c1] += detJ*wij*(N1x*ux*(B12*N4y + B16*N4x) + N1y*uy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N1x*uy + N1y*ux))
            KNL[4+c4, 1+c1] += detJ*wij*(N1x*vx*(B12*N4y + B16*N4x) + N1y*vy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N1x*vy + N1y*vx))
            KNL[4+c4, 2+c1] += detJ*wij*(N1x*wx*(B12*N4y + B16*N4x) + N1y*wy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N1x*wy + N1y*wx))
            KNL[4+c4, 0+c2] += detJ*wij*(N2x*ux*(B12*N4y + B16*N4x) + N2y*uy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N2x*uy + N2y*ux))
            KNL[4+c4, 1+c2] += detJ*wij*(N2x*vx*(B12*N4y + B16*N4x) + N2y*vy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N2x*vy + N2y*vx))
            KNL[4+c4, 2+c2] += detJ*wij*(N2x*wx*(B12*N4y + B16*N4x) + N2y*wy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N2x*wy + N2y*wx))
            KNL[4+c4, 0+c3] += detJ*wij*(N3x*ux*(B12*N4y + B16*N4x) + N3y*uy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N3x*uy + N3y*ux))
            KNL[4+c4, 1+c3] += detJ*wij*(N3x*vx*(B12*N4y + B16*N4x) + N3y*vy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N3x*vy + N3y*vx))
            KNL[4+c4, 2+c3] += detJ*wij*(N3x*wx*(B12*N4y + B16*N4x) + N3y*wy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N3x*wy + N3y*wx))
            KNL[4+c4, 0+c4] += detJ*wij*(N4x*ux*(B12*N4y + B16*N4x) + N4y*uy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N4x*uy + N4y*ux))
            KNL[4+c4, 1+c4] += detJ*wij*(N4x*vx*(B12*N4y + B16*N4x) + N4y*vy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N4x*vy + N4y*vx))
            KNL[4+c4, 2+c4] += detJ*wij*(N4x*wx*(B12*N4y + B16*N4x) + N4y*wy*(B22*N4y + B26*N4x) + (B26*N4y + B66*N4x)*(N4x*wy + N4y*wx))


def calc_fint(quads, u0, nid_pos, ncoords):
    """Calculate the internal force vector

    Properties
    ----------
    quads : list of `.Quad4R`objects
        The quad elements to be added to the internal force vector
    u0: array-like
        A displacement state ``u0`` in global coordinates
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model

    """
    fint = np.zeros_like(u0)

    for quad in quads:
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

        # positions c1, c2 in the stiffness and mass matrices
        c1 = DOF*pos1
        c2 = DOF*pos2
        c3 = DOF*pos3
        c4 = DOF*pos4

        u = np.concatenate((u0[c1:c1+DOF], u0[c2:c2+DOF], u0[c3:c3+DOF], u0[c4:c4+DOF]))

        #NOTE full 2-point Gauss-Legendre quadrature integration for KNL
        weights_points =[[1., -0.577350269189625764509148780501957455647601751270126876],
                         [1., +0.577350269189625764509148780501957455647601751270126876]]

        #NOTE reduced integration with 1 point at center
        #NOTE this seems to work the same as using the full integration
        weights_points =[[2., 0]]

        for wi, xi in weights_points:
            for wj, eta in weights_points:
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
                ux = N1x*u[0] + N2x*u[5] + N3x*u[10] + N4x*u[15]
                uy = N1y*u[0] + N2y*u[5] + N3y*u[10] + N4y*u[15]
                vx = N1x*u[1] + N2x*u[6] + N3x*u[11] + N4x*u[16]
                vy = N1y*u[1] + N2y*u[6] + N3y*u[11] + N4y*u[16]
                wx = N1x*u[2] + N2x*u[7] + N3x*u[12] + N4x*u[17]
                wy = N1y*u[2] + N2y*u[7] + N3y*u[12] + N4y*u[17]
                Nxx = u[0]*(A11*(N1x*ux/2 + N1x) + A12*N1y*uy/2 + A16*(N1x*uy/2 + N1y*ux/2 + N1y)) + u[10]*(A11*(N3x*ux/2 + N3x) + A12*N3y*uy/2 + A16*(N3x*uy/2 + N3y*ux/2 + N3y)) + u[11]*(A11*N3x*vx/2 + A12*(N3y*vy/2 + N3y) + A16*(N3x*vy/2 + N3x + N3y*vx/2)) + u[12]*(A11*N3x*wx/2 + A12*N3y*wy/2 + A16*(N3x*wy/2 + N3y*wx/2)) + u[13]*(B11*N3x + B16*N3y) + u[14]*(B12*N3y + B16*N3x) + u[15]*(A11*(N4x*ux/2 + N4x) + A12*N4y*uy/2 + A16*(N4x*uy/2 + N4y*ux/2 + N4y)) + u[16]*(A11*N4x*vx/2 + A12*(N4y*vy/2 + N4y) + A16*(N4x*vy/2 + N4x + N4y*vx/2)) + u[17]*(A11*N4x*wx/2 + A12*N4y*wy/2 + A16*(N4x*wy/2 + N4y*wx/2)) + u[18]*(B11*N4x + B16*N4y) + u[19]*(B12*N4y + B16*N4x) + u[1]*(A11*N1x*vx/2 + A12*(N1y*vy/2 + N1y) + A16*(N1x*vy/2 + N1x + N1y*vx/2)) + u[2]*(A11*N1x*wx/2 + A12*N1y*wy/2 + A16*(N1x*wy/2 + N1y*wx/2)) + u[3]*(B11*N1x + B16*N1y) + u[4]*(B12*N1y + B16*N1x) + u[5]*(A11*(N2x*ux/2 + N2x) + A12*N2y*uy/2 + A16*(N2x*uy/2 + N2y*ux/2 + N2y)) + u[6]*(A11*N2x*vx/2 + A12*(N2y*vy/2 + N2y) + A16*(N2x*vy/2 + N2x + N2y*vx/2)) + u[7]*(A11*N2x*wx/2 + A12*N2y*wy/2 + A16*(N2x*wy/2 + N2y*wx/2)) + u[8]*(B11*N2x + B16*N2y) + u[9]*(B12*N2y + B16*N2x)
                Nyy = u[0]*(A12*(N1x*ux/2 + N1x) + A22*N1y*uy/2 + A26*(N1x*uy/2 + N1y*ux/2 + N1y)) + u[10]*(A12*(N3x*ux/2 + N3x) + A22*N3y*uy/2 + A26*(N3x*uy/2 + N3y*ux/2 + N3y)) + u[11]*(A12*N3x*vx/2 + A22*(N3y*vy/2 + N3y) + A26*(N3x*vy/2 + N3x + N3y*vx/2)) + u[12]*(A12*N3x*wx/2 + A22*N3y*wy/2 + A26*(N3x*wy/2 + N3y*wx/2)) + u[13]*(B12*N3x + B26*N3y) + u[14]*(B22*N3y + B26*N3x) + u[15]*(A12*(N4x*ux/2 + N4x) + A22*N4y*uy/2 + A26*(N4x*uy/2 + N4y*ux/2 + N4y)) + u[16]*(A12*N4x*vx/2 + A22*(N4y*vy/2 + N4y) + A26*(N4x*vy/2 + N4x + N4y*vx/2)) + u[17]*(A12*N4x*wx/2 + A22*N4y*wy/2 + A26*(N4x*wy/2 + N4y*wx/2)) + u[18]*(B12*N4x + B26*N4y) + u[19]*(B22*N4y + B26*N4x) + u[1]*(A12*N1x*vx/2 + A22*(N1y*vy/2 + N1y) + A26*(N1x*vy/2 + N1x + N1y*vx/2)) + u[2]*(A12*N1x*wx/2 + A22*N1y*wy/2 + A26*(N1x*wy/2 + N1y*wx/2)) + u[3]*(B12*N1x + B26*N1y) + u[4]*(B22*N1y + B26*N1x) + u[5]*(A12*(N2x*ux/2 + N2x) + A22*N2y*uy/2 + A26*(N2x*uy/2 + N2y*ux/2 + N2y)) + u[6]*(A12*N2x*vx/2 + A22*(N2y*vy/2 + N2y) + A26*(N2x*vy/2 + N2x + N2y*vx/2)) + u[7]*(A12*N2x*wx/2 + A22*N2y*wy/2 + A26*(N2x*wy/2 + N2y*wx/2)) + u[8]*(B12*N2x + B26*N2y) + u[9]*(B22*N2y + B26*N2x)
                Nxy = u[0]*(A16*(N1x*ux/2 + N1x) + A26*N1y*uy/2 + A66*(N1x*uy/2 + N1y*ux/2 + N1y)) + u[10]*(A16*(N3x*ux/2 + N3x) + A26*N3y*uy/2 + A66*(N3x*uy/2 + N3y*ux/2 + N3y)) + u[11]*(A16*N3x*vx/2 + A26*(N3y*vy/2 + N3y) + A66*(N3x*vy/2 + N3x + N3y*vx/2)) + u[12]*(A16*N3x*wx/2 + A26*N3y*wy/2 + A66*(N3x*wy/2 + N3y*wx/2)) + u[13]*(B16*N3x + B66*N3y) + u[14]*(B26*N3y + B66*N3x) + u[15]*(A16*(N4x*ux/2 + N4x) + A26*N4y*uy/2 + A66*(N4x*uy/2 + N4y*ux/2 + N4y)) + u[16]*(A16*N4x*vx/2 + A26*(N4y*vy/2 + N4y) + A66*(N4x*vy/2 + N4x + N4y*vx/2)) + u[17]*(A16*N4x*wx/2 + A26*N4y*wy/2 + A66*(N4x*wy/2 + N4y*wx/2)) + u[18]*(B16*N4x + B66*N4y) + u[19]*(B26*N4y + B66*N4x) + u[1]*(A16*N1x*vx/2 + A26*(N1y*vy/2 + N1y) + A66*(N1x*vy/2 + N1x + N1y*vx/2)) + u[2]*(A16*N1x*wx/2 + A26*N1y*wy/2 + A66*(N1x*wy/2 + N1y*wx/2)) + u[3]*(B16*N1x + B66*N1y) + u[4]*(B26*N1y + B66*N1x) + u[5]*(A16*(N2x*ux/2 + N2x) + A26*N2y*uy/2 + A66*(N2x*uy/2 + N2y*ux/2 + N2y)) + u[6]*(A16*N2x*vx/2 + A26*(N2y*vy/2 + N2y) + A66*(N2x*vy/2 + N2x + N2y*vx/2)) + u[7]*(A16*N2x*wx/2 + A26*N2y*wy/2 + A66*(N2x*wy/2 + N2y*wx/2)) + u[8]*(B16*N2x + B66*N2y) + u[9]*(B26*N2y + B66*N2x)
                Mxx = u[0]*(B11*(N1x*ux/2 + N1x) + B12*N1y*uy/2 + B16*(N1x*uy/2 + N1y*ux/2 + N1y)) + u[10]*(B11*(N3x*ux/2 + N3x) + B12*N3y*uy/2 + B16*(N3x*uy/2 + N3y*ux/2 + N3y)) + u[11]*(B11*N3x*vx/2 + B12*(N3y*vy/2 + N3y) + B16*(N3x*vy/2 + N3x + N3y*vx/2)) + u[12]*(B11*N3x*wx/2 + B12*N3y*wy/2 + B16*(N3x*wy/2 + N3y*wx/2)) + u[13]*(D11*N3x + D16*N3y) + u[14]*(D12*N3y + D16*N3x) + u[15]*(B11*(N4x*ux/2 + N4x) + B12*N4y*uy/2 + B16*(N4x*uy/2 + N4y*ux/2 + N4y)) + u[16]*(B11*N4x*vx/2 + B12*(N4y*vy/2 + N4y) + B16*(N4x*vy/2 + N4x + N4y*vx/2)) + u[17]*(B11*N4x*wx/2 + B12*N4y*wy/2 + B16*(N4x*wy/2 + N4y*wx/2)) + u[18]*(D11*N4x + D16*N4y) + u[19]*(D12*N4y + D16*N4x) + u[1]*(B11*N1x*vx/2 + B12*(N1y*vy/2 + N1y) + B16*(N1x*vy/2 + N1x + N1y*vx/2)) + u[2]*(B11*N1x*wx/2 + B12*N1y*wy/2 + B16*(N1x*wy/2 + N1y*wx/2)) + u[3]*(D11*N1x + D16*N1y) + u[4]*(D12*N1y + D16*N1x) + u[5]*(B11*(N2x*ux/2 + N2x) + B12*N2y*uy/2 + B16*(N2x*uy/2 + N2y*ux/2 + N2y)) + u[6]*(B11*N2x*vx/2 + B12*(N2y*vy/2 + N2y) + B16*(N2x*vy/2 + N2x + N2y*vx/2)) + u[7]*(B11*N2x*wx/2 + B12*N2y*wy/2 + B16*(N2x*wy/2 + N2y*wx/2)) + u[8]*(D11*N2x + D16*N2y) + u[9]*(D12*N2y + D16*N2x)
                Myy = u[0]*(B12*(N1x*ux/2 + N1x) + B22*N1y*uy/2 + B26*(N1x*uy/2 + N1y*ux/2 + N1y)) + u[10]*(B12*(N3x*ux/2 + N3x) + B22*N3y*uy/2 + B26*(N3x*uy/2 + N3y*ux/2 + N3y)) + u[11]*(B12*N3x*vx/2 + B22*(N3y*vy/2 + N3y) + B26*(N3x*vy/2 + N3x + N3y*vx/2)) + u[12]*(B12*N3x*wx/2 + B22*N3y*wy/2 + B26*(N3x*wy/2 + N3y*wx/2)) + u[13]*(D12*N3x + D26*N3y) + u[14]*(D22*N3y + D26*N3x) + u[15]*(B12*(N4x*ux/2 + N4x) + B22*N4y*uy/2 + B26*(N4x*uy/2 + N4y*ux/2 + N4y)) + u[16]*(B12*N4x*vx/2 + B22*(N4y*vy/2 + N4y) + B26*(N4x*vy/2 + N4x + N4y*vx/2)) + u[17]*(B12*N4x*wx/2 + B22*N4y*wy/2 + B26*(N4x*wy/2 + N4y*wx/2)) + u[18]*(D12*N4x + D26*N4y) + u[19]*(D22*N4y + D26*N4x) + u[1]*(B12*N1x*vx/2 + B22*(N1y*vy/2 + N1y) + B26*(N1x*vy/2 + N1x + N1y*vx/2)) + u[2]*(B12*N1x*wx/2 + B22*N1y*wy/2 + B26*(N1x*wy/2 + N1y*wx/2)) + u[3]*(D12*N1x + D26*N1y) + u[4]*(D22*N1y + D26*N1x) + u[5]*(B12*(N2x*ux/2 + N2x) + B22*N2y*uy/2 + B26*(N2x*uy/2 + N2y*ux/2 + N2y)) + u[6]*(B12*N2x*vx/2 + B22*(N2y*vy/2 + N2y) + B26*(N2x*vy/2 + N2x + N2y*vx/2)) + u[7]*(B12*N2x*wx/2 + B22*N2y*wy/2 + B26*(N2x*wy/2 + N2y*wx/2)) + u[8]*(D12*N2x + D26*N2y) + u[9]*(D22*N2y + D26*N2x)
                Mxy = u[0]*(B16*(N1x*ux/2 + N1x) + B26*N1y*uy/2 + B66*(N1x*uy/2 + N1y*ux/2 + N1y)) + u[10]*(B16*(N3x*ux/2 + N3x) + B26*N3y*uy/2 + B66*(N3x*uy/2 + N3y*ux/2 + N3y)) + u[11]*(B16*N3x*vx/2 + B26*(N3y*vy/2 + N3y) + B66*(N3x*vy/2 + N3x + N3y*vx/2)) + u[12]*(B16*N3x*wx/2 + B26*N3y*wy/2 + B66*(N3x*wy/2 + N3y*wx/2)) + u[13]*(D16*N3x + D66*N3y) + u[14]*(D26*N3y + D66*N3x) + u[15]*(B16*(N4x*ux/2 + N4x) + B26*N4y*uy/2 + B66*(N4x*uy/2 + N4y*ux/2 + N4y)) + u[16]*(B16*N4x*vx/2 + B26*(N4y*vy/2 + N4y) + B66*(N4x*vy/2 + N4x + N4y*vx/2)) + u[17]*(B16*N4x*wx/2 + B26*N4y*wy/2 + B66*(N4x*wy/2 + N4y*wx/2)) + u[18]*(D16*N4x + D66*N4y) + u[19]*(D26*N4y + D66*N4x) + u[1]*(B16*N1x*vx/2 + B26*(N1y*vy/2 + N1y) + B66*(N1x*vy/2 + N1x + N1y*vx/2)) + u[2]*(B16*N1x*wx/2 + B26*N1y*wy/2 + B66*(N1x*wy/2 + N1y*wx/2)) + u[3]*(D16*N1x + D66*N1y) + u[4]*(D26*N1y + D66*N1x) + u[5]*(B16*(N2x*ux/2 + N2x) + B26*N2y*uy/2 + B66*(N2x*uy/2 + N2y*ux/2 + N2y)) + u[6]*(B16*N2x*vx/2 + B26*(N2y*vy/2 + N2y) + B66*(N2x*vy/2 + N2x + N2y*vx/2)) + u[7]*(B16*N2x*wx/2 + B26*N2y*wy/2 + B66*(N2x*wy/2 + N2y*wx/2)) + u[8]*(D16*N2x + D66*N2y) + u[9]*(D26*N2y + D66*N2x)
                Qx = E44*N1*u[4] + E44*N2*u[9] + E44*N3*u[14] + E44*N4*u[19] + E45*N1*u[3] + E45*N2*u[8] + E45*N3*u[13] + E45*N4*u[18] + u[12]*(E44*N3y + E45*N3x) + u[17]*(E44*N4y + E45*N4x) + u[2]*(E44*N1y + E45*N1x) + u[7]*(E44*N2y + E45*N2x)
                Qy = E45*N1*u[4] + E45*N2*u[9] + E45*N3*u[14] + E45*N4*u[19] + E55*N1*u[3] + E55*N2*u[8] + E55*N3*u[13] + E55*N4*u[18] + u[12]*(E45*N3y + E55*N3x) + u[17]*(E45*N4y + E55*N4x) + u[2]*(E45*N1y + E55*N1x) + u[7]*(E45*N2y + E55*N2x)

                fint[0 + c1] += N1y*Nyy*detJ*uy*wij + Nxx*detJ*wij*(N1x*ux + N1x) + Nxy*detJ*wij*(N1x*uy + N1y*ux + N1y)
                fint[1 + c1] += N1x*Nxx*detJ*vx*wij + Nxy*detJ*wij*(N1x*vy + N1x + N1y*vx) + Nyy*detJ*wij*(N1y*vy + N1y)
                fint[2 + c1] += N1x*Nxx*detJ*wij*wx + N1x*Qy*detJ*wij + N1y*Nyy*detJ*wij*wy + N1y*Qx*detJ*wij + Nxy*detJ*wij*(N1x*wy + N1y*wx)
                fint[3 + c1] += Mxx*N1x*detJ*wij + Mxy*N1y*detJ*wij + N1*Qy*detJ*wij
                fint[4 + c1] += Mxy*N1x*detJ*wij + Myy*N1y*detJ*wij + N1*Qx*detJ*wij
                fint[0 + c2] += N2y*Nyy*detJ*uy*wij + Nxx*detJ*wij*(N2x*ux + N2x) + Nxy*detJ*wij*(N2x*uy + N2y*ux + N2y)
                fint[1 + c2] += N2x*Nxx*detJ*vx*wij + Nxy*detJ*wij*(N2x*vy + N2x + N2y*vx) + Nyy*detJ*wij*(N2y*vy + N2y)
                fint[2 + c2] += N2x*Nxx*detJ*wij*wx + N2x*Qy*detJ*wij + N2y*Nyy*detJ*wij*wy + N2y*Qx*detJ*wij + Nxy*detJ*wij*(N2x*wy + N2y*wx)
                fint[3 + c2] += Mxx*N2x*detJ*wij + Mxy*N2y*detJ*wij + N2*Qy*detJ*wij
                fint[4 + c2] += Mxy*N2x*detJ*wij + Myy*N2y*detJ*wij + N2*Qx*detJ*wij
                fint[0 + c3] += N3y*Nyy*detJ*uy*wij + Nxx*detJ*wij*(N3x*ux + N3x) + Nxy*detJ*wij*(N3x*uy + N3y*ux + N3y)
                fint[1 + c3] += N3x*Nxx*detJ*vx*wij + Nxy*detJ*wij*(N3x*vy + N3x + N3y*vx) + Nyy*detJ*wij*(N3y*vy + N3y)
                fint[2 + c3] += N3x*Nxx*detJ*wij*wx + N3x*Qy*detJ*wij + N3y*Nyy*detJ*wij*wy + N3y*Qx*detJ*wij + Nxy*detJ*wij*(N3x*wy + N3y*wx)
                fint[3 + c3] += Mxx*N3x*detJ*wij + Mxy*N3y*detJ*wij + N3*Qy*detJ*wij
                fint[4 + c3] += Mxy*N3x*detJ*wij + Myy*N3y*detJ*wij + N3*Qx*detJ*wij
                fint[0 + c4] += N4y*Nyy*detJ*uy*wij + Nxx*detJ*wij*(N4x*ux + N4x) + Nxy*detJ*wij*(N4x*uy + N4y*ux + N4y)
                fint[1 + c4] += N4x*Nxx*detJ*vx*wij + Nxy*detJ*wij*(N4x*vy + N4x + N4y*vx) + Nyy*detJ*wij*(N4y*vy + N4y)
                fint[2 + c4] += N4x*Nxx*detJ*wij*wx + N4x*Qy*detJ*wij + N4y*Nyy*detJ*wij*wy + N4y*Qx*detJ*wij + Nxy*detJ*wij*(N4x*wy + N4y*wx)
                fint[3 + c4] += Mxx*N4x*detJ*wij + Mxy*N4y*detJ*wij + N4*Qy*detJ*wij
                fint[4 + c4] += Mxy*N4x*detJ*wij + Myy*N4y*detJ*wij + N4*Qx*detJ*wij
    return fint


def update_M(quad, nid_pos, ncoords, M, lumped=False):
    """Update global M with Me from a quad element

    Properties
    ----------
    quad : `.Quad4R` object
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
    quad : `.Quad4R`object
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
