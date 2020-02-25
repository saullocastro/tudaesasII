import numpy as np

DOF = 2

class Tria3PlaneStrainIso(object):
    __slots__ = ['n1', 'n2', 'n3', 'E', 'nu', 'A', 'h', 'rho']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        self.n3 = None
        # Material Lastrobe Lescalloy
        self.E = None
        self.nu = None
        self.rho = None

def update_K_M(tria, nid_pos, ncoords, K, M, lumped=False):
    """Update a global stiffness matrix K and mass matrix M

    Properties
    ----------
    tria : `.Tria3PlaneStrainIso` object
        The Tria3PlaneStrainIso element being added to K
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    K : np.array
        Global stiffness matrix updated in-place
    M : np.array
        Global mass matrix updated in-place (affected by parameter `lumped`)
    lumped : bool
        If lumped mass matrix should be used

    """
    pos1 = nid_pos[tria.n1]
    pos2 = nid_pos[tria.n2]
    pos3 = nid_pos[tria.n3]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    x3, y3 = ncoords[pos3]
    A = abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))/2)

    N1x = (y2 - y3)/(2*A)
    N2x = (-y1 + y3)/(2*A)
    N3x = (y1 - y2)/(2*A)
    N1y = (-x2 + x3)/(2*A)
    N2y = (x1 - x3)/(2*A)
    N3y = (-x1 + x2)/(2*A)

    tria.A = A
    E = tria.E
    nu = tria.nu
    h = tria.h
    rho = tria.rho

    # positions the global matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3

    K[0+c1, 0+c1] += A*E*h*(N1x**2*nu - N1x**2 + 2*N1y**2*nu - N1y**2)/(2*nu**2 + nu - 1)
    K[0+c1, 1+c1] += A*E*N1x*N1y*h*(nu - 1)/(2*nu**2 + nu - 1)
    K[0+c1, 0+c2] += A*E*h*(N1x*N2x*nu - N1x*N2x + 2*N1y*N2y*nu - N1y*N2y)/(2*nu**2 + nu - 1)
    K[0+c1, 1+c2] += A*E*h*(-N1x*N2y*nu + 2*N1y*N2x*nu - N1y*N2x)/(2*nu**2 + nu - 1)
    K[0+c1, 0+c3] += A*E*h*(N1x*N3x*nu - N1x*N3x + 2*N1y*N3y*nu - N1y*N3y)/(2*nu**2 + nu - 1)
    K[0+c1, 1+c3] += A*E*h*(-N1x*N3y*nu + 2*N1y*N3x*nu - N1y*N3x)/(2*nu**2 + nu - 1)
    K[1+c1, 0+c1] += A*E*N1x*N1y*h*(nu - 1)/(2*nu**2 + nu - 1)
    K[1+c1, 1+c1] += A*E*h*(2*N1x**2*nu - N1x**2 + N1y**2*nu - N1y**2)/(2*nu**2 + nu - 1)
    K[1+c1, 0+c2] += A*E*h*(2*N1x*N2y*nu - N1x*N2y - N1y*N2x*nu)/(2*nu**2 + nu - 1)
    K[1+c1, 1+c2] += A*E*h*(2*N1x*N2x*nu - N1x*N2x + N1y*N2y*nu - N1y*N2y)/(2*nu**2 + nu - 1)
    K[1+c1, 0+c3] += A*E*h*(2*N1x*N3y*nu - N1x*N3y - N1y*N3x*nu)/(2*nu**2 + nu - 1)
    K[1+c1, 1+c3] += A*E*h*(2*N1x*N3x*nu - N1x*N3x + N1y*N3y*nu - N1y*N3y)/(2*nu**2 + nu - 1)
    K[0+c2, 0+c1] += A*E*h*(N1x*N2x*nu - N1x*N2x + 2*N1y*N2y*nu - N1y*N2y)/(2*nu**2 + nu - 1)
    K[0+c2, 1+c1] += A*E*h*(2*N1x*N2y*nu - N1x*N2y - N1y*N2x*nu)/(2*nu**2 + nu - 1)
    K[0+c2, 0+c2] += A*E*h*(N2x**2*nu - N2x**2 + 2*N2y**2*nu - N2y**2)/(2*nu**2 + nu - 1)
    K[0+c2, 1+c2] += A*E*N2x*N2y*h*(nu - 1)/(2*nu**2 + nu - 1)
    K[0+c2, 0+c3] += A*E*h*(N2x*N3x*nu - N2x*N3x + 2*N2y*N3y*nu - N2y*N3y)/(2*nu**2 + nu - 1)
    K[0+c2, 1+c3] += A*E*h*(-N2x*N3y*nu + 2*N2y*N3x*nu - N2y*N3x)/(2*nu**2 + nu - 1)
    K[1+c2, 0+c1] += A*E*h*(-N1x*N2y*nu + 2*N1y*N2x*nu - N1y*N2x)/(2*nu**2 + nu - 1)
    K[1+c2, 1+c1] += A*E*h*(2*N1x*N2x*nu - N1x*N2x + N1y*N2y*nu - N1y*N2y)/(2*nu**2 + nu - 1)
    K[1+c2, 0+c2] += A*E*N2x*N2y*h*(nu - 1)/(2*nu**2 + nu - 1)
    K[1+c2, 1+c2] += A*E*h*(2*N2x**2*nu - N2x**2 + N2y**2*nu - N2y**2)/(2*nu**2 + nu - 1)
    K[1+c2, 0+c3] += A*E*h*(2*N2x*N3y*nu - N2x*N3y - N2y*N3x*nu)/(2*nu**2 + nu - 1)
    K[1+c2, 1+c3] += A*E*h*(2*N2x*N3x*nu - N2x*N3x + N2y*N3y*nu - N2y*N3y)/(2*nu**2 + nu - 1)
    K[0+c3, 0+c1] += A*E*h*(N1x*N3x*nu - N1x*N3x + 2*N1y*N3y*nu - N1y*N3y)/(2*nu**2 + nu - 1)
    K[0+c3, 1+c1] += A*E*h*(2*N1x*N3y*nu - N1x*N3y - N1y*N3x*nu)/(2*nu**2 + nu - 1)
    K[0+c3, 0+c2] += A*E*h*(N2x*N3x*nu - N2x*N3x + 2*N2y*N3y*nu - N2y*N3y)/(2*nu**2 + nu - 1)
    K[0+c3, 1+c2] += A*E*h*(2*N2x*N3y*nu - N2x*N3y - N2y*N3x*nu)/(2*nu**2 + nu - 1)
    K[0+c3, 0+c3] += A*E*h*(N3x**2*nu - N3x**2 + 2*N3y**2*nu - N3y**2)/(2*nu**2 + nu - 1)
    K[0+c3, 1+c3] += A*E*N3x*N3y*h*(nu - 1)/(2*nu**2 + nu - 1)
    K[1+c3, 0+c1] += A*E*h*(-N1x*N3y*nu + 2*N1y*N3x*nu - N1y*N3x)/(2*nu**2 + nu - 1)
    K[1+c3, 1+c1] += A*E*h*(2*N1x*N3x*nu - N1x*N3x + N1y*N3y*nu - N1y*N3y)/(2*nu**2 + nu - 1)
    K[1+c3, 0+c2] += A*E*h*(-N2x*N3y*nu + 2*N2y*N3x*nu - N2y*N3x)/(2*nu**2 + nu - 1)
    K[1+c3, 1+c2] += A*E*h*(2*N2x*N3x*nu - N2x*N3x + N2y*N3y*nu - N2y*N3y)/(2*nu**2 + nu - 1)
    K[1+c3, 0+c3] += A*E*N3x*N3y*h*(nu - 1)/(2*nu**2 + nu - 1)
    K[1+c3, 1+c3] += A*E*h*(2*N3x**2*nu - N3x**2 + N3y**2*nu - N3y**2)/(2*nu**2 + nu - 1)

    if lumped:
        M[0+c1, 0+c1] += A*h*rho/3
        M[1+c1, 1+c1] += A*h*rho/3
        M[0+c2, 0+c2] += A*h*rho/3
        M[1+c2, 1+c2] += A*h*rho/3
        M[0+c3, 0+c3] += A*h*rho/3
        M[1+c3, 1+c3] += A*h*rho/3
    else:
        M[0+c1, 0+c1] += A*h*rho/6
        M[0+c1, 0+c2] += A*h*rho/12
        M[0+c1, 0+c3] += A*h*rho/12
        M[1+c1, 1+c1] += A*h*rho/6
        M[1+c1, 1+c2] += A*h*rho/12
        M[1+c1, 1+c3] += A*h*rho/12
        M[0+c2, 0+c1] += A*h*rho/12
        M[0+c2, 0+c2] += A*h*rho/6
        M[0+c2, 0+c3] += A*h*rho/12
        M[1+c2, 1+c1] += A*h*rho/12
        M[1+c2, 1+c2] += A*h*rho/6
        M[1+c2, 1+c3] += A*h*rho/12
        M[0+c3, 0+c1] += A*h*rho/12
        M[0+c3, 0+c2] += A*h*rho/12
        M[0+c3, 0+c3] += A*h*rho/6
        M[1+c3, 1+c1] += A*h*rho/12
        M[1+c3, 1+c2] += A*h*rho/12
        M[1+c3, 1+c3] += A*h*rho/6
