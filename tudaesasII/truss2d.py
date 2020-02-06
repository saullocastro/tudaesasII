import numpy as np

#NOTE be careful when using the Beam2D with the Truss2D because currently the
#     Truss2D is derived with only 2 DOFs per node, while the Beam2D is defined
#     with 3 DOFs per node
DOF = 2

class Truss2D(object):
    __slots__ = ['n1', 'n2', 'E', 'A', 'le', 'rho', 'thetarad']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        # Material Lastrobe Lescalloy
        self.E = 203.e9 # Pa
        self.rho = 7.83e3 # kg/m3
        self.A = None
        self.le = None
        self.thetarad = None

def update_K_M(truss, nid_pos, ncoords, K, M, lumped=False):
    """Update a global stiffness matrix K and mass matrix M

    Properties
    ----------
    truss : `.Truss2D` object
        The Truss2D element being added to K
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    K : np.array
        Global stiffness matrix updated in-place
    M : np.array
        Global mass matrix updated in-place
    lumped : bool
        Whether to use the lumped mass matrix
    """
    pos1 = nid_pos[truss.n1]
    pos2 = nid_pos[truss.n2]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    theta = np.arctan2(y2-y1, x2-x1)
    truss.thetarad = theta
    le = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    c = np.cos(theta)
    s = np.sin(theta)
    A = truss.A
    E = truss.E
    truss.le = le
    rho = truss.rho

    # positions the global matrices
    c1 = DOF*pos1
    c2 = DOF*pos2

    K[0+c1, 0+c1] += A*E*c**2/le
    K[0+c1, 1+c1] += A*E*c*s/le
    K[0+c1, 0+c2] += -A*E*c**2/le
    K[0+c1, 1+c2] += -A*E*c*s/le
    K[1+c1, 0+c1] += A*E*c*s/le
    K[1+c1, 1+c1] += A*E*s**2/le
    K[1+c1, 0+c2] += -A*E*c*s/le
    K[1+c1, 1+c2] += -A*E*s**2/le
    K[0+c2, 0+c1] += -A*E*c**2/le
    K[0+c2, 1+c1] += -A*E*c*s/le
    K[0+c2, 0+c2] += A*E*c**2/le
    K[0+c2, 1+c2] += A*E*c*s/le
    K[1+c2, 0+c1] += -A*E*c*s/le
    K[1+c2, 1+c1] += -A*E*s**2/le
    K[1+c2, 0+c2] += A*E*c*s/le
    K[1+c2, 1+c2] += A*E*s**2/le

    if not lumped:
        M[0+c1, 0+c1] += A*le*c**2*rho/3 + A*le*rho*s**2/3
        M[0+c1, 0+c2] += A*le*c**2*rho/6 + A*le*rho*s**2/6
        M[1+c1, 1+c1] += A*le*c**2*rho/3 + A*le*rho*s**2/3
        M[1+c1, 1+c2] += A*le*c**2*rho/6 + A*le*rho*s**2/6
        M[0+c2, 0+c1] += A*le*c**2*rho/6 + A*le*rho*s**2/6
        M[0+c2, 0+c2] += A*le*c**2*rho/3 + A*le*rho*s**2/3
        M[1+c2, 1+c1] += A*le*c**2*rho/6 + A*le*rho*s**2/6
        M[1+c2, 1+c2] += A*le*c**2*rho/3 + A*le*rho*s**2/3

    if lumped:
        M[0+c1, 0+c1] += A*le*c**2*rho/2 + A*le*rho*s**2/2
        M[1+c1, 1+c1] += A*le*c**2*rho/2 + A*le*rho*s**2/2
        M[0+c2, 0+c2] += A*le*c**2*rho/2 + A*le*rho*s**2/2
        M[1+c2, 1+c2] += A*le*c**2*rho/2 + A*le*rho*s**2/2

