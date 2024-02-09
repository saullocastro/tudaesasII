import numpy as np
from numba import njit, jit

#NOTE be careful when using the Beam2D with the Truss2D because currently the
#     Truss2D is derived with only 2 DOFs per node, while the Beam2D is defined
#     with 3 DOFs per node
DOF = 2

class Truss2D(object):
    __slots__ = ['n1', 'n2', 'E', 'A', 'le', 'rho']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        # Material Lastrobe Lescalloy
        self.E = None
        self.A = None
        self.le = None
        self.rho = None

@njit
def update_K_M(i, A, E, rho,
        pos1, pos2,
        ncoords, rowK, colK, valK, rowM, colM, valM,
        lumped=False):
    """Update a global stiffness matrix K and mass matrix M
    """
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    theta = np.arctan2(y2-y1, x2-x1)
    le = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    c = np.cos(theta)
    s = np.sin(theta)

    # positions the global matrices
    c1 = DOF*pos1
    c2 = DOF*pos2

    rowK[16*i+0] = 0+c1
    colK[16*i+0] = 0+c1
    valK[16*i+0] = A*E*c**2/le
    rowK[16*i+1] = 0+c1
    colK[16*i+1] = 1+c1
    valK[16*i+1] = A*E*c*s/le
    rowK[16*i+2] = 0+c1
    colK[16*i+2] = 0+c2
    valK[16*i+2] = -A*E*c**2/le
    rowK[16*i+3] = 0+c1
    colK[16*i+3] = 1+c2
    valK[16*i+3] = -A*E*c*s/le
    rowK[16*i+4] = 1+c1
    colK[16*i+4] = 0+c1
    valK[16*i+4] = A*E*c*s/le
    rowK[16*i+5] = 1+c1
    colK[16*i+5] = 1+c1
    valK[16*i+5] = A*E*s**2/le
    rowK[16*i+6] = 1+c1
    colK[16*i+6] = 0+c2
    valK[16*i+6] = -A*E*c*s/le
    rowK[16*i+7] = 1+c1
    colK[16*i+7] = 1+c2
    valK[16*i+7] = -A*E*s**2/le
    rowK[16*i+8] = 0+c2
    colK[16*i+8] = 0+c1
    valK[16*i+8] = -A*E*c**2/le
    rowK[16*i+9] = 0+c2
    colK[16*i+9] = 1+c1
    valK[16*i+9] = -A*E*c*s/le
    rowK[16*i+10] = 0+c2
    colK[16*i+10] = 0+c2
    valK[16*i+10] = A*E*c**2/le
    rowK[16*i+11] = 0+c2
    colK[16*i+11] = 1+c2
    valK[16*i+11] = A*E*c*s/le
    rowK[16*i+12] = 1+c2
    colK[16*i+12] = 0+c1
    valK[16*i+12] = -A*E*c*s/le
    rowK[16*i+13] = 1+c2
    colK[16*i+13] = 1+c1
    valK[16*i+13] = -A*E*s**2/le
    rowK[16*i+14] = 1+c2
    colK[16*i+14] = 0+c2
    valK[16*i+14] = A*E*c*s/le
    rowK[16*i+15] = 1+c2
    colK[16*i+15] = 1+c2
    valK[16*i+15] = A*E*s**2/le

    if not lumped:
        rowM[8*i+0] = 0+c1
        colM[8*i+0] = 0+c1
        valM[8*i+0] = A*le*c**2*rho/3. + A*le*rho*s**2/3.
        rowM[8*i+1] = 0+c1
        colM[8*i+1] = 0+c2
        valM[8*i+1] = A*le*c**2*rho/6. + A*le*rho*s**2/6.
        rowM[8*i+2] = 1+c1
        colM[8*i+2] = 1+c1
        valM[8*i+2] = A*le*c**2*rho/3. + A*le*rho*s**2/3.
        rowM[8*i+3] = 1+c1
        colM[8*i+3] = 1+c2
        valM[8*i+3] = A*le*c**2*rho/6. + A*le*rho*s**2/6.
        rowM[8*i+4] = 0+c2
        colM[8*i+4] = 0+c1
        valM[8*i+4] = A*le*c**2*rho/6. + A*le*rho*s**2/6.
        rowM[8*i+5] = 0+c2
        colM[8*i+5] = 0+c2
        valM[8*i+5] = A*le*c**2*rho/3. + A*le*rho*s**2/3.
        rowM[8*i+6] = 1+c2
        colM[8*i+6] = 1+c1
        valM[8*i+6] = A*le*c**2*rho/6. + A*le*rho*s**2/6.
        rowM[8*i+7] = 1+c2
        colM[8*i+7] = 1+c2
        valM[8*i+7] = A*le*c**2*rho/3. + A*le*rho*s**2/3.

    if lumped:
        rowM[4*i+0] = 0+c1
        colM[4*i+0] = 0+c1
        valM[4*i+0] = A*le*c**2*rho/2. + A*le*rho*s**2/2.
        rowM[4*i+1] = 1+c1
        colM[4*i+1] = 1+c1
        valM[4*i+1] = A*le*c**2*rho/2. + A*le*rho*s**2/2.
        rowM[4*i+2] = 0+c2
        colM[4*i+2] = 0+c2
        valM[4*i+2] = A*le*c**2*rho/2. + A*le*rho*s**2/2.
        rowM[4*i+3] = 1+c2
        colM[4*i+3] = 1+c2
        valM[4*i+3] = A*le*c**2*rho/2. + A*le*rho*s**2/2.

