import numpy as np

DOF = 3

class Beam2D(object):
    """Euler-Bernoulli beam element

    Formulated using Euler-Bernoulli beam element with two interpolation
    polynomials available:
        - Hermitian cubic
        - Legendre

    """
    __slots__ = ['n1', 'n2', 'E', 'rho', 'Izz1', 'Izz2', 'A1', 'A2',
            'interpolation', 'le', 'thetarad']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        # Material Lastrobe Lescalloy
        self.E = 203.e9 # Pa
        self.rho = 7.83e3 # kg/m3
        self.interpolation = 'hermitian_cubic'
        self.le = None
        self.thetarad = None

def update_K(beam, nid_pos, ncoords, K):
    """Update global K with beam element

    Properties
    ----------
    beam : `.Beam` object
        The beam element being added to K
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates
    K : np.array
        Global stiffness matrix

    """
    pos1 = nid_pos[beam.n1]
    pos2 = nid_pos[beam.n2]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    E = beam.E
    rho = beam.rho
    Izz1 = beam.Izz1
    Izz2 = beam.Izz2
    A1 = beam.A1
    A2 = beam.A2
    le = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    beam.le = le
    beam.thetarad = np.arctan2(y2 - y1, x2 - x1)
    cosr = np.cos(beam.thetarad)
    sinr = np.sin(beam.thetarad)

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2

    if beam.interpolation == 'hermitian_cubic':
        K[0+c1, 0+c1] += E*(cosr**2*le**2*(A1 + A2) + 12*sinr**2*(Izz1 + Izz2))/(2*le**3)
        K[0+c1, 1+c1] += E*cosr*sinr*(-12*Izz1 - 12*Izz2 + le**2*(A1 + A2))/(2*le**3)
        K[0+c1, 2+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c1, 0+c2] += -E*(cosr**2*le**2*(A1 + A2) + 12*sinr**2*(Izz1 + Izz2))/(2*le**3)
        K[0+c1, 1+c2] += E*cosr*sinr*(12*Izz1 + 12*Izz2 - le**2*(A1 + A2))/(2*le**3)
        K[0+c1, 2+c2] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c1, 0+c1] += E*cosr*sinr*(-12*Izz1 - 12*Izz2 + le**2*(A1 + A2))/(2*le**3)
        K[1+c1, 1+c1] += E*(12*cosr**2*(Izz1 + Izz2) + le**2*sinr**2*(A1 + A2))/(2*le**3)
        K[1+c1, 2+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c1, 0+c2] += E*cosr*sinr*(12*Izz1 + 12*Izz2 - le**2*(A1 + A2))/(2*le**3)
        K[1+c1, 1+c2] += -E*(12*cosr**2*(Izz1 + Izz2) + le**2*sinr**2*(A1 + A2))/(2*le**3)
        K[1+c1, 2+c2] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c1, 0+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c1] += E*(3*Izz1 + Izz2)/le
        K[2+c1, 0+c2] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c2] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c2] += E*(Izz1 + Izz2)/le
        K[0+c2, 0+c1] += -E*(cosr**2*le**2*(A1 + A2) + 12*sinr**2*(Izz1 + Izz2))/(2*le**3)
        K[0+c2, 1+c1] += E*cosr*sinr*(12*Izz1 + 12*Izz2 - le**2*(A1 + A2))/(2*le**3)
        K[0+c2, 2+c1] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c2, 0+c2] += E*(cosr**2*le**2*(A1 + A2) + 12*sinr**2*(Izz1 + Izz2))/(2*le**3)
        K[0+c2, 1+c2] += E*cosr*sinr*(-12*Izz1 - 12*Izz2 + le**2*(A1 + A2))/(2*le**3)
        K[0+c2, 2+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c2, 0+c1] += E*cosr*sinr*(12*Izz1 + 12*Izz2 - le**2*(A1 + A2))/(2*le**3)
        K[1+c2, 1+c1] += -E*(12*cosr**2*(Izz1 + Izz2) + le**2*sinr**2*(A1 + A2))/(2*le**3)
        K[1+c2, 2+c1] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c2, 0+c2] += E*cosr*sinr*(-12*Izz1 - 12*Izz2 + le**2*(A1 + A2))/(2*le**3)
        K[1+c2, 1+c2] += E*(12*cosr**2*(Izz1 + Izz2) + le**2*sinr**2*(A1 + A2))/(2*le**3)
        K[1+c2, 2+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 0+c1] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c1] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c1] += E*(Izz1 + Izz2)/le
        K[2+c2, 0+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c2] += E*(Izz1 + 3*Izz2)/le

    elif beam.interpolation == 'legendre':
        K[0+c1, 0+c1] += E*(cosr**2*le**2*(A1 + A2) + 12*sinr**2*(Izz1 + Izz2))/(2*le**3)
        K[0+c1, 1+c1] += E*cosr*sinr*(-12*Izz1 - 12*Izz2 + le**2*(A1 + A2))/(2*le**3)
        K[0+c1, 2+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c1, 0+c2] += -E*(cosr**2*le**2*(A1 + A2) + 12*sinr**2*(Izz1 + Izz2))/(2*le**3)
        K[0+c1, 1+c2] += E*cosr*sinr*(12*Izz1 + 12*Izz2 - le**2*(A1 + A2))/(2*le**3)
        K[0+c1, 2+c2] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c1, 0+c1] += E*cosr*sinr*(-12*Izz1 - 12*Izz2 + le**2*(A1 + A2))/(2*le**3)
        K[1+c1, 1+c1] += E*(12*cosr**2*(Izz1 + Izz2) + le**2*sinr**2*(A1 + A2))/(2*le**3)
        K[1+c1, 2+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c1, 0+c2] += E*cosr*sinr*(12*Izz1 + 12*Izz2 - le**2*(A1 + A2))/(2*le**3)
        K[1+c1, 1+c2] += -E*(12*cosr**2*(Izz1 + Izz2) + le**2*sinr**2*(A1 + A2))/(2*le**3)
        K[1+c1, 2+c2] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c1, 0+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c1] += E*(3*Izz1 + Izz2)/le
        K[2+c1, 0+c2] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c2] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c2] += E*(Izz1 + Izz2)/le
        K[0+c2, 0+c1] += -E*(cosr**2*le**2*(A1 + A2) + 12*sinr**2*(Izz1 + Izz2))/(2*le**3)
        K[0+c2, 1+c1] += E*cosr*sinr*(12*Izz1 + 12*Izz2 - le**2*(A1 + A2))/(2*le**3)
        K[0+c2, 2+c1] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c2, 0+c2] += E*(cosr**2*le**2*(A1 + A2) + 12*sinr**2*(Izz1 + Izz2))/(2*le**3)
        K[0+c2, 1+c2] += E*cosr*sinr*(-12*Izz1 - 12*Izz2 + le**2*(A1 + A2))/(2*le**3)
        K[0+c2, 2+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c2, 0+c1] += E*cosr*sinr*(12*Izz1 + 12*Izz2 - le**2*(A1 + A2))/(2*le**3)
        K[1+c2, 1+c1] += -E*(12*cosr**2*(Izz1 + Izz2) + le**2*sinr**2*(A1 + A2))/(2*le**3)
        K[1+c2, 2+c1] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c2, 0+c2] += E*cosr*sinr*(-12*Izz1 - 12*Izz2 + le**2*(A1 + A2))/(2*le**3)
        K[1+c2, 1+c2] += E*(12*cosr**2*(Izz1 + Izz2) + le**2*sinr**2*(A1 + A2))/(2*le**3)
        K[1+c2, 2+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 0+c1] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c1] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c1] += E*(Izz1 + Izz2)/le
        K[2+c2, 0+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c2] += E*(Izz1 + 3*Izz2)/le

    else:
        raise NotImplementedError('beam interpolation "%s" not implemented' % beam.interpolation)


def update_M(beam, nid_pos, M, lumped=False):
    """Update a global M with a beam element

    Properties
    ----------
    beam : `.Beam` object
        The beam element being added to M
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    M : np.array
        Global mass matrix
    lumped : bool, optional
        If lumped mass should be used

    """
    pos1 = nid_pos[beam.n1]
    pos2 = nid_pos[beam.n2]
    E = beam.E
    rho = beam.rho
    Izz1 = beam.Izz1
    Izz2 = beam.Izz2
    A1 = beam.A1
    A2 = beam.A2
    le = beam.le
    cosr = np.cos(beam.thetarad)
    sinr = np.sin(beam.thetarad)

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2

    if lumped:
        M[0+c1, 0+c1] += le*rho*(3*A1 + A2)*(cosr**2 + sinr**2)/8
        M[1+c1, 1+c1] += le*rho*(3*A1 + A2)*(cosr**2 + sinr**2)/8
        M[2+c1, 2+c1] += le*rho*(5*A1*le**2 + 3*A2*le**2 + 72*Izz1 + 24*Izz2)/192
        M[0+c2, 0+c2] += le*rho*(A1 + 3*A2)*(cosr**2 + sinr**2)/8
        M[1+c2, 1+c2] += le*rho*(A1 + 3*A2)*(cosr**2 + sinr**2)/8
        M[2+c2, 2+c2] += le*rho*(3*A1*le**2 + 5*A2*le**2 + 24*Izz1 + 72*Izz2)/192

    elif not lumped and beam.interpolation == 'hermitian_cubic':
        M[0+c1, 0+c1] += rho*(cosr**2*le**2*(105*A1 + 35*A2) + sinr**2*(120*A1*le**2 + 36*A2*le**2 + 252*Izz1 + 252*Izz2))/(420*le)
        M[0+c1, 1+c1] += -cosr*rho*sinr*(15*A1*le**2 + A2*le**2 + 252*Izz1 + 252*Izz2)/(420*le)
        M[0+c1, 2+c1] += -rho*sinr*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
        M[0+c1, 0+c2] += rho*(35*cosr**2*le**2*(A1 + A2) + 9*sinr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2))/(420*le)
        M[0+c1, 1+c2] += cosr*rho*sinr*(2*A1*le**2 + 2*A2*le**2 + 63*Izz1 + 63*Izz2)/(105*le)
        M[0+c1, 2+c2] += rho*sinr*(7*A1*le**2 + 6*A2*le**2 - 42*Izz1)/420
        M[1+c1, 0+c1] += -cosr*rho*sinr*(15*A1*le**2 + A2*le**2 + 252*Izz1 + 252*Izz2)/(420*le)
        M[1+c1, 1+c1] += rho*(cosr**2*(120*A1*le**2 + 36*A2*le**2 + 252*Izz1 + 252*Izz2) + le**2*sinr**2*(105*A1 + 35*A2))/(420*le)
        M[1+c1, 2+c1] += cosr*rho*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
        M[1+c1, 0+c2] += cosr*rho*sinr*(2*A1*le**2 + 2*A2*le**2 + 63*Izz1 + 63*Izz2)/(105*le)
        M[1+c1, 1+c2] += rho*(9*cosr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2) + 35*le**2*sinr**2*(A1 + A2))/(420*le)
        M[1+c1, 2+c2] += cosr*rho*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
        M[2+c1, 0+c1] += -rho*sinr*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
        M[2+c1, 1+c1] += cosr*rho*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
        M[2+c1, 2+c1] += le*rho*(5*A1*le**2 + 3*A2*le**2 + 84*Izz1 + 28*Izz2)/840
        M[2+c1, 0+c2] += rho*sinr*(-6*A1*le**2 - 7*A2*le**2 + 42*Izz2)/420
        M[2+c1, 1+c2] += cosr*rho*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
        M[2+c1, 2+c2] += -le*rho*(3*A1*le**2 + 3*A2*le**2 + 14*Izz1 + 14*Izz2)/840
        M[0+c2, 0+c1] += rho*(35*cosr**2*le**2*(A1 + A2) + 9*sinr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2))/(420*le)
        M[0+c2, 1+c1] += cosr*rho*sinr*(2*A1*le**2 + 2*A2*le**2 + 63*Izz1 + 63*Izz2)/(105*le)
        M[0+c2, 2+c1] += rho*sinr*(-6*A1*le**2 - 7*A2*le**2 + 42*Izz2)/420
        M[0+c2, 0+c2] += rho*(cosr**2*le**2*(35*A1 + 105*A2) + sinr**2*(36*A1*le**2 + 120*A2*le**2 + 252*Izz1 + 252*Izz2))/(420*le)
        M[0+c2, 1+c2] += -cosr*rho*sinr*(A1*le**2 + 15*A2*le**2 + 252*Izz1 + 252*Izz2)/(420*le)
        M[0+c2, 2+c2] += rho*sinr*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
        M[1+c2, 0+c1] += cosr*rho*sinr*(2*A1*le**2 + 2*A2*le**2 + 63*Izz1 + 63*Izz2)/(105*le)
        M[1+c2, 1+c1] += rho*(9*cosr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2) + 35*le**2*sinr**2*(A1 + A2))/(420*le)
        M[1+c2, 2+c1] += cosr*rho*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
        M[1+c2, 0+c2] += -cosr*rho*sinr*(A1*le**2 + 15*A2*le**2 + 252*Izz1 + 252*Izz2)/(420*le)
        M[1+c2, 1+c2] += rho*(cosr**2*(36*A1*le**2 + 120*A2*le**2 + 252*Izz1 + 252*Izz2) + le**2*sinr**2*(35*A1 + 105*A2))/(420*le)
        M[1+c2, 2+c2] += -cosr*rho*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
        M[2+c2, 0+c1] += rho*sinr*(7*A1*le**2 + 6*A2*le**2 - 42*Izz1)/420
        M[2+c2, 1+c1] += cosr*rho*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
        M[2+c2, 2+c1] += -le*rho*(3*A1*le**2 + 3*A2*le**2 + 14*Izz1 + 14*Izz2)/840
        M[2+c2, 0+c2] += rho*sinr*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
        M[2+c2, 1+c2] += -cosr*rho*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
        M[2+c2, 2+c2] += le*rho*(3*A1*le**2 + 5*A2*le**2 + 28*Izz1 + 84*Izz2)/840

    elif not lumped and beam.interpolation == 'legendre':
        M[0+c1, 0+c1] += rho*(cosr**2*le**2*(105*A1 + 35*A2) + sinr**2*(120*A1*le**2 + 36*A2*le**2 + 252*Izz1 + 252*Izz2))/(420*le)
        M[0+c1, 1+c1] += -cosr*rho*sinr*(15*A1*le**2 + A2*le**2 + 252*Izz1 + 252*Izz2)/(420*le)
        M[0+c1, 2+c1] += -rho*sinr*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
        M[0+c1, 0+c2] += rho*(35*cosr**2*le**2*(A1 + A2) + 9*sinr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2))/(420*le)
        M[0+c1, 1+c2] += cosr*rho*sinr*(2*A1*le**2 + 2*A2*le**2 + 63*Izz1 + 63*Izz2)/(105*le)
        M[0+c1, 2+c2] += rho*sinr*(7*A1*le**2 + 6*A2*le**2 - 42*Izz1)/420
        M[1+c1, 0+c1] += -cosr*rho*sinr*(15*A1*le**2 + A2*le**2 + 252*Izz1 + 252*Izz2)/(420*le)
        M[1+c1, 1+c1] += rho*(cosr**2*(120*A1*le**2 + 36*A2*le**2 + 252*Izz1 + 252*Izz2) + le**2*sinr**2*(105*A1 + 35*A2))/(420*le)
        M[1+c1, 2+c1] += cosr*rho*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
        M[1+c1, 0+c2] += cosr*rho*sinr*(2*A1*le**2 + 2*A2*le**2 + 63*Izz1 + 63*Izz2)/(105*le)
        M[1+c1, 1+c2] += rho*(9*cosr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2) + 35*le**2*sinr**2*(A1 + A2))/(420*le)
        M[1+c1, 2+c2] += cosr*rho*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
        M[2+c1, 0+c1] += -rho*sinr*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
        M[2+c1, 1+c1] += cosr*rho*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
        M[2+c1, 2+c1] += le*rho*(5*A1*le**2 + 3*A2*le**2 + 84*Izz1 + 28*Izz2)/840
        M[2+c1, 0+c2] += rho*sinr*(-6*A1*le**2 - 7*A2*le**2 + 42*Izz2)/420
        M[2+c1, 1+c2] += cosr*rho*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
        M[2+c1, 2+c2] += -le*rho*(3*A1*le**2 + 3*A2*le**2 + 14*Izz1 + 14*Izz2)/840
        M[0+c2, 0+c1] += rho*(35*cosr**2*le**2*(A1 + A2) + 9*sinr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2))/(420*le)
        M[0+c2, 1+c1] += cosr*rho*sinr*(2*A1*le**2 + 2*A2*le**2 + 63*Izz1 + 63*Izz2)/(105*le)
        M[0+c2, 2+c1] += rho*sinr*(-6*A1*le**2 - 7*A2*le**2 + 42*Izz2)/420
        M[0+c2, 0+c2] += rho*(cosr**2*le**2*(35*A1 + 105*A2) + sinr**2*(36*A1*le**2 + 120*A2*le**2 + 252*Izz1 + 252*Izz2))/(420*le)
        M[0+c2, 1+c2] += -cosr*rho*sinr*(A1*le**2 + 15*A2*le**2 + 252*Izz1 + 252*Izz2)/(420*le)
        M[0+c2, 2+c2] += rho*sinr*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
        M[1+c2, 0+c1] += cosr*rho*sinr*(2*A1*le**2 + 2*A2*le**2 + 63*Izz1 + 63*Izz2)/(105*le)
        M[1+c2, 1+c1] += rho*(9*cosr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2) + 35*le**2*sinr**2*(A1 + A2))/(420*le)
        M[1+c2, 2+c1] += cosr*rho*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
        M[1+c2, 0+c2] += -cosr*rho*sinr*(A1*le**2 + 15*A2*le**2 + 252*Izz1 + 252*Izz2)/(420*le)
        M[1+c2, 1+c2] += rho*(cosr**2*(36*A1*le**2 + 120*A2*le**2 + 252*Izz1 + 252*Izz2) + le**2*sinr**2*(35*A1 + 105*A2))/(420*le)
        M[1+c2, 2+c2] += -cosr*rho*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
        M[2+c2, 0+c1] += rho*sinr*(7*A1*le**2 + 6*A2*le**2 - 42*Izz1)/420
        M[2+c2, 1+c1] += cosr*rho*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
        M[2+c2, 2+c1] += -le*rho*(3*A1*le**2 + 3*A2*le**2 + 14*Izz1 + 14*Izz2)/840
        M[2+c2, 0+c2] += rho*sinr*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
        M[2+c2, 1+c2] += -cosr*rho*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
        M[2+c2, 2+c2] += le*rho*(3*A1*le**2 + 5*A2*le**2 + 28*Izz1 + 84*Izz2)/840

    else:
        raise NotImplementedError('beam interpolation "%s" not implemented' % beam.interpolation)

def uv(beam, u1, v1, beta1, u2, v2, beta2, n=100):
    """Calculate u and v for a Beam2D

    Parameters
    ----------
    beam : Beam2D
        The Beam2D finite element
    u1, v1, beta1, u2, v2, beta2 : float or array-like
        Nodal displacements and rotations
    n : int
        Number of points where the axial strain should be calculated within the
        beam element

    Returns
    -------
    uv : (2, :, n) array-like
        Displacements `u` and `uv`  at all `n` points. The second array
        dimension depends on the dimension of the nodal displacements and
        rotations
    """
    inputs = [u1, v1, beta1, u2, v2, beta2]
    inputs = list(map(np.atleast_1d, inputs))
    maxshape = max([np.shape(i)[0] for i in inputs])
    for i in range(len(inputs)):
        if inputs[i].shape[0] == 1:
            inputs[i] = np.ones(maxshape)*inputs[i][0]
        else:
            assert inputs[i].shape[0] == maxshape
    u1, v1, beta1, u2, v2, beta2 = inputs
    # transforming displacements to element's coordinates
    cosr = np.cos(beam.thetarad)
    sinr = np.sin(beam.thetarad)
    u1e = cosr*u1 + sinr*v1
    v1e = -sinr*u1 + cosr*v1
    beta1e = beta1
    u2e = cosr*u2 + sinr*v2
    v2e = -sinr*u2 + cosr*v2
    beta2e = beta2
    # calculating u, v
    le = beam.le
    xi = np.linspace(-1, +1, n)
    Nu1 = u1e[:, None]*(1-xi)/2
    Nu2 = u2e[:, None]*(1+xi)/2
    Nv1 = v1e[:, None]*(1/2 - 3*xi/4 + 1*xi**3/4)
    Nv2 = beta1e[:, None]*(le*(1/8 - 1*xi/8 - 1*xi**2/8 + 1*xi**3/8))
    Nv3 = v2e[:, None]*(1/2 + 3*xi/4 - 1*xi**3/4)
    Nv4 = beta2e[:, None]*(le*(-1/8 - 1*xi/8 + 1*xi**2/8 + 1*xi**3/8))
    ue = Nu1+Nu2
    ve = Nv1+Nv2+Nv3+Nv4
    # transforming displacements to global coordinates
    u = cosr*ue - sinr*ve
    v = sinr*ue + cosr*ve
    # final shape will be (uv, n, maxshape)
    return np.array([u, v])

def exx(beam, y, xi, u1, v1, beta1, u2, v2, beta2):
    """Calculate the axial strain exx for a Beam2D element

    Strains are calculated assuming constant rot

    exx = exx0 + y*kxx

    |exx0| = |BL||u1, v1, beta1, u2, v2, beta2|^T
    |kxx |

    Parameters
    ----------
    beam : Beam2D
        The Beam2D finite element
    y : float
        Distance from the neutral axis
    xi : float
        Natural coordinate along beam axis
    u1, v1, beta1, u2, v2, beta2 : float or array-like
        Nodal displacements and rotations in global coordinates

    Returns
    -------
    exx : float
        The calculated axial strain

    """
    cosr = np.cos(beam.thetarad)
    sinr = np.sin(beam.thetarad)
    le = beam.le
    BL = np.array([[-cosr/le, -sinr/le, 0, cosr/le, sinr/le, 0],
                   [6*sinr*xi/le**2, -6*cosr*xi/le**2, 2*(1/2 - 3*xi/2)/le, -6*sinr*xi/le**2, 6*cosr*xi/le**2, 2*(-3*xi/2 - 1/2)/le]])
    exx0, kxx = BL @ np.array([u1, v1, beta1, u2, v2, beta2])
    return exx0 + y*kxx
