import numpy as np

DOF = 3


class Beam2D(object):
    """Euler-Bernoulli beam element

    Formulated using Euler-Bernoulli beam element with two interpolation
    polynomials available:

        - Hermitian cubic
        - Legendre

    Attributes
    ----------
    le : double
        Element length.
    E : double
        Elastic modulus.
    rho : double
        Material density.
    A1, A2 : double
        Cross section area at nodes 1 and 2.
    Izz1, Izz2 : double
        Second moment of area (moment of inertia) at nodes 1 and 2.
    n1, n2 : int
        Node identification number of the two beam nodes.
    thetarad : double
        Beam orientation with respect to the horizontal direction.
    interpolation : str
        Either ``'hermitian_cubic'`` or ``'legendre'``.

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
    beam : `.Beam2D` object
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


def update_KG(beam, Ppreload_u, u0, nid_pos, ncoords, KG):
    """Update geometric stiffness matrix KG with beam element

    Properties
    ----------
    beam : `.Beam2D` object
        The beam element being added to KG
    Ppreload_u: float or array-like
        A constant load applied to pre-stress the beam element ``Ppreload``; or
        a displacement state ``u`` in global coordinates.
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates
    KG : np.array
        Global geometric stiffness matrix

    """
    pos1 = nid_pos[beam.n1]
    pos2 = nid_pos[beam.n2]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    le = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    beam.le = le
    E = beam.E
    A1 = beam.A1
    A2 = beam.A2
    beam.thetarad = np.arctan2(y2 - y1, x2 - x1)
    cosr = np.cos(beam.thetarad)
    sinr = np.sin(beam.thetarad)

    # positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2


    if isinstance(Ppreload_u, float):
        ucon = np.zeros(2*DOF)
        Ppreload = Ppreload_u
    else:
        u = Ppreload_u
        ucon = np.concatenate((u[c1:c1+DOF], u[c2:c2+DOF]))
        u0con = np.concatenate((u0[c1:c1+DOF], u0[c2:c2+DOF]))

    # NOTE 2-point Gauss-Legendre quadrature
    points = [-0.577350269189625764509148780501957455647601751270126876,
               0.577350269189625764509148780501957455647601751270126876]
    weights = [1., 1.]

    # NOTE omitting weight because it is equal to 1.
    if isinstance(Ppreload_u, float):
        KG[0+c1, 0+c1] += Ppreload*(5*cosr**2 + 6*sinr**2)/(5*le)
        KG[0+c1, 1+c1] += -Ppreload*cosr*sinr/(5*le)
        KG[0+c1, 2+c1] += -Ppreload*sinr/10
        KG[0+c1, 0+c2] += -Ppreload*(5*cosr**2 + 6*sinr**2)/(5*le)
        KG[0+c1, 1+c2] += Ppreload*cosr*sinr/(5*le)
        KG[0+c1, 2+c2] += -Ppreload*sinr/10
        KG[1+c1, 0+c1] += -Ppreload*cosr*sinr/(5*le)
        KG[1+c1, 1+c1] += Ppreload*(6*cosr**2 + 5*sinr**2)/(5*le)
        KG[1+c1, 2+c1] += Ppreload*cosr/10
        KG[1+c1, 0+c2] += Ppreload*cosr*sinr/(5*le)
        KG[1+c1, 1+c2] += -Ppreload*(6*cosr**2 + 5*sinr**2)/(5*le)
        KG[1+c1, 2+c2] += Ppreload*cosr/10
        KG[2+c1, 0+c1] += -Ppreload*sinr/10
        KG[2+c1, 1+c1] += Ppreload*cosr/10
        KG[2+c1, 2+c1] += 2*Ppreload*le/15
        KG[2+c1, 0+c2] += Ppreload*sinr/10
        KG[2+c1, 1+c2] += -Ppreload*cosr/10
        KG[2+c1, 2+c2] += -Ppreload*le/30
        KG[0+c2, 0+c1] += -Ppreload*(5*cosr**2 + 6*sinr**2)/(5*le)
        KG[0+c2, 1+c1] += Ppreload*cosr*sinr/(5*le)
        KG[0+c2, 2+c1] += Ppreload*sinr/10
        KG[0+c2, 0+c2] += Ppreload*(5*cosr**2 + 6*sinr**2)/(5*le)
        KG[0+c2, 1+c2] += -Ppreload*cosr*sinr/(5*le)
        KG[0+c2, 2+c2] += Ppreload*sinr/10
        KG[1+c2, 0+c1] += Ppreload*cosr*sinr/(5*le)
        KG[1+c2, 1+c1] += -Ppreload*(6*cosr**2 + 5*sinr**2)/(5*le)
        KG[1+c2, 2+c1] += -Ppreload*cosr/10
        KG[1+c2, 0+c2] += -Ppreload*cosr*sinr/(5*le)
        KG[1+c2, 1+c2] += Ppreload*(6*cosr**2 + 5*sinr**2)/(5*le)
        KG[1+c2, 2+c2] += -Ppreload*cosr/10
        KG[2+c2, 0+c1] += -Ppreload*sinr/10
        KG[2+c2, 1+c1] += Ppreload*cosr/10
        KG[2+c2, 2+c1] += -Ppreload*le/30
        KG[2+c2, 0+c2] += Ppreload*sinr/10
        KG[2+c2, 1+c2] += -Ppreload*cosr/10
        KG[2+c2, 2+c2] += 2*Ppreload*le/15

    elif beam.interpolation == 'hermitian_cubic':
        for i in range(2):
            xi = points[i]
            ux = -cosr*ucon[0]/le + cosr*ucon[3]/le - sinr*ucon[1]/le + sinr*ucon[4]/le
            vx = 2*cosr*ucon[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*ucon[4]*(3/4 - 3*xi**2/4)/le + ucon[2]*(3*xi**2/4 - xi/2 - 1/4) + ucon[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*ucon[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*ucon[3]*(3/4 - 3*xi**2/4)/le
            v0x = 2*cosr*u0con[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*u0con[4]*(3/4 - 3*xi**2/4)/le + u0con[2]*(3*xi**2/4 - xi/2 - 1/4) + u0con[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*u0con[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*u0con[3]*(3/4 - 3*xi**2/4)/le
            Pxx = E*ucon[2]*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(le*(1 - xi)**2/8 + le*(xi + 1)*(2*xi - 2)/8)/le + vx*(le*(1 - xi)**2/8 + le*(xi + 1)*(2*xi - 2)/8)/le) + E*ucon[5]*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(le*(xi - 1)*(2*xi + 2)/8 + le*(xi + 1)**2/8)/le + vx*(le*(xi - 1)*(2*xi + 2)/8 + le*(xi + 1)**2/8)/le) + ucon[0]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(-ux/(2*le) - 1/le) - E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le + vx*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le)) + ucon[1]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le + vx*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le) + E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(-ux/(2*le) - 1/le)) + ucon[3]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(ux/(2*le) + 1/le) - E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le + vx*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le)) + ucon[4]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le + vx*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le) + E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(ux/(2*le) + 1/le))

            KG[0+c1, 0+c1] += Pxx*(4*cosr**2 + 9*sinr**2*(xi - 1)**2*(xi + 1)**2)/(8*le)
            KG[0+c1, 1+c1] += Pxx*cosr*sinr*(-9*xi**4 + 18*xi**2 - 5)/(8*le)
            KG[0+c1, 2+c1] += -3*Pxx*sinr*(xi - 1)**2*(xi + 1)*(3*xi + 1)/16
            KG[0+c1, 0+c2] += -Pxx*(4*cosr**2 + 9*sinr**2*(xi - 1)**2*(xi + 1)**2)/(8*le)
            KG[0+c1, 1+c2] += Pxx*cosr*sinr*(9*xi**4 - 18*xi**2 + 5)/(8*le)
            KG[0+c1, 2+c2] += -3*Pxx*sinr*(xi - 1)*(xi + 1)**2*(3*xi - 1)/16
            KG[1+c1, 0+c1] += Pxx*cosr*sinr*(-9*xi**4 + 18*xi**2 - 5)/(8*le)
            KG[1+c1, 1+c1] += Pxx*(9*cosr**2*(xi - 1)**2*(xi + 1)**2 + 4*sinr**2)/(8*le)
            KG[1+c1, 2+c1] += 3*Pxx*cosr*(xi - 1)**2*(xi + 1)*(3*xi + 1)/16
            KG[1+c1, 0+c2] += Pxx*cosr*sinr*(9*xi**4 - 18*xi**2 + 5)/(8*le)
            KG[1+c1, 1+c2] += -Pxx*(9*cosr**2*(xi - 1)**2*(xi + 1)**2 + 4*sinr**2)/(8*le)
            KG[1+c1, 2+c2] += 3*Pxx*cosr*(xi - 1)*(xi + 1)**2*(3*xi - 1)/16
            KG[2+c1, 0+c1] += -3*Pxx*sinr*(xi - 1)**2*(xi + 1)*(3*xi + 1)/16
            KG[2+c1, 1+c1] += 3*Pxx*cosr*(xi - 1)**2*(xi + 1)*(3*xi + 1)/16
            KG[2+c1, 2+c1] += Pxx*le*(xi - 1)**2*(3*xi + 1)**2/32
            KG[2+c1, 0+c2] += 3*Pxx*sinr*(xi - 1)**2*(xi + 1)*(3*xi + 1)/16
            KG[2+c1, 1+c2] += -3*Pxx*cosr*(xi - 1)**2*(xi + 1)*(3*xi + 1)/16
            KG[2+c1, 2+c2] += Pxx*le*(9*xi**4 - 10*xi**2 + 1)/32
            KG[0+c2, 0+c1] += -Pxx*(4*cosr**2 + 9*sinr**2*(xi - 1)**2*(xi + 1)**2)/(8*le)
            KG[0+c2, 1+c1] += Pxx*cosr*sinr*(9*xi**4 - 18*xi**2 + 5)/(8*le)
            KG[0+c2, 2+c1] += 3*Pxx*sinr*(xi - 1)**2*(xi + 1)*(3*xi + 1)/16
            KG[0+c2, 0+c2] += Pxx*(4*cosr**2 + 9*sinr**2*(xi - 1)**2*(xi + 1)**2)/(8*le)
            KG[0+c2, 1+c2] += Pxx*cosr*sinr*(-9*xi**4 + 18*xi**2 - 5)/(8*le)
            KG[0+c2, 2+c2] += 3*Pxx*sinr*(xi - 1)*(xi + 1)**2*(3*xi - 1)/16
            KG[1+c2, 0+c1] += Pxx*cosr*sinr*(9*xi**4 - 18*xi**2 + 5)/(8*le)
            KG[1+c2, 1+c1] += -Pxx*(9*cosr**2*(xi - 1)**2*(xi + 1)**2 + 4*sinr**2)/(8*le)
            KG[1+c2, 2+c1] += -3*Pxx*cosr*(xi - 1)**2*(xi + 1)*(3*xi + 1)/16
            KG[1+c2, 0+c2] += Pxx*cosr*sinr*(-9*xi**4 + 18*xi**2 - 5)/(8*le)
            KG[1+c2, 1+c2] += Pxx*(9*cosr**2*(xi - 1)**2*(xi + 1)**2 + 4*sinr**2)/(8*le)
            KG[1+c2, 2+c2] += -3*Pxx*cosr*(xi - 1)*(xi + 1)**2*(3*xi - 1)/16
            KG[2+c2, 0+c1] += -3*Pxx*sinr*(xi - 1)*(xi + 1)**2*(3*xi - 1)/16
            KG[2+c2, 1+c1] += 3*Pxx*cosr*(xi - 1)*(xi + 1)**2*(3*xi - 1)/16
            KG[2+c2, 2+c1] += Pxx*le*(9*xi**4 - 10*xi**2 + 1)/32
            KG[2+c2, 0+c2] += 3*Pxx*sinr*(xi - 1)*(xi + 1)**2*(3*xi - 1)/16
            KG[2+c2, 1+c2] += -3*Pxx*cosr*(xi - 1)*(xi + 1)**2*(3*xi - 1)/16
            KG[2+c2, 2+c2] += Pxx*le*(xi + 1)**2*(3*xi - 1)**2/32

    elif beam.interpolation == 'legendre':
        for i in range(2):
            xi = points[i]
            ux = -cosr*ucon[0]/le + cosr*ucon[3]/le - sinr*ucon[1]/le + sinr*ucon[4]/le
            vx = 2*cosr*ucon[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*ucon[4]*(3/4 - 3*xi**2/4)/le + ucon[2]*(3*xi**2/4 - xi/2 - 1/4) + ucon[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*ucon[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*ucon[3]*(3/4 - 3*xi**2/4)/le
            v0x = 2*cosr*u0con[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*u0con[4]*(3/4 - 3*xi**2/4)/le + u0con[2]*(3*xi**2/4 - xi/2 - 1/4) + u0con[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*u0con[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*u0con[3]*(3/4 - 3*xi**2/4)/le
            Pxx = E*ucon[2]*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3*xi**2/8 - xi/4 - 1/8) + vx*(3*xi**2/8 - xi/4 - 1/8)) + E*ucon[5]*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3*xi**2/8 + xi/4 - 1/8) + vx*(3*xi**2/8 + xi/4 - 1/8)) + ucon[0]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(-ux/(2*le) - 1/le) - E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3*xi**2/4 - 3/4)/le + vx*(3*xi**2/4 - 3/4)/le)) + ucon[1]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3*xi**2/4 - 3/4)/le + vx*(3*xi**2/4 - 3/4)/le) + E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(-ux/(2*le) - 1/le)) + ucon[3]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(ux/(2*le) + 1/le) - E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3/4 - 3*xi**2/4)/le + vx*(3/4 - 3*xi**2/4)/le)) + ucon[4]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3/4 - 3*xi**2/4)/le + vx*(3/4 - 3*xi**2/4)/le) + E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(ux/(2*le) + 1/le))

            KG[0+c1, 0+c1] += Pxx*(4*cosr**2 + 9*sinr**2*(xi**2 - 1)**2)/(8*le)
            KG[0+c1, 1+c1] += Pxx*cosr*sinr*(4 - 9*(xi**2 - 1)**2)/(8*le)
            KG[0+c1, 2+c1] += 3*Pxx*sinr*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1)/16
            KG[0+c1, 0+c2] += -Pxx*(4*cosr**2 + 9*sinr**2*(xi**2 - 1)**2)/(8*le)
            KG[0+c1, 1+c2] += Pxx*cosr*sinr*(9*(xi**2 - 1)**2 - 4)/(8*le)
            KG[0+c1, 2+c2] += -3*Pxx*sinr*(xi**2 - 1)*(3*xi**2 + 2*xi - 1)/16
            KG[1+c1, 0+c1] += Pxx*cosr*sinr*(4 - 9*(xi**2 - 1)**2)/(8*le)
            KG[1+c1, 1+c1] += Pxx*(9*cosr**2*(xi**2 - 1)**2 + 4*sinr**2)/(8*le)
            KG[1+c1, 2+c1] += -3*Pxx*cosr*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1)/16
            KG[1+c1, 0+c2] += Pxx*cosr*sinr*(9*(xi**2 - 1)**2 - 4)/(8*le)
            KG[1+c1, 1+c2] += -Pxx*(9*cosr**2*(xi**2 - 1)**2 + 4*sinr**2)/(8*le)
            KG[1+c1, 2+c2] += 3*Pxx*cosr*(xi**2 - 1)*(3*xi**2 + 2*xi - 1)/16
            KG[2+c1, 0+c1] += 3*Pxx*sinr*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1)/16
            KG[2+c1, 1+c1] += -3*Pxx*cosr*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1)/16
            KG[2+c1, 2+c1] += Pxx*le*(-3*xi**2 + 2*xi + 1)**2/32
            KG[2+c1, 0+c2] += -3*Pxx*sinr*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1)/16
            KG[2+c1, 1+c2] += 3*Pxx*cosr*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1)/16
            KG[2+c1, 2+c2] += Pxx*le*(9*xi**4 - 10*xi**2 + 1)/32
            KG[0+c2, 0+c1] += -Pxx*(4*cosr**2 + 9*sinr**2*(xi**2 - 1)**2)/(8*le)
            KG[0+c2, 1+c1] += Pxx*cosr*sinr*(9*(xi**2 - 1)**2 - 4)/(8*le)
            KG[0+c2, 2+c1] += -3*Pxx*sinr*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1)/16
            KG[0+c2, 0+c2] += Pxx*(4*cosr**2 + 9*sinr**2*(xi**2 - 1)**2)/(8*le)
            KG[0+c2, 1+c2] += Pxx*cosr*sinr*(4 - 9*(xi**2 - 1)**2)/(8*le)
            KG[0+c2, 2+c2] += 3*Pxx*sinr*(xi**2 - 1)*(3*xi**2 + 2*xi - 1)/16
            KG[1+c2, 0+c1] += Pxx*cosr*sinr*(9*(xi**2 - 1)**2 - 4)/(8*le)
            KG[1+c2, 1+c1] += -Pxx*(9*cosr**2*(xi**2 - 1)**2 + 4*sinr**2)/(8*le)
            KG[1+c2, 2+c1] += 3*Pxx*cosr*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1)/16
            KG[1+c2, 0+c2] += Pxx*cosr*sinr*(4 - 9*(xi**2 - 1)**2)/(8*le)
            KG[1+c2, 1+c2] += Pxx*(9*cosr**2*(xi**2 - 1)**2 + 4*sinr**2)/(8*le)
            KG[1+c2, 2+c2] += -3*Pxx*cosr*(xi**2 - 1)*(3*xi**2 + 2*xi - 1)/16
            KG[2+c2, 0+c1] += -3*Pxx*sinr*(xi**2 - 1)*(3*xi**2 + 2*xi - 1)/16
            KG[2+c2, 1+c1] += 3*Pxx*cosr*(xi**2 - 1)*(3*xi**2 + 2*xi - 1)/16
            KG[2+c2, 2+c1] += Pxx*le*(9*xi**4 - 10*xi**2 + 1)/32
            KG[2+c2, 0+c2] += 3*Pxx*sinr*(xi**2 - 1)*(3*xi**2 + 2*xi - 1)/16
            KG[2+c2, 1+c2] += -3*Pxx*cosr*(xi**2 - 1)*(3*xi**2 + 2*xi - 1)/16
            KG[2+c2, 2+c2] += Pxx*le*(3*xi**2 + 2*xi - 1)**2/32


def update_KNL(beam, u, u0, nid_pos, ncoords, KNL):
    """Update the nonlinear part of global constitutive stiffness KNL with beam element

    Properties
    ----------
    beam : `.Beam2D` object
        The beam element being added to KNL
    u: array-like
        Displacement state ``u`` in global coordinates.
    u0: array-like
        Imperfection state ``u0`` in global coordinates.
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates
    KNL : np.array
        Nonlinear part of global constitutive stiffness matrix

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

    ucon = np.concatenate((u[c1:c1+DOF], u[c2:c2+DOF]))
    u0con = np.concatenate((u0[c1:c1+DOF], u0[c2:c2+DOF]))

    #NOTE 2-point Gauss-Legendre quadrature
    points = [-0.577350269189625764509148780501957455647601751270126876,
               0.577350269189625764509148780501957455647601751270126876]
    weights = [1., 1.]

    # NOTE omitting weight because it is equal to 1.
    if beam.interpolation == 'hermitian_cubic':
        for i in range(2):
            xi = points[i]
            ux = -cosr*ucon[0]/le + cosr*ucon[3]/le - sinr*ucon[1]/le + sinr*ucon[4]/le
            vx = 2*cosr*ucon[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*ucon[4]*(3/4 - 3*xi**2/4)/le + ucon[2]*(3*xi**2/4 - xi/2 - 1/4) + ucon[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*ucon[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*ucon[3]*(3/4 - 3*xi**2/4)/le
            v0x = 2*cosr*u0con[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*u0con[4]*(3/4 - 3*xi**2/4)/le + u0con[2]*(3*xi**2/4 - xi/2 - 1/4) + u0con[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*u0con[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*u0con[3]*(3/4 - 3*xi**2/4)/le
            KNL[0+c1, 0+c1] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi - 1)*(xi + 1)) + 3*sinr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))))/(16*le**3)
            KNL[0+c1, 1+c1] += E*(-3*cosr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi - 1)*(xi + 1)))/(16*le**3)
            KNL[0+c1, 2+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(3*xi + 1) - 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)*(3*xi + 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)))/(32*le**2)
            KNL[0+c1, 0+c2] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi - 1)*(xi + 1)) + 3*sinr*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1) + 3*sinr*(le**2*(-2*A1 + (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + xi**2*(-32*Izz1 + 16*(Izz1 - Izz2)*(xi + 1)))))/(16*le**3)
            KNL[0+c1, 1+c2] += E*(-3*cosr*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1) + 3*sinr*(le**2*(-2*A1 + (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + xi**2*(-32*Izz1 + 16*(Izz1 - Izz2)*(xi + 1)))) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi - 1)*(xi + 1)))/(16*le**3)
            KNL[0+c1, 2+c2] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi + 1)*(3*xi - 1) - 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1)**2*(3*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)))/(32*le**2)
            KNL[1+c1, 0+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi - 1)*(xi + 1) - 2*sinr*(ux + 1)) + 3*sinr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1)))/(16*le**3)
            KNL[1+c1, 1+c1] += E*(-3*cosr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi - 1)*(xi + 1) - 2*sinr*(ux + 1)))/(16*le**3)
            KNL[1+c1, 2+c1] += E*(3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)*(3*xi + 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(3*xi + 1))/(32*le**2)
            KNL[1+c1, 0+c2] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi - 1)*(xi + 1) - 2*sinr*(ux + 1)) - 3*sinr*(3*cosr*(le**2*(-2*A1 + (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(-2*Izz1 + (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1)))/(16*le**3)
            KNL[1+c1, 1+c2] += E*(3*cosr*(3*cosr*(le**2*(-2*A1 + (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(-2*Izz1 + (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi - 1)*(xi + 1) - 2*sinr*(ux + 1)))/(16*le**3)
            KNL[1+c1, 2+c2] += E*(3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1)**2*(3*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi + 1)*(3*xi - 1))/(32*le**2)
            KNL[2+c1, 0+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(3*xi + 1) - 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)*(3*xi + 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)))/(32*le**2)
            KNL[2+c1, 1+c1] += E*(3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)*(3*xi + 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(3*xi + 1))/(32*le**2)
            KNL[2+c1, 2+c1] += E*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(3*xi + 1)**2 + 16*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)**2)/(64*le)
            KNL[2+c1, 0+c2] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(3*xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)*(3*xi + 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)))/(32*le**2)
            KNL[2+c1, 1+c2] += E*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)*(3*xi + 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(3*xi + 1))/(32*le**2)
            KNL[2+c1, 2+c2] += E*(3*xi - 1)*(3*xi + 1)*(32*Izz1 + le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1) - 16*(Izz1 - Izz2)*(xi + 1))/(64*le)
            KNL[0+c2, 0+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi - 1)*(xi + 1)) - 3*sinr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))))/(16*le**3)
            KNL[0+c2, 1+c1] += E*(3*cosr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi - 1)*(xi + 1)))/(16*le**3)
            KNL[0+c2, 2+c1] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(3*xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)*(3*xi + 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)))/(32*le**2)
            KNL[0+c2, 0+c2] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi - 1)*(xi + 1)) + 3*sinr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))))/(16*le**3)
            KNL[0+c2, 1+c2] += E*(-3*cosr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi - 1)*(xi + 1)))/(16*le**3)
            KNL[0+c2, 2+c2] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi + 1)*(3*xi - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1)**2*(3*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)))/(32*le**2)
            KNL[1+c2, 0+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(-3*cosr*(v0x + vx)*(xi - 1)*(xi + 1) + 2*sinr*(ux + 1)) - 3*sinr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1)))/(16*le**3)
            KNL[1+c2, 1+c1] += E*(3*cosr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(-3*cosr*(v0x + vx)*(xi - 1)*(xi + 1) + 2*sinr*(ux + 1)))/(16*le**3)
            KNL[1+c2, 2+c1] += E*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)*(3*xi + 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(3*xi + 1))/(32*le**2)
            KNL[1+c2, 0+c2] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi - 1)*(xi + 1) - 2*sinr*(ux + 1)) + 3*sinr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1)))/(16*le**3)
            KNL[1+c2, 1+c2] += E*(-3*cosr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)**2*(xi + 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi - 1)*(xi + 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi - 1)*(xi + 1) - 2*sinr*(ux + 1)))/(16*le**3)
            KNL[1+c2, 2+c2] += E*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1)**2*(3*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi + 1)*(3*xi - 1))/(32*le**2)
            KNL[2+c2, 0+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi + 1)*(3*xi - 1) - 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1)**2*(3*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)))/(32*le**2)
            KNL[2+c2, 1+c1] += E*(3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1)**2*(3*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi + 1)*(3*xi - 1))/(32*le**2)
            KNL[2+c2, 2+c1] += E*(3*xi - 1)*(3*xi + 1)*(32*Izz1 + le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1) - 16*(Izz1 - Izz2)*(xi + 1))/(64*le)
            KNL[2+c2, 0+c2] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi + 1)*(3*xi - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1)**2*(3*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)))/(32*le**2)
            KNL[2+c2, 1+c2] += E*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi - 1)*(xi + 1)**2*(3*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi + 1)*(3*xi - 1))/(32*le**2)
            KNL[2+c2, 2+c2] += E*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi + 1)**2*(3*xi - 1)**2 + 16*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)**2)/(64*le)

    elif beam.interpolation == 'legendre':
        for i in range(2):
            xi = points[i]
            ux = -cosr*ucon[0]/le + cosr*ucon[3]/le - sinr*ucon[1]/le + sinr*ucon[4]/le
            vx = 2*cosr*ucon[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*ucon[4]*(3/4 - 3*xi**2/4)/le + ucon[2]*(3*xi**2/4 - xi/2 - 1/4) + ucon[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*ucon[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*ucon[3]*(3/4 - 3*xi**2/4)/le
            v0x = 2*cosr*u0con[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*u0con[4]*(3/4 - 3*xi**2/4)/le + u0con[2]*(3*xi**2/4 - xi/2 - 1/4) + u0con[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*u0con[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*u0con[3]*(3/4 - 3*xi**2/4)/le
            KNL[0+c1, 0+c1] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi**2 - 1)) + 3*sinr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))))/(16*le**3)
            KNL[0+c1, 1+c1] += E*(-3*cosr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi**2 - 1)))/(16*le**3)
            KNL[0+c1, 2+c1] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(-3*xi**2 + 2*xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1) - 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)))/(32*le**2)
            KNL[0+c1, 0+c2] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi**2 - 1)) - 3*sinr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1) - 3*sinr*(le**2*(-2*A1 + (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + xi**2*(-32*Izz1 + 16*(Izz1 - Izz2)*(xi + 1)))))/(16*le**3)
            KNL[0+c1, 1+c2] += E*(3*cosr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1) - 3*sinr*(le**2*(-2*A1 + (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + xi**2*(-32*Izz1 + 16*(Izz1 - Izz2)*(xi + 1)))) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi**2 - 1)))/(16*le**3)
            KNL[0+c1, 2+c2] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(3*xi**2 + 2*xi - 1) - 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(3*xi**2 + 2*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)))/(32*le**2)
            KNL[1+c1, 0+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi**2 - 1) - 2*sinr*(ux + 1)) + 3*sinr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1)))/(16*le**3)
            KNL[1+c1, 1+c1] += E*(-3*cosr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi**2 - 1) - 2*sinr*(ux + 1)))/(16*le**3)
            KNL[1+c1, 2+c1] += E*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1) - 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(-3*xi**2 + 2*xi + 1))/(32*le**2)
            KNL[1+c1, 0+c2] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi**2 - 1) - 2*sinr*(ux + 1)) - 3*sinr*(3*cosr*(le**2*(-2*A1 + (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + xi**2*(-32*Izz1 + 16*(Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1)))/(16*le**3)
            KNL[1+c1, 1+c2] += E*(3*cosr*(3*cosr*(le**2*(-2*A1 + (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + xi**2*(-32*Izz1 + 16*(Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi**2 - 1) - 2*sinr*(ux + 1)))/(16*le**3)
            KNL[1+c1, 2+c2] += E*(3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(3*xi**2 + 2*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(3*xi**2 + 2*xi - 1))/(32*le**2)
            KNL[2+c1, 0+c1] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(-3*xi**2 + 2*xi + 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1) - 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)))/(32*le**2)
            KNL[2+c1, 1+c1] += E*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1) - 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(-3*xi**2 + 2*xi + 1))/(32*le**2)
            KNL[2+c1, 2+c1] += E*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(-3*xi**2 + 2*xi + 1)**2 + 16*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)**2)/(64*le)
            KNL[2+c1, 0+c2] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(-3*xi**2 + 2*xi + 1) - 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1) - 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)))/(32*le**2)
            KNL[2+c1, 1+c2] += E*(3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1) - 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(-3*xi**2 + 2*xi + 1))/(32*le**2)
            KNL[2+c1, 2+c2] += E*(-le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(-3*xi**2 + 2*xi + 1)*(3*xi**2 + 2*xi - 1) + 16*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)*(3*xi + 1))/(64*le)
            KNL[0+c2, 0+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi**2 - 1)) - 3*sinr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))))/(16*le**3)
            KNL[0+c2, 1+c1] += E*(3*cosr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi**2 - 1)))/(16*le**3)
            KNL[0+c2, 2+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(-3*xi**2 + 2*xi + 1) - 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1) - 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)))/(32*le**2)
            KNL[0+c2, 0+c2] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi**2 - 1)) + 3*sinr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))))/(16*le**3)
            KNL[0+c2, 1+c2] += E*(-3*cosr*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1)))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(2*cosr*(ux + 1) + 3*sinr*(v0x + vx)*(xi**2 - 1)))/(16*le**3)
            KNL[0+c2, 2+c2] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(3*xi**2 + 2*xi - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(3*xi**2 + 2*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)))/(32*le**2)
            KNL[1+c2, 0+c1] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi**2 - 1) - 2*sinr*(ux + 1)) - 3*sinr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1)))/(16*le**3)
            KNL[1+c2, 1+c1] += E*(3*cosr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi**2 - 1) - 2*sinr*(ux + 1)))/(16*le**3)
            KNL[1+c2, 2+c1] += E*(3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(-3*xi**2 + 2*xi + 1) - 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(-3*xi**2 + 2*xi + 1))/(32*le**2)
            KNL[1+c2, 0+c2] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi**2 - 1) - 2*sinr*(ux + 1)) + 3*sinr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1)))/(16*le**3)
            KNL[1+c2, 1+c2] += E*(-3*cosr*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)**2 + 16*xi**2*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(xi**2 - 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(3*cosr*(v0x + vx)*(xi**2 - 1) - 2*sinr*(ux + 1)))/(16*le**3)
            KNL[1+c2, 2+c2] += E*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(3*xi**2 + 2*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(3*xi**2 + 2*xi - 1))/(32*le**2)
            KNL[2+c2, 0+c1] += E*(-2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(3*xi**2 + 2*xi - 1) - 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(3*xi**2 + 2*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)))/(32*le**2)
            KNL[2+c2, 1+c1] += E*(3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(3*xi**2 + 2*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)) - 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(3*xi**2 + 2*xi - 1))/(32*le**2)
            KNL[2+c2, 2+c1] += E*(-le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(-3*xi**2 + 2*xi + 1)*(3*xi**2 + 2*xi - 1) + 16*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi - 1)*(3*xi + 1))/(64*le)
            KNL[2+c2, 0+c2] += E*(2*cosr*le**2*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(3*xi**2 + 2*xi - 1) + 3*sinr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(3*xi**2 + 2*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)))/(32*le**2)
            KNL[2+c2, 1+c2] += E*(-3*cosr*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(xi**2 - 1)*(3*xi**2 + 2*xi - 1) + 16*xi*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)) + 2*le**2*sinr*(2*A1 - (A1 - A2)*(xi + 1))*(ux + 1)*(v0x + vx)*(3*xi**2 + 2*xi - 1))/(32*le**2)
            KNL[2+c2, 2+c2] += E*(le**2*(2*A1 - (A1 - A2)*(xi + 1))*(v0x + vx)**2*(3*xi**2 + 2*xi - 1)**2 + 16*(2*Izz1 - (Izz1 - Izz2)*(xi + 1))*(3*xi + 1)**2)/(64*le)

    else:
        raise NotImplementedError('beam interpolation "%s" not implemented' % beam.interpolation)


def calc_fint(beams, u, u0, nid_pos, ncoords):
    """Calculate the internal force vector

    Properties
    ----------
    beams : list of `.Beam2D`objects
        The beam elements to be added to the internal force vector
    u: array-like
        Displacement state ``u`` in global coordinates.
    u0: array-like
        Imperfection state ``u0`` in global coordinates.
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model

    """
    fint = np.zeros_like(u0)

    for beam in beams:
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

        ucon = np.concatenate((u[c1:c1+DOF], u[c2:c2+DOF]))
        u0con = np.concatenate((u0[c1:c1+DOF], u0[c2:c2+DOF]))

        #NOTE 2-point Gauss-Legendre quadrature
        points = [-0.577350269189625764509148780501957455647601751270126876,
                   0.577350269189625764509148780501957455647601751270126876]
        weights = [1., 1.]

        if beam.interpolation == 'hermitian_cubic':
            for i in range(2):
                xi = points[i]
                ux = -cosr*ucon[0]/le + cosr*ucon[3]/le - sinr*ucon[1]/le + sinr*ucon[4]/le
                vx = 2*cosr*ucon[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*ucon[4]*(3/4 - 3*xi**2/4)/le + ucon[2]*(3*xi**2/4 - xi/2 - 1/4) + ucon[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*ucon[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*ucon[3]*(3/4 - 3*xi**2/4)/le
                v0x = 2*cosr*u0con[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*u0con[4]*(3/4 - 3*xi**2/4)/le + u0con[2]*(3*xi**2/4 - xi/2 - 1/4) + u0con[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*u0con[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*u0con[3]*(3/4 - 3*xi**2/4)/le
                Pxx = E*ucon[2]*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(le*(1 - xi)**2/8 + le*(xi + 1)*(2*xi - 2)/8)/le + vx*(le*(1 - xi)**2/8 + le*(xi + 1)*(2*xi - 2)/8)/le) + E*ucon[5]*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(le*(xi - 1)*(2*xi + 2)/8 + le*(xi + 1)**2/8)/le + vx*(le*(xi - 1)*(2*xi + 2)/8 + le*(xi + 1)**2/8)/le) + ucon[0]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(-ux/(2*le) - 1/le) - E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le + vx*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le)) + ucon[1]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le + vx*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le) + E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(-ux/(2*le) - 1/le)) + ucon[3]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(ux/(2*le) + 1/le) - E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le + vx*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le)) + ucon[4]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le + vx*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le) + E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(ux/(2*le) + 1/le))
                Mxx = -6*E*cosr*ucon[1]*xi*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)/le**2 + 6*E*cosr*ucon[4]*xi*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)/le**2 + 6*E*sinr*ucon[0]*xi*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)/le**2 - 6*E*sinr*ucon[3]*xi*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)/le**2 - 4*E*ucon[2]*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)*(le*(xi + 1)/4 + le*(2*xi - 2)/4)/le**2 - 4*E*ucon[5]*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)*(le*(xi - 1)/4 + le*(2*xi + 2)/4)/le**2

                fint[0 + c1] += Pxx*cosr*le*(-ux/le - 1/le)/2 - sinr*(-3*Mxx*xi/le + Pxx*le*(2*v0x*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le + 2*vx*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le)/2)
                fint[1 + c1] += Pxx*le*sinr*(-ux/le - 1/le)/2 + cosr*(-3*Mxx*xi/le + Pxx*le*(2*v0x*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le + 2*vx*((1 - xi)**2/4 + (xi + 2)*(2*xi - 2)/4)/le)/2)
                fint[2 + c1] += -2*Mxx*(le*(xi + 1)/4 + le*(2*xi - 2)/4)/le + Pxx*le*(2*v0x*(le*(1 - xi)**2/8 + le*(xi + 1)*(2*xi - 2)/8)/le + 2*vx*(le*(1 - xi)**2/8 + le*(xi + 1)*(2*xi - 2)/8)/le)/2
                fint[0 + c2] += Pxx*cosr*le*(ux/le + 1/le)/2 - sinr*(3*Mxx*xi/le + Pxx*le*(2*v0x*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le + 2*vx*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le)/2)
                fint[1 + c2] += Pxx*le*sinr*(ux/le + 1/le)/2 + cosr*(3*Mxx*xi/le + Pxx*le*(2*v0x*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le + 2*vx*((2 - xi)*(2*xi + 2)/4 - (xi + 1)**2/4)/le)/2)
                fint[2 + c2] += -2*Mxx*(le*(xi - 1)/4 + le*(2*xi + 2)/4)/le + Pxx*le*(2*v0x*(le*(xi - 1)*(2*xi + 2)/8 + le*(xi + 1)**2/8)/le + 2*vx*(le*(xi - 1)*(2*xi + 2)/8 + le*(xi + 1)**2/8)/le)/2

        elif beam.interpolation == 'legendre':
            for i in range(2):
                xi = points[i]
                ux = -cosr*ucon[0]/le + cosr*ucon[3]/le - sinr*ucon[1]/le + sinr*ucon[4]/le
                vx = 2*cosr*ucon[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*ucon[4]*(3/4 - 3*xi**2/4)/le + ucon[2]*(3*xi**2/4 - xi/2 - 1/4) + ucon[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*ucon[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*ucon[3]*(3/4 - 3*xi**2/4)/le
                v0x = 2*cosr*u0con[1]*(3*xi**2/4 - 3/4)/le + 2*cosr*u0con[4]*(3/4 - 3*xi**2/4)/le + u0con[2]*(3*xi**2/4 - xi/2 - 1/4) + u0con[5]*(3*xi**2/4 + xi/2 - 1/4) - 2*sinr*u0con[0]*(3*xi**2/4 - 3/4)/le - 2*sinr*u0con[3]*(3/4 - 3*xi**2/4)/le
                Pxx = E*ucon[2]*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3*xi**2/8 - xi/4 - 1/8) + vx*(3*xi**2/8 - xi/4 - 1/8)) + E*ucon[5]*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3*xi**2/8 + xi/4 - 1/8) + vx*(3*xi**2/8 + xi/4 - 1/8)) + ucon[0]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(-ux/(2*le) - 1/le) - E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3*xi**2/4 - 3/4)/le + vx*(3*xi**2/4 - 3/4)/le)) + ucon[1]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3*xi**2/4 - 3/4)/le + vx*(3*xi**2/4 - 3/4)/le) + E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(-ux/(2*le) - 1/le)) + ucon[3]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(ux/(2*le) + 1/le) - E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3/4 - 3*xi**2/4)/le + vx*(3/4 - 3*xi**2/4)/le)) + ucon[4]*(E*cosr*(A1 + (-A1 + A2)*(xi + 1)/2)*(2*v0x*(3/4 - 3*xi**2/4)/le + vx*(3/4 - 3*xi**2/4)/le) + E*sinr*(A1 + (-A1 + A2)*(xi + 1)/2)*(ux/(2*le) + 1/le))
                Mxx = -6*E*cosr*ucon[1]*xi*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)/le**2 + 6*E*cosr*ucon[4]*xi*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)/le**2 + 2*E*ucon[2]*(1/2 - 3*xi/2)*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)/le + 2*E*ucon[5]*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)*(-3*xi/2 - 1/2)/le + 6*E*sinr*ucon[0]*xi*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)/le**2 - 6*E*sinr*ucon[3]*xi*(Izz1 + (-Izz1 + Izz2)*(xi + 1)/2)/le**2

                fint[0 + c1] += Pxx*cosr*le*(-ux/le - 1/le)/2 - sinr*(-3*Mxx*xi/le + Pxx*le*(2*v0x*(3*xi**2/4 - 3/4)/le + 2*vx*(3*xi**2/4 - 3/4)/le)/2)
                fint[1 + c1] += Pxx*le*sinr*(-ux/le - 1/le)/2 + cosr*(-3*Mxx*xi/le + Pxx*le*(2*v0x*(3*xi**2/4 - 3/4)/le + 2*vx*(3*xi**2/4 - 3/4)/le)/2)
                fint[2 + c1] += Mxx*(1/2 - 3*xi/2) + Pxx*le*(2*v0x*(3*xi**2/8 - xi/4 - 1/8) + 2*vx*(3*xi**2/8 - xi/4 - 1/8))/2
                fint[0 + c2] += Pxx*cosr*le*(ux/le + 1/le)/2 - sinr*(3*Mxx*xi/le + Pxx*le*(2*v0x*(3/4 - 3*xi**2/4)/le + 2*vx*(3/4 - 3*xi**2/4)/le)/2)
                fint[1 + c2] += Pxx*le*sinr*(ux/le + 1/le)/2 + cosr*(3*Mxx*xi/le + Pxx*le*(2*v0x*(3/4 - 3*xi**2/4)/le + 2*vx*(3/4 - 3*xi**2/4)/le)/2)
                fint[2 + c2] += Mxx*(-3*xi/2 - 1/2) + Pxx*le*(2*v0x*(3*xi**2/8 + xi/4 - 1/8) + 2*vx*(3*xi**2/8 + xi/4 - 1/8))/2

    return fint


def update_M(beam, nid_pos, M, lumped=False):
    """Update a global M with a beam element

    Properties
    ----------
    beam : `.Beam2D` object
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

    Strains are calculated assuming a constant cross-section rotation as::

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
