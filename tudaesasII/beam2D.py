import numpy as np

class Beam2D(object):
    __slots__ = ['n1', 'n2', 'E', 'rho', 'Izz1', 'Izz2', 'A1', 'A2',
            'interpolation', 'le']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        # Material Lastrobe Lescalloy
        self.E = 203.e9 # Pa
        self.rho = 7.83e3 # kg/m3
        self.interpolation = 'hermitian_cubic'
        self.le = None

def update_K_M(beam, nid_pos, ncoords, K, M):
    """Update K and M according to a beam element

    Properties
    ----------
    beam : Beam object
        The beam element being added to K and M
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    K : np.array
        Global stiffness matrix
    M : np.array
        Global mass matrix
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
    r = np.arctan2(y2 - y1, x2 - x1)
    cosr = np.cos(r)
    sinr = np.sin(r)

    # positions c1, c2 in the stiffness and mass matrices
    c1 = 3*pos1
    c2 = 3*pos2

    if beam.interpolation == 'hermitian_cubic':
        K[0+c1, 0+c1] += E*cosr**2*(A1 + A2)/(2*le) + 6.0*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c1, 1+c1] += E*cosr*sinr*(A1 + A2)/(2*le) - 6.0*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c1, 2+c1] += -E*sinr*(4.0*Izz1 + 2.0*Izz2)/le**2
        K[0+c1, 0+c2] += -E*cosr**2*(A1 + A2)/(2*le) - 6.0*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c1, 1+c2] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6.0*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c1, 2+c2] += -E*sinr*(2.0*Izz1 + 4.0*Izz2)/le**2
        K[1+c1, 0+c1] += E*cosr*sinr*(A1 + A2)/(2*le) - 6.0*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c1, 1+c1] += 6.0*E*cosr**2*(Izz1 + Izz2)/le**3 + E*sinr**2*(A1 + A2)/(2*le)
        K[1+c1, 2+c1] += E*cosr*(4.0*Izz1 + 2.0*Izz2)/le**2
        K[1+c1, 0+c2] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6.0*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c1, 1+c2] += -6.0*E*cosr**2*(Izz1 + Izz2)/le**3 - E*sinr**2*(A1 + A2)/(2*le)
        K[1+c1, 2+c2] += E*cosr*(2.0*Izz1 + 4.0*Izz2)/le**2
        K[2+c1, 0+c1] += -E*sinr*(4.0*Izz1 + 2.0*Izz2)/le**2
        K[2+c1, 1+c1] += E*cosr*(4.0*Izz1 + 2.0*Izz2)/le**2
        K[2+c1, 2+c1] += E*(3*Izz1 + Izz2)/le
        K[2+c1, 0+c2] += E*sinr*(4.0*Izz1 + 2.0*Izz2)/le**2
        K[2+c1, 1+c2] += -E*cosr*(4.0*Izz1 + 2.0*Izz2)/le**2
        K[2+c1, 2+c2] += E*(Izz1 + Izz2)/le
        K[0+c2, 0+c1] += -E*cosr**2*(A1 + A2)/(2*le) - 6.0*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c2, 1+c1] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6.0*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c2, 2+c1] += E*sinr*(4.0*Izz1 + 2.0*Izz2)/le**2
        K[0+c2, 0+c2] += E*cosr**2*(A1 + A2)/(2*le) + 6.0*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c2, 1+c2] += E*cosr*sinr*(A1 + A2)/(2*le) - 6.0*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c2, 2+c2] += E*sinr*(2.0*Izz1 + 4.0*Izz2)/le**2
        K[1+c2, 0+c1] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6.0*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c2, 1+c1] += -6.0*E*cosr**2*(Izz1 + Izz2)/le**3 - E*sinr**2*(A1 + A2)/(2*le)
        K[1+c2, 2+c1] += -E*cosr*(4.0*Izz1 + 2.0*Izz2)/le**2
        K[1+c2, 0+c2] += E*cosr*sinr*(A1 + A2)/(2*le) - 6.0*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c2, 1+c2] += 6.0*E*cosr**2*(Izz1 + Izz2)/le**3 + E*sinr**2*(A1 + A2)/(2*le)
        K[1+c2, 2+c2] += -E*cosr*(2.0*Izz1 + 4.0*Izz2)/le**2
        K[2+c2, 0+c1] += -E*sinr*(2.0*Izz1 + 4.0*Izz2)/le**2
        K[2+c2, 1+c1] += E*cosr*(2.0*Izz1 + 4.0*Izz2)/le**2
        K[2+c2, 2+c1] += E*(Izz1 + Izz2)/le
        K[2+c2, 0+c2] += E*sinr*(2.0*Izz1 + 4.0*Izz2)/le**2
        K[2+c2, 1+c2] += -E*cosr*(2.0*Izz1 + 4.0*Izz2)/le**2
        K[2+c2, 2+c2] += E*(Izz1 + 3*Izz2)/le

        M[0+c1, 0+c1] += cosr**2*le*rho*(3*A1 + A2)/12 + le*rho*sinr**2*(0.285714285714286*A1 + 0.0857142857142857*A2)
        M[0+c1, 1+c1] += -cosr*le*rho*sinr*(0.285714285714286*A1 + 0.0857142857142857*A2) + cosr*le*rho*sinr*(3*A1 + A2)/12
        M[0+c1, 2+c1] += -le**2*rho*sinr*(0.0357142857142857*A1 + 0.0166666666666667*A2)
        M[0+c1, 0+c2] += cosr**2*le*rho*(A1 + A2)/12 + 0.0642857142857143*le*rho*sinr**2*(A1 + A2)
        M[0+c1, 1+c2] += 0.019047619047619*cosr*le*rho*sinr*(A1 + A2)
        M[0+c1, 2+c2] += le**2*rho*sinr*(0.0166666666666667*A1 + 0.0142857142857143*A2)
        M[1+c1, 0+c1] += -cosr*le*rho*sinr*(0.285714285714286*A1 + 0.0857142857142857*A2) + cosr*le*rho*sinr*(3*A1 + A2)/12
        M[1+c1, 1+c1] += cosr**2*le*rho*(0.285714285714286*A1 + 0.0857142857142857*A2) + le*rho*sinr**2*(3*A1 + A2)/12
        M[1+c1, 2+c1] += cosr*le**2*rho*(0.0357142857142857*A1 + 0.0166666666666667*A2)
        M[1+c1, 0+c2] += 0.019047619047619*cosr*le*rho*sinr*(A1 + A2)
        M[1+c1, 1+c2] += 0.0642857142857143*cosr**2*le*rho*(A1 + A2) + le*rho*sinr**2*(A1 + A2)/12
        M[1+c1, 2+c2] += -cosr*le**2*rho*(0.0166666666666667*A1 + 0.0142857142857143*A2)
        M[2+c1, 0+c1] += -le**2*rho*sinr*(0.0357142857142857*A1 + 0.0166666666666667*A2)
        M[2+c1, 1+c1] += cosr*le**2*rho*(0.0357142857142857*A1 + 0.0166666666666667*A2)
        M[2+c1, 2+c1] += le**3*rho*(5*A1 + 3*A2)/840
        M[2+c1, 0+c2] += -le**2*rho*sinr*(0.0142857142857143*A1 + 0.0166666666666667*A2)
        M[2+c1, 1+c2] += cosr*le**2*rho*(0.0142857142857143*A1 + 0.0166666666666667*A2)
        M[2+c1, 2+c2] += -le**3*rho*(A1 + A2)/280
        M[0+c2, 0+c1] += cosr**2*le*rho*(A1 + A2)/12 + 0.0642857142857143*le*rho*sinr**2*(A1 + A2)
        M[0+c2, 1+c1] += 0.019047619047619*cosr*le*rho*sinr*(A1 + A2)
        M[0+c2, 2+c1] += -le**2*rho*sinr*(0.0142857142857143*A1 + 0.0166666666666667*A2)
        M[0+c2, 0+c2] += cosr**2*le*rho*(A1 + 3*A2)/12 + le*rho*sinr**2*(0.0857142857142857*A1 + 0.285714285714286*A2)
        M[0+c2, 1+c2] += -cosr*le*rho*sinr*(0.0857142857142857*A1 + 0.285714285714286*A2) + cosr*le*rho*sinr*(A1 + 3*A2)/12
        M[0+c2, 2+c2] += le**2*rho*sinr*(0.0166666666666667*A1 + 0.0357142857142857*A2)
        M[1+c2, 0+c1] += 0.019047619047619*cosr*le*rho*sinr*(A1 + A2)
        M[1+c2, 1+c1] += 0.0642857142857143*cosr**2*le*rho*(A1 + A2) + le*rho*sinr**2*(A1 + A2)/12
        M[1+c2, 2+c1] += cosr*le**2*rho*(0.0142857142857143*A1 + 0.0166666666666667*A2)
        M[1+c2, 0+c2] += -cosr*le*rho*sinr*(0.0857142857142857*A1 + 0.285714285714286*A2) + cosr*le*rho*sinr*(A1 + 3*A2)/12
        M[1+c2, 1+c2] += cosr**2*le*rho*(0.0857142857142857*A1 + 0.285714285714286*A2) + le*rho*sinr**2*(A1 + 3*A2)/12
        M[1+c2, 2+c2] += -cosr*le**2*rho*(0.0166666666666667*A1 + 0.0357142857142857*A2)
        M[2+c2, 0+c1] += le**2*rho*sinr*(0.0166666666666667*A1 + 0.0142857142857143*A2)
        M[2+c2, 1+c1] += -cosr*le**2*rho*(0.0166666666666667*A1 + 0.0142857142857143*A2)
        M[2+c2, 2+c1] += -le**3*rho*(A1 + A2)/280
        M[2+c2, 0+c2] += le**2*rho*sinr*(0.0166666666666667*A1 + 0.0357142857142857*A2)
        M[2+c2, 1+c2] += -cosr*le**2*rho*(0.0166666666666667*A1 + 0.0357142857142857*A2)
        M[2+c2, 2+c2] += le**3*rho*(3*A1 + 5*A2)/840

    elif beam.interpolation == 'legendre':
        K[0+c1, 0+c1] += E*cosr**2*(A1 + A2)/(2*le) + 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c1, 1+c1] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c1, 2+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c1, 0+c2] += -E*cosr**2*(A1 + A2)/(2*le) - 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c1, 1+c2] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c1, 2+c2] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c1, 0+c1] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c1, 1+c1] += 6*E*cosr**2*(Izz1 + Izz2)/le**3 + E*sinr**2*(A1 + A2)/(2*le)
        K[1+c1, 2+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c1, 0+c2] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c1, 1+c2] += -6*E*cosr**2*(Izz1 + Izz2)/le**3 - E*sinr**2*(A1 + A2)/(2*le)
        K[1+c1, 2+c2] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c1, 0+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c1] += E*(3*Izz1 + Izz2)/le
        K[2+c1, 0+c2] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c2] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c2] += E*(Izz1 + Izz2)/le
        K[0+c2, 0+c1] += -E*cosr**2*(A1 + A2)/(2*le) - 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c2, 1+c1] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c2, 2+c1] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c2, 0+c2] += E*cosr**2*(A1 + A2)/(2*le) + 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c2, 1+c2] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c2, 2+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c2, 0+c1] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c2, 1+c1] += -6*E*cosr**2*(Izz1 + Izz2)/le**3 - E*sinr**2*(A1 + A2)/(2*le)
        K[1+c2, 2+c1] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c2, 0+c2] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c2, 1+c2] += 6*E*cosr**2*(Izz1 + Izz2)/le**3 + E*sinr**2*(A1 + A2)/(2*le)
        K[1+c2, 2+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 0+c1] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c1] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c1] += E*(Izz1 + Izz2)/le
        K[2+c2, 0+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c2] += E*(Izz1 + 3*Izz2)/le

        M[0+c1, 0+c1] += cosr**2*le*rho*(3*A1 + A2)/12 + le*rho*sinr**2*(10*A1 + 3*A2)/35
        M[0+c1, 1+c1] += cosr*le*rho*sinr*(3*A1 + A2)/12 - cosr*le*rho*sinr*(10*A1 + 3*A2)/35
        M[0+c1, 2+c1] += -le**2*rho*sinr*(15*A1 + 7*A2)/420
        M[0+c1, 0+c2] += cosr**2*le*rho*(A1 + A2)/12 + 9*le*rho*sinr**2*(A1 + A2)/140
        M[0+c1, 1+c2] += 2*cosr*le*rho*sinr*(A1 + A2)/105
        M[0+c1, 2+c2] += le**2*rho*sinr*(7*A1 + 6*A2)/420
        M[1+c1, 0+c1] += cosr*le*rho*sinr*(3*A1 + A2)/12 - cosr*le*rho*sinr*(10*A1 + 3*A2)/35
        M[1+c1, 1+c1] += cosr**2*le*rho*(10*A1 + 3*A2)/35 + le*rho*sinr**2*(3*A1 + A2)/12
        M[1+c1, 2+c1] += cosr*le**2*rho*(15*A1 + 7*A2)/420
        M[1+c1, 0+c2] += 2*cosr*le*rho*sinr*(A1 + A2)/105
        M[1+c1, 1+c2] += 9*cosr**2*le*rho*(A1 + A2)/140 + le*rho*sinr**2*(A1 + A2)/12
        M[1+c1, 2+c2] += -cosr*le**2*rho*(7*A1 + 6*A2)/420
        M[2+c1, 0+c1] += -le**2*rho*sinr*(15*A1 + 7*A2)/420
        M[2+c1, 1+c1] += cosr*le**2*rho*(15*A1 + 7*A2)/420
        M[2+c1, 2+c1] += le**3*rho*(5*A1 + 3*A2)/840
        M[2+c1, 0+c2] += -le**2*rho*sinr*(6*A1 + 7*A2)/420
        M[2+c1, 1+c2] += cosr*le**2*rho*(6*A1 + 7*A2)/420
        M[2+c1, 2+c2] += -le**3*rho*(A1 + A2)/280
        M[0+c2, 0+c1] += cosr**2*le*rho*(A1 + A2)/12 + 9*le*rho*sinr**2*(A1 + A2)/140
        M[0+c2, 1+c1] += 2*cosr*le*rho*sinr*(A1 + A2)/105
        M[0+c2, 2+c1] += -le**2*rho*sinr*(6*A1 + 7*A2)/420
        M[0+c2, 0+c2] += cosr**2*le*rho*(A1 + 3*A2)/12 + le*rho*sinr**2*(3*A1 + 10*A2)/35
        M[0+c2, 1+c2] += cosr*le*rho*sinr*(A1 + 3*A2)/12 - cosr*le*rho*sinr*(3*A1 + 10*A2)/35
        M[0+c2, 2+c2] += le**2*rho*sinr*(7*A1 + 15*A2)/420
        M[1+c2, 0+c1] += 2*cosr*le*rho*sinr*(A1 + A2)/105
        M[1+c2, 1+c1] += 9*cosr**2*le*rho*(A1 + A2)/140 + le*rho*sinr**2*(A1 + A2)/12
        M[1+c2, 2+c1] += cosr*le**2*rho*(6*A1 + 7*A2)/420
        M[1+c2, 0+c2] += cosr*le*rho*sinr*(A1 + 3*A2)/12 - cosr*le*rho*sinr*(3*A1 + 10*A2)/35
        M[1+c2, 1+c2] += cosr**2*le*rho*(3*A1 + 10*A2)/35 + le*rho*sinr**2*(A1 + 3*A2)/12
        M[1+c2, 2+c2] += -cosr*le**2*rho*(7*A1 + 15*A2)/420
        M[2+c2, 0+c1] += le**2*rho*sinr*(7*A1 + 6*A2)/420
        M[2+c2, 1+c1] += -cosr*le**2*rho*(7*A1 + 6*A2)/420
        M[2+c2, 2+c1] += -le**3*rho*(A1 + A2)/280
        M[2+c2, 0+c2] += le**2*rho*sinr*(7*A1 + 15*A2)/420
        M[2+c2, 1+c2] += -cosr*le**2*rho*(7*A1 + 15*A2)/420
        M[2+c2, 2+c2] += le**3*rho*(3*A1 + 5*A2)/840

    else:
        raise NotImplementedError('beam interpolation "%s" not implemented' % beam.interpolation)

def uv(le, u1, v1, beta1, u2, v2, beta2, n=100):
    inputs = [u1, v1, beta1, u2, v2, beta2]
    inputs = list(map(np.atleast_1d, inputs))
    maxshape = max([np.shape(i)[0] for i in inputs])
    for i in range(len(inputs)):
        if inputs[i].shape[0] == 1:
            inputs[i] = np.ones(maxshape)*inputs[i][0]
        else:
            assert inputs[i].shape[0] == maxshape
    u1, v1, beta1, u2, v2, beta2 = inputs
    xi = np.linspace(-1, +1, n)
    Nu1 = u1[:, None]*(1-xi)/2
    Nu2 = u2[:, None]*(1+xi)/2
    Nv1 = v1[:, None]*(1/2 - 3*xi/4 + 1*xi**3/4)
    Nv2 = beta1[:, None]*(le*(1/8 - 1*xi/8 - 1*xi**2/8 + 1*xi**3/8))
    Nv3 = v2[:, None]*(1/2 + 3*xi/4 - 1*xi**3/4)
    Nv4 = beta2[:, None]*(le*(-1/8 - 1*xi/8 + 1*xi**2/8 + 1*xi**3/8))
    # final shape will be (uv, n, maxshape)
    return np.array([Nu1+Nu2, Nv1+Nv2+Nv3+Nv4])

