import numpy as np

from .quadrature import quadrature_xi_wi

class Beam2D(object):
    __slots__ = ['n1', 'n2', 'E', 'rho', 'Izz1', 'Izz2', 'A1', 'A2',
            'nquadrature', 'interpolation']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        # Material Lastrobe Lescalloy
        self.E = 203.e9 # Pa
        self.rho = 7.83e3 # kg/m3
        self.nquadrature = 10
        self.interpolation = 'hermitian_cubic'

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
    ncoords = np.atleast_2d(ncoords)
    assert len(nid_pos) == ncoords.shape[0]
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
    r = np.arctan2(y2 - y1, x2 - x1)
    cosr = np.cos(r)
    sinr = np.sin(r)
    print('DEBUG, x1, y1, x2, y2', x1, y1, x2, y2)
    print('DEBUG, r, cosr, sinr', r, cosr, sinr)

    # positions c1, c2 in the stiffness and mass matrices
    c1 = 3*pos1
    c2 = 3*pos2
    print('DEBUG, c1, c2', c1, c2)
    quad = quadrature_xi_wi(beam.nquadrature)
    for xi, wi in zip(*quad):
        Izz = Izz1 + (Izz2 - Izz1)*(xi - (-1))/(1 - (-1))
        A = A1 + (A2 - A1)*(xi - (-1))/(1 - (-1))

        #TODO allow comparison between Hermitian cubic and Legendre polynomials

        K[0+c1, 0+c1] += wi*( E*(A*cosr**2*le**2 + 36.0*Izz*sinr**2*xi**2)/(2*le**3) )
        K[0+c1, 1+c1] += wi*( E*cosr*sinr*(A*le**2 - 36.0*Izz*xi**2)/(2*le**3) )
        K[0+c1, 2+c1] += wi*( -3.0*E*Izz*sinr*xi*(3*xi - 1)/le**2 )
        K[0+c1, 0+c2] += wi*( -E*(A*cosr**2*le**2 + 36.0*Izz*sinr**2*xi**2)/(2*le**3) )
        K[0+c1, 1+c2] += wi*( E*cosr*sinr*(-A*le**2 + 36.0*Izz*xi**2)/(2*le**3) )
        K[0+c1, 2+c2] += wi*( -3.0*E*Izz*sinr*xi*(3*xi + 1)/le**2 )
        K[1+c1, 0+c1] += wi*( E*cosr*sinr*(A*le**2 - 36.0*Izz*xi**2)/(2*le**3) )
        K[1+c1, 1+c1] += wi*( E*(A*le**2*sinr**2 + 36.0*Izz*cosr**2*xi**2)/(2*le**3) )
        K[1+c1, 2+c1] += wi*( 3.0*E*Izz*cosr*xi*(3*xi - 1)/le**2 )
        K[1+c1, 0+c2] += wi*( E*cosr*sinr*(-A*le**2 + 36.0*Izz*xi**2)/(2*le**3) )
        K[1+c1, 1+c2] += wi*( -E*(A*le**2*sinr**2 + 36.0*Izz*cosr**2*xi**2)/(2*le**3) )
        K[1+c1, 2+c2] += wi*( 3.0*E*Izz*cosr*xi*(3*xi + 1)/le**2 )
        K[2+c1, 0+c1] += wi*( -3.0*E*Izz*sinr*xi*(3*xi - 1)/le**2 )
        K[2+c1, 1+c1] += wi*( 3.0*E*Izz*cosr*xi*(3*xi - 1)/le**2 )
        K[2+c1, 2+c1] += wi*( E*Izz*(3*xi - 1)**2/(2*le) )
        K[2+c1, 0+c2] += wi*( 3.0*E*Izz*sinr*xi*(3*xi - 1)/le**2 )
        K[2+c1, 1+c2] += wi*( -3.0*E*Izz*cosr*xi*(3*xi - 1)/le**2 )
        K[2+c1, 2+c2] += wi*( E*Izz*(9*xi**2 - 1)/(2*le) )
        K[0+c2, 0+c1] += wi*( -E*(A*cosr**2*le**2 + 36.0*Izz*sinr**2*xi**2)/(2*le**3) )
        K[0+c2, 1+c1] += wi*( E*cosr*sinr*(-A*le**2 + 36.0*Izz*xi**2)/(2*le**3) )
        K[0+c2, 2+c1] += wi*( 3.0*E*Izz*sinr*xi*(3*xi - 1)/le**2 )
        K[0+c2, 0+c2] += wi*( E*(A*cosr**2*le**2 + 36.0*Izz*sinr**2*xi**2)/(2*le**3) )
        K[0+c2, 1+c2] += wi*( E*cosr*sinr*(A*le**2 - 36.0*Izz*xi**2)/(2*le**3) )
        K[0+c2, 2+c2] += wi*( 3.0*E*Izz*sinr*xi*(3*xi + 1)/le**2 )
        K[1+c2, 0+c1] += wi*( E*cosr*sinr*(-A*le**2 + 36.0*Izz*xi**2)/(2*le**3) )
        K[1+c2, 1+c1] += wi*( -E*(A*le**2*sinr**2 + 36.0*Izz*cosr**2*xi**2)/(2*le**3) )
        K[1+c2, 2+c1] += wi*( -3.0*E*Izz*cosr*xi*(3*xi - 1)/le**2 )
        K[1+c2, 0+c2] += wi*( E*cosr*sinr*(A*le**2 - 36.0*Izz*xi**2)/(2*le**3) )
        K[1+c2, 1+c2] += wi*( E*(A*le**2*sinr**2 + 36.0*Izz*cosr**2*xi**2)/(2*le**3) )
        K[1+c2, 2+c2] += wi*( -3.0*E*Izz*cosr*xi*(3*xi + 1)/le**2 )
        K[2+c2, 0+c1] += wi*( -3.0*E*Izz*sinr*xi*(3*xi + 1)/le**2 )
        K[2+c2, 1+c1] += wi*( 3.0*E*Izz*cosr*xi*(3*xi + 1)/le**2 )
        K[2+c2, 2+c1] += wi*( E*Izz*(9*xi**2 - 1)/(2*le) )
        K[2+c2, 0+c2] += wi*( 3.0*E*Izz*sinr*xi*(3*xi + 1)/le**2 )
        K[2+c2, 1+c2] += wi*( -3.0*E*Izz*cosr*xi*(3*xi + 1)/le**2 )
        K[2+c2, 2+c2] += wi*( E*Izz*(3*xi + 1)**2/(2*le) )

        M[0+c1, 0+c1] += wi*( A*le*rho*(cosr**2 + 0.25*sinr**2*(xi - 1)**2*(xi + 2)**2)*(xi - 1)**2/8 )
        M[0+c1, 1+c1] += wi*( A*cosr*le*rho*sinr*(xi - 1)**2*(-0.25*(xi - 1)**2*(xi + 2)**2 + 1)/8 )
        M[0+c1, 2+c1] += wi*( -0.015625*A*le**2*rho*sinr*(xi - 1)**4*(xi + 1)*(xi + 2) )
        M[0+c1, 0+c2] += wi*( -A*le*rho*(cosr**2 + 0.25*sinr**2*(xi - 2)*(xi - 1)*(xi + 1)*(xi + 2))*(xi - 1)*(xi + 1)/8 )
        M[0+c1, 1+c2] += wi*( A*cosr*le*rho*sinr*xi**2*(0.03125*xi**4 - 0.1875*xi**2 + 0.15625) )
        M[0+c1, 2+c2] += wi*( -0.015625*A*le**2*rho*sinr*(xi - 1)**3*(xi + 1)**2*(xi + 2) )
        M[1+c1, 0+c1] += wi*( A*cosr*le*rho*sinr*(xi - 1)**2*(-0.25*(xi - 1)**2*(xi + 2)**2 + 1)/8 )
        M[1+c1, 1+c1] += wi*( A*le*rho*(xi - 1)**2*(0.25*cosr**2*(xi - 1)**2*(xi + 2)**2 + sinr**2)/8 )
        M[1+c1, 2+c1] += wi*( 0.015625*A*cosr*le**2*rho*(xi - 1)**4*(xi + 1)*(xi + 2) )
        M[1+c1, 0+c2] += wi*( A*cosr*le*rho*sinr*xi**2*(0.03125*xi**4 - 0.1875*xi**2 + 0.15625) )
        M[1+c1, 1+c2] += wi*( -A*le*rho*(xi - 1)*(xi + 1)*(0.25*cosr**2*(xi - 2)*(xi - 1)*(xi + 1)*(xi + 2) + sinr**2)/8 )
        M[1+c1, 2+c2] += wi*( 0.015625*A*cosr*le**2*rho*(xi - 1)**3*(xi + 1)**2*(xi + 2) )
        M[2+c1, 0+c1] += wi*( -0.015625*A*le**2*rho*sinr*(xi - 1)**4*(xi + 1)*(xi + 2) )
        M[2+c1, 1+c1] += wi*( 0.015625*A*cosr*le**2*rho*(xi - 1)**4*(xi + 1)*(xi + 2) )
        M[2+c1, 2+c1] += wi*( A*le**3*rho*(xi - 1)**4*(xi + 1)**2/128 )
        M[2+c1, 0+c2] += wi*( 0.015625*A*le**2*rho*sinr*(xi - 2)*(xi - 1)**2*(xi + 1)**3 )
        M[2+c1, 1+c2] += wi*( -0.015625*A*cosr*le**2*rho*(xi - 2)*(xi - 1)**2*(xi + 1)**3 )
        M[2+c1, 2+c2] += wi*( A*le**3*rho*(xi - 1)**3*(xi + 1)**3/128 )
        M[0+c2, 0+c1] += wi*( -A*le*rho*(cosr**2 + 0.25*sinr**2*(xi - 2)*(xi - 1)*(xi + 1)*(xi + 2))*(xi - 1)*(xi + 1)/8 )
        M[0+c2, 1+c1] += wi*( A*cosr*le*rho*sinr*xi**2*(0.03125*xi**4 - 0.1875*xi**2 + 0.15625) )
        M[0+c2, 2+c1] += wi*( 0.015625*A*le**2*rho*sinr*(xi - 2)*(xi - 1)**2*(xi + 1)**3 )
        M[0+c2, 0+c2] += wi*( A*le*rho*(cosr**2 + 0.25*sinr**2*(xi - 2)**2*(xi + 1)**2)*(xi + 1)**2/8 )
        M[0+c2, 1+c2] += wi*( A*cosr*le*rho*sinr*(xi + 1)**2*(-0.25*(xi - 2)**2*(xi + 1)**2 + 1)/8 )
        M[0+c2, 2+c2] += wi*( 0.015625*A*le**2*rho*sinr*(xi - 2)*(xi - 1)*(xi + 1)**4 )
        M[1+c2, 0+c1] += wi*( A*cosr*le*rho*sinr*xi**2*(0.03125*xi**4 - 0.1875*xi**2 + 0.15625) )
        M[1+c2, 1+c1] += wi*( -A*le*rho*(xi - 1)*(xi + 1)*(0.25*cosr**2*(xi - 2)*(xi - 1)*(xi + 1)*(xi + 2) + sinr**2)/8 )
        M[1+c2, 2+c1] += wi*( -0.015625*A*cosr*le**2*rho*(xi - 2)*(xi - 1)**2*(xi + 1)**3 )
        M[1+c2, 0+c2] += wi*( A*cosr*le*rho*sinr*(xi + 1)**2*(-0.25*(xi - 2)**2*(xi + 1)**2 + 1)/8 )
        M[1+c2, 1+c2] += wi*( A*le*rho*(xi + 1)**2*(0.25*cosr**2*(xi - 2)**2*(xi + 1)**2 + sinr**2)/8 )
        M[1+c2, 2+c2] += wi*( -0.015625*A*cosr*le**2*rho*(xi - 2)*(xi - 1)*(xi + 1)**4 )
        M[2+c2, 0+c1] += wi*( -0.015625*A*le**2*rho*sinr*(xi - 1)**3*(xi + 1)**2*(xi + 2) )
        M[2+c2, 1+c1] += wi*( 0.015625*A*cosr*le**2*rho*(xi - 1)**3*(xi + 1)**2*(xi + 2) )
        M[2+c2, 2+c1] += wi*( A*le**3*rho*(xi - 1)**3*(xi + 1)**3/128 )
        M[2+c2, 0+c2] += wi*( 0.015625*A*le**2*rho*sinr*(xi - 2)*(xi - 1)*(xi + 1)**4 )
        M[2+c2, 1+c2] += wi*( -0.015625*A*cosr*le**2*rho*(xi - 2)*(xi - 1)*(xi + 1)**4 )
        M[2+c2, 2+c2] += wi*( A*le**3*rho*(xi - 1)**2*(xi + 1)**4/128 )
