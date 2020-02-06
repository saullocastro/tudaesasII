import sys
sys.path.append('..')

import numpy as np
from scipy.linalg import solve

from tudaesasII.beam2d import Beam2D, update_K, update_M, exx

DOF = 3

def test_beam2d_displ(plot=False):
    # number of nodes
    n = 2
    # Material Lastrobe Lescalloy
    E = 203.e9 # Pa
    F = -7000
    rho = 7.83e3 # kg/m3

    L = 3
    x = np.linspace(0, L, n)
    y = np.zeros_like(x)

    # tapered properties
    b_root = 0.05 # m
    b_tip = b_root # m
    h_root = 0.05 # m
    h_tip = h_root # m
    A_root = h_root*b_root
    A_tip = h_tip*b_tip
    Izz_root = b_root*h_root**3/12
    Izz_tip = b_tip*h_tip**3/12

    # getting nodes
    ncoords = np.vstack((x ,y)).T
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    n1s = nids[0:-1]
    n2s = nids[1:]

    K = np.zeros((3*n, 3*n))
    M = np.zeros((3*n, 3*n))
    beams = []
    for n1, n2 in zip(n1s, n2s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        x1, y1 = ncoords[pos1]
        x2, y2 = ncoords[pos2]
        A1 = A2 = A_root
        Izz1 = Izz2 = Izz_root
        beam = Beam2D()
        beam.interpolation = 'legendre'
        beam.n1 = n1
        beam.n2 = n2
        # Material Lastrobe Lescalloy
        beam.E = E
        beam.rho = rho
        beam.A1, beam.A2 = A1, A2
        beam.Izz1, beam.Izz2 = Izz1, Izz2
        update_K(beam, nid_pos, ncoords, K)
        update_M(beam, nid_pos, M)
        beams.append(beam)

    # applying boundary conditions
    bk = np.zeros(K.shape[0], dtype=bool) #array to store known DOFs
    check = np.isclose(x, 0.) # locating node at root
    # clamping at root
    for i in range(DOF):
        bk[i::DOF] = check
    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # sub-matrices corresponding to unknown DOFs
    Kuu = K[bu, :][:, bu]
    Muu = M[bu, :][:, bu]

    # test
    f = np.zeros(K.shape[0])
    f[-2] = F
    fu = f[bu]

    # solving
    uu = solve(Kuu, fu)

    # vector u containing displacements for all DOFs
    u = np.zeros(K.shape[0], dtype=float)
    u[bu] = uu

    beam = beams[0]
    nplot = 10
    xplot = np.linspace(0, L, nplot)
    yplot = np.zeros_like(xplot)
    h = np.linspace(h_root, h_tip, nplot)
    exxbot = exx(-h/2, beam, *u[:6], n=nplot)
    exxtop = exx(+h/2, beam, *u[:6], n=nplot)
    ref = np.array([-0.00496552, -0.00441379, -0.00386207, -0.00331034, -0.00275862, -0.0022069,
  -0.00165517, -0.00110345, -0.00055172, 0.])
    assert np.allclose(exxbot[0], ref)
    assert np.allclose(exxtop[0], -ref)

    if plot:
        # plotting
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.plot(xplot, exxbot[0])
        plt.plot(xplot, exxtop[0])
        plt.show()

if __name__ == '__main__':
    test_beam2d_displ(plot=True)
