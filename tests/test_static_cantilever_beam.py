import numpy as np
from structsolve import solve

from tudaesasII.beam2D import Beam2D, update_K_M

def test_cantilever_beam(plot=False):
    n = 100

    L = 3 # total size of the beam along x

    # Material Lastrobe Lescalloy
    E = 203.e9 # Pa
    rho = 7.83e3 # kg/m3

    x = np.linspace(0, L, n)
    # path
    y = np.ones_like(x)
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
        A1 = A_root + (A_tip-A_root)*x1*A_tip/(L - 0)
        A2 = A_root + (A_tip-A_root)*x2*A_tip/(L - 0)
        Izz1 = Izz_root + (Izz_tip-Izz_root)*x1*Izz_tip/(L - 0)
        Izz2 = Izz_root + (Izz_tip-Izz_root)*x2*Izz_tip/(L - 0)
        beam = Beam2D()
        beam.n1 = n1
        beam.n2 = n2
        beam.E = E
        beam.rho = rho
        beam.A1, beam.A2 = A1, A2
        beam.Izz1, beam.Izz2 = Izz1, Izz2
        update_K_M(beam, nid_pos, ncoords, K, M)
        beams.append(beam)

    # applying boundary conditions
    # clamping at root
    K[0:3, :] = 0
    K[:, 0:3] = 0
    M[0:3, :] = 0
    M[:, 0:3] = 0


    # test
    Fy = 700
    f = np.zeros(K.shape[0])
    f[-2] = Fy

    # solving
    u = solve(K, f)
    u = u.reshape(n, -1)

    a = L
    deflection = Fy*a**3/(3*E*Izz_root)*(1+3*(L-a)/2*a)
    print('Theoretical deflection', deflection)
    print('Numerical deflection', u[-1, 1])
    assert np.isclose(deflection, u[-1, 1])

    if plot:
        # plotting
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.plot(x, y, '-k')
        plt.plot(x+u[:, 0], y+u[:, 1], '--r')
        plt.show()

if __name__ == '__main__':
    test_cantilever_beam(plot=True)
