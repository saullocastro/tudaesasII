import numpy as np

from tudaesasII.beam2D import Beam2D, update_K_M
from structsolve import solve

def test_static_curved_beam(plot=False):
    # number of nodes
    n = 24
    # Material Lastrobe Lescalloy
    E = 203.e9 # Pa
    F = 7000

    rho = 7.83e3 # kg/m3
    thetas = np.linspace(np.pi/2, 0, n)
    radius = 1.2
    x = radius*np.cos(thetas)
    y = -1.2 + radius*np.sin(thetas)
    # path
    print('y_tip', y[-1])

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
        update_K_M(beam, nid_pos, ncoords, K, M)
        beams.append(beam)

    # applying boundary conditions
    # clamping at root
    K[0:3, :] = 0
    K[:, 0:3] = 0
    M[0:3, :] = 0
    M[:, 0:3] = 0

    # test
    f = np.zeros(K.shape[0])
    f[-2] = F

    # solving
    u = solve(K, f)
    u = u.reshape(n, -1)

    print('u_tip', u[-1])
    ref_from_Abaqus = 5.738e-2, 4.064e-2
    print('Reference from Abaqus: u_tip', ref_from_Abaqus)
    assert np.allclose(u[-1, 0:2], ref_from_Abaqus, rtol=0.01)

    if plot:
        # plotting
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.gca().set_aspect('equal')
        plt.plot(x, y, '-k')
        plt.plot(x+u[:, 0], y+u[:, 1], '--r')
        plt.show()

if __name__ == '__main__':
    test_static_curved_beam(plot=True)
