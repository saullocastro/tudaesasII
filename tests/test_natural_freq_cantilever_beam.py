import numpy as np
from structsolve import freq

from tudaesasII.beam2D import Beam2D, update_K_M

def test_nat_freq_cantilever_beam(plot=False):
    n = 100

    L = 3 # total size of the beam along x

    # Material Lastrobe Lescalloy
    E = 203.e9 # Pa
    rho = 7.83e3 # kg/m3

    x = np.linspace(0, L, n)
    # path
    y = np.ones_like(x)
    # tapered properties
    b = 0.05 # m
    h = 0.05 # m
    A = h*b
    Izz = b*h**3/12

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
        A1 = A2 = A
        Izz1 = Izz2 = Izz
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

    eigvals, eigmodes = freq(K, M, sparse_solver=True)
    omegan = ((-eigvals)**0.5).real
    alpha123 = np.array([1.875, 4.694, 7.885])
    omega123 = alpha123**2*np.sqrt(E*Izz/(rho*A*L**4))
    print('Theoretical omega123', omega123)
    print('Numerical omega123', omegan[0:3])
    assert np.allclose(omega123, omegan[0:3], rtol=0.01)

    if plot:
        # plotting
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.plot(x, y, '-k')
        plt.plot(x+u[:, 0], y+u[:, 1], '--r')
        plt.show()

if __name__ == '__main__':
    test_nat_freq_cantilever_beam(plot=True)
