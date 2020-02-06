import numpy as np
from scipy.linalg import solve

from tudaesasII.beam2d import Beam2D, update_K, update_M, uv

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
    nplot = 100
    xplot = np.linspace(0, L, nplot)
    yplot = np.zeros_like(xplot)
    uinterp, vinterp = uv(beam, *u[:6], n=nplot)

    assert np.allclose(vinterp[0], [ 0.00000000e+00, -9.08870216e-05,
        -3.62319884e-04, -8.12456281e-04, -1.43945391e-03, -2.24147047e-03,
        -3.21666364e-03, -4.36319114e-03, -5.67921065e-03, -7.16287987e-03,
        -8.81235649e-03, -1.06257982e-02, -1.26013627e-02, -1.47372077e-02,
        -1.70314909e-02, -1.94823700e-02, -2.20880027e-02, -2.48465466e-02,
        -2.77561595e-02, -3.08149990e-02, -3.40212230e-02, -3.73729889e-02,
        -4.08684547e-02, -4.45057778e-02, -4.82831161e-02, -5.21986273e-02,
        -5.62504690e-02, -6.04367989e-02, -6.47557747e-02, -6.92055542e-02,
        -7.37842949e-02, -7.84901547e-02, -8.33212912e-02, -8.82758621e-02,
        -9.33520250e-02, -9.85479378e-02, -1.03861758e-01, -1.09291644e-01,
        -1.14835752e-01, -1.20492241e-01, -1.26259268e-01, -1.32134991e-01,
        -1.38117568e-01, -1.44205156e-01, -1.50395913e-01, -1.56687997e-01,
        -1.63079565e-01, -1.69568776e-01, -1.76153786e-01, -1.82832754e-01,
        -1.89603837e-01, -1.96465193e-01, -2.03414980e-01, -2.10451355e-01,
        -2.17572476e-01, -2.24776501e-01, -2.32061587e-01, -2.39425892e-01,
        -2.46867574e-01, -2.54384790e-01, -2.61975699e-01, -2.69638457e-01,
        -2.77371223e-01, -2.85172155e-01, -2.93039409e-01, -3.00971144e-01,
        -3.08965517e-01, -3.17020687e-01, -3.25134810e-01, -3.33306044e-01,
        -3.41532548e-01, -3.49812478e-01, -3.58143993e-01, -3.66525251e-01,
        -3.74954408e-01, -3.83429623e-01, -3.91949053e-01, -4.00510856e-01,
        -4.09113189e-01, -4.17754212e-01, -4.26432080e-01, -4.35144952e-01,
        -4.43890985e-01, -4.52668338e-01, -4.61475168e-01, -4.70309632e-01,
        -4.79169888e-01, -4.88054095e-01, -4.96960409e-01, -5.05886988e-01,
        -5.14831990e-01, -5.23793574e-01, -5.32769895e-01, -5.41759113e-01,
        -5.50759384e-01, -5.59768868e-01, -5.68785720e-01, -5.77808099e-01,
        -5.86834163e-01, -5.95862069e-01])

    if plot:
        # plotting
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.gca().set_aspect('equal')
        plt.plot(x, y, '-k')
        plt.plot(xplot+uinterp[0], yplot+vinterp[0], '--r')
        plt.show()

if __name__ == '__main__':
    test_beam2d_displ(plot=True)
