import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.linalg import eigh
from composites import isotropic_plate

from tudaesasII.quad4r import Quad4R, update_K, update_M, DOF


def test_nat_freq_plate(plot=False, mode=0):
    nx = 11
    ny = 13

    a = 0.3
    b = 0.5

    # Material Lastrobe Lescalloy
    E = 203.e9 # Pa
    nu = 0.33

    rho = 7.83e3 # kg/m3
    h = 0.01 # m

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)

    dx = xtmp[1] - xtmp[0]
    dy = ytmp[1] - ytmp[0]

    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T

    x = ncoords[:, 0]
    y = ncoords[:, 1]

    inner = np.logical_not(isclose(x, 0) | isclose(x, a) | isclose(y, 0) | isclose(y, b))
    np.random.seed(20)
    rdm = (-1 + 2*np.random.rand(x[inner].shape[0]))
    np.random.seed(20)
    rdm = (-1 + 2*np.random.rand(y[inner].shape[0]))
    x[inner] += dx*rdm*0.45
    y[inner] += dy*rdm*0.45


    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    plate = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)

    K = np.zeros((DOF*nx*ny, DOF*nx*ny))
    M = np.zeros((DOF*nx*ny, DOF*nx*ny))
    quads = []

    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        pos3 = nid_pos[n3]
        pos4 = nid_pos[n4]
        r1 = ncoords[pos1]
        r2 = ncoords[pos2]
        r3 = ncoords[pos3]
        r4 = ncoords[pos4]
        normal = np.cross(r2 - r1, r3 - r2)
        assert normal > 0 # guaranteeing that all elements have CCW positive normal
        quad = Quad4R()
        quad.rho = rho
        quad.n1 = n1
        quad.n2 = n2
        quad.n3 = n3
        quad.n4 = n4
        quad.scf13 = plate.scf_k13
        quad.scf23 = plate.scf_k23
        quad.h = h
        quad.ABDE = plate.ABDE
        update_K(quad, nid_pos, ncoords, K)
        update_M(quad, nid_pos, ncoords, M, lumped=True)
        quads.append(quad)

    print('elements created')

    # applying boundary conditions
    # simply supported
    bk = np.zeros(K.shape[0], dtype=bool) #array to store known DOFs
    check = isclose(x, 0.) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
    bk[2::DOF] = check

    #eliminating all u,v displacements
    bk[0::DOF] = True
    bk[1::DOF] = True

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # sub-matrices corresponding to unknown DOFs
    Kuu = K[bu, :][:, bu]
    Muu = M[bu, :][:, bu]

    eigvals, U = eigh(a=Kuu, b=Muu)
    omegan = eigvals**0.5

    # vector u containing displacements for all DOFs
    u = np.zeros(K.shape[0], dtype=float)
    u[bu] = U[:, mode]

    # theoretical reference
    m = 1
    n = 1
    D = 2*h**3*E/(3*(1 - nu**2))
    wmn = (m**2/a**2 + n**2/b**2)*np.sqrt(D*np.pi**4/(2*rho*h))/2

    print('Theoretical omega123', wmn)
    wmn_ref = 2500
    print('Numerical omega123', omegan[0:10])

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.clf()
        for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
            pos1 = nid_pos[n1]
            pos2 = nid_pos[n2]
            pos3 = nid_pos[n3]
            pos4 = nid_pos[n4]
            r1 = ncoords[pos1]
            r2 = ncoords[pos2]
            r3 = ncoords[pos3]
            r4 = ncoords[pos4]
            plt.plot([r1[0], r2[0]], [r1[1], r2[1]], 'k-')
            plt.plot([r2[0], r3[0]], [r2[1], r3[1]], 'k-')
            plt.plot([r3[0], r4[0]], [r3[1], r4[1]], 'k-')
            plt.plot([r4[0], r1[0]], [r4[1], r1[1]], 'k-')
        plt.contourf(xmesh, ymesh, u[2::DOF].reshape(nx, ny).T)
        plt.show()

    assert np.isclose(wmn_ref, omegan[0], rtol=0.05)



if __name__ == '__main__':
    test_nat_freq_plate(plot=True, mode=0)
