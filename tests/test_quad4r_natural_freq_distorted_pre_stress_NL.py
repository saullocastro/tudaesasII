import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.linalg import eigh
from composites import isotropic_plate

from tudaesasII.quad4r import (Quad4R, update_K, update_KG, update_KNL,
        calc_fint, update_M, DOF)


def test_nat_freq_plate_pre_stress_NL(plot=False, mode=0):
    nx = 9
    ny = 11

    a = 0.3
    b = 0.5

    # Material Lastrobe Lescalloy
    E = 203.e9 # Pa
    nu = 0.33

    rho = 7.83e3 # kg/m3
    h = 0.001 # m

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
    x[inner] += dx*rdm*0.4
    y[inner] += dy*rdm*0.4


    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    plate = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)

    N = DOF*nx*ny
    K = np.zeros((N, N))
    M = np.zeros((N, N))
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
        update_M(quad, nid_pos, ncoords, M)
        quads.append(quad)

    print('elements created')

    # applying boundary conditions
    # simply supported
    bk = np.zeros(N, dtype=bool) #array to store known DOFs
    edges = isclose(x, 0.) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
    bk[2::DOF][edges] = True
    u_constr = isclose(x, a/2) & (isclose(y, 0) | isclose(y, b))
    bk[0::DOF][u_constr] = True
    v_constr = isclose(y, b/2) & (isclose(x, 0) | isclose(x, a))
    bk[1::DOF][v_constr] = True

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # sub-matrices corresponding to unknown DOFs
    Kuu = K[bu, :][:, bu]
    Muu = M[bu, :][:, bu]

    # static case to calculate linear buckling load
    def calc_fext(load):
        fext = np.zeros(N)
        ftotal = load
        # at x=0
        check = (isclose(x, 0) & ~isclose(y, 0) & ~isclose(y, b))
        fext[0::DOF][check] = -ftotal/(ny - 1)
        check = ((isclose(x, 0) & isclose(y, 0))
                |(isclose(x, 0) & isclose(y, b)))
        fext[0::DOF][check] = -ftotal/(ny - 1)/2
        assert isclose(fext.sum(), -ftotal)
        # at x=a
        check = (isclose(x, a) & ~isclose(y, 0) & ~isclose(y, b))
        fext[0::DOF][check] = ftotal/(ny - 1)
        check = ((isclose(x, a) & isclose(y, 0))
                |(isclose(x, a) & isclose(y, b)))
        fext[0::DOF][check] = ftotal/(ny - 1)/2
        assert isclose(fext.sum(), 0)
        return fext

    fext = calc_fext(-1.)
    u0 = np.zeros(N)
    u0u = np.linalg.solve(Kuu, fext[bu])
    u0[bu] = u0u

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
            plt.plot([r1[0], r2[0]], [r1[1], r2[1]], 'k-', lw=1.)
            plt.plot([r2[0], r3[0]], [r2[1], r3[1]], 'k-', lw=1.)
            plt.plot([r3[0], r4[0]], [r3[1], r4[1]], 'k-', lw=1.)
            plt.plot([r4[0], r1[0]], [r4[1], r1[1]], 'k-', lw=1.)
        plt.contourf(xmesh, ymesh, u0[0::DOF].reshape(nx, ny).T, levels=10)
        plt.colorbar()
        plt.show()

    KG = np.zeros((N, N))
    for quad in quads:
        update_KG(quad, u0, nid_pos, ncoords, KG)
    KGuu = KG[bu, :][:, bu]

    # linear buckling analysis
    num_modes = 2
    linbuck_eigvals, linbuck_eigvecsu = eigh(a=KGuu, b=Kuu, subset_by_index=[0, num_modes-1])
    lambda_CR = -0.999999*1/linbuck_eigvals[0]
    linbuck_eigvecs = np.zeros((N, num_modes))
    linbuck_eigvecs[bu] = linbuck_eigvecsu

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
            plt.plot([r1[0], r2[0]], [r1[1], r2[1]], 'k-', lw=1.)
            plt.plot([r2[0], r3[0]], [r2[1], r3[1]], 'k-', lw=1.)
            plt.plot([r3[0], r4[0]], [r3[1], r4[1]], 'k-', lw=1.)
            plt.plot([r4[0], r1[0]], [r4[1], r1[1]], 'k-', lw=1.)
        plt.contourf(xmesh, ymesh, linbuck_eigvecs[2::DOF, 0].reshape(nx, ny).T, levels=10)
        plt.colorbar()
        plt.show()

    num_modes = 2
    eigvals, U = eigh(a=Kuu, b=Muu, subset_by_index=(0, num_modes-1))
    omegan = np.sqrt(eigvals)
    print('Natural frequencies [rad/s]', omegan)

    assert isclose(omegan[0], 324.76639, atol=1.)

    eigvals, U = eigh(a=Kuu+lambda_CR*KGuu, b=Muu, subset_by_index=(0, num_modes-1))
    omegan = np.sqrt(eigvals)
    print('Pre-stressed natural frequencies [rad/s]', omegan)

    assert isclose(omegan[0], 0.33, atol=0.1)

    #NOTE reaching nonlinear equilibrium pre-stress state with Newton-Raphson

    def calc_KT(u):
        KNL = np.zeros((N, N))
        KG = np.zeros((N, N))
        for quad in quads:
            update_KNL(quad, u, nid_pos, ncoords, KNL)
            update_KG(quad, u, nid_pos, ncoords, KG)
        assert np.allclose(K + KNL + KG, (K + KNL + KG).T)
        return K + KNL + KG

    u = np.zeros(K.shape[0])
    loads = np.abs(lambda_CR)*np.linspace(0.1, 0.99995, 3)
    for load in loads:
        fext = calc_fext(-load)
        print('load', -load)
        if isclose(load, 0.1*np.abs(lambda_CR)):
            KT = K
            uu = np.linalg.solve(Kuu, fext[bu])
            u[bu] = uu
        for i in range(100):
            fint = calc_fint(quads, u, nid_pos, ncoords)
            R = fint - fext
            check = np.abs(R[bu]).max()
            if check < 0.1:
                KT = calc_KT(u) #NOTE modified Newton-Raphson since KT is calculated only after each load step
                break
            duu = np.linalg.solve(KT[bu, :][:, bu], -R[bu])
            u[bu] += duu
        assert i < 99

    KTuu = calc_KT(u)[bu, :][:, bu]
    eigvals, U = eigh(a=KTuu, b=Muu, subset_by_index=(0, num_modes-1))
    omegan = np.sqrt(eigvals)
    print('NL pre-stressed natural frequencies [rad/s]', omegan)
    assert isclose(omegan[0], 1.685, atol=1.)


if __name__ == '__main__':
    test_nat_freq_plate_pre_stress_NL(plot=False, mode=0)
