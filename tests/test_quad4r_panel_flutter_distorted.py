import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
from composites.laminate import read_isotropic

from tudaesasII.quad4r import Quad4R, update_K, update_M, update_KA, DOF


def test_panel_flutter_plate(plot=False):
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
    x[inner] += dx*rdm*0.4
    y[inner] += dy*rdm*0.4


    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    plate = read_isotropic(thickness=h, E=E, nu=nu, calc_scf=True)

    K = np.zeros((DOF*nx*ny, DOF*nx*ny))
    M = np.zeros((DOF*nx*ny, DOF*nx*ny))
    KA = np.zeros((DOF*nx*ny, DOF*nx*ny))

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
        update_KA(quad, nid_pos, ncoords, KA)
        update_M(quad, nid_pos, ncoords, M)
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
    Kuu = csc_matrix(K[bu, :][:, bu])
    Muu = csc_matrix(M[bu, :][:, bu])
    KAuu = csc_matrix(KA[bu, :][:, bu])

    num_eigenvalues = 10

    def MAC(mode1, mode2):
        return (mode1@mode2)**2/((mode1@mode1)*(mode2@mode2))

    MACmatrix = np.zeros((num_eigenvalues, num_eigenvalues))
    betastar = np.linspace(0, 200, 50)
    betas = betastar*E*h**3/a**3
    omegan_vec = []
    for i, beta in enumerate(betas):
        print('analysis i', i)
        # solving generalized eigenvalue problem
        eigvals, eigvecsu = eigs(A=Kuu + beta*KAuu, M=Muu,
                k=num_eigenvalues, which='LM', sigma=-1.)
        eigvecs = np.zeros((K.shape[0], num_eigenvalues), dtype=float)
        eigvecs[bu, :] = eigvecsu
        omegan_vec.append(eigvals**0.5)

        if i == 0:
            eigvecs_ref = eigvecs

        for j in range(num_eigenvalues):
            for k in range(num_eigenvalues):
                MACmatrix[j, k] = MAC(eigvecs_ref[:, j], eigvecs[:, k])
        print(np.round(MACmatrix, 1))

        eigvecs_ref = eigvecs.copy()


    omegan_vec = np.array(omegan_vec)

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        for i in range(num_eigenvalues):
            plt.plot(betastar, omegan_vec[:, i])
        plt.show()


if __name__ == '__main__':
    test_panel_flutter_plate(plot=True)
