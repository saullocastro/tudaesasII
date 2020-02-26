import sys
sys.path.append('..')

import numpy as np
from scipy.linalg import solve
from composites.laminate import read_isotropic

from tudaesasII.quad4r import Quad4R, update_K, DOF


def test_static_plate_quad_point_load(plot=False):
    nx = 7
    ny = 11

    # geometry
    a = 3
    b = 7
    h = 0.005 # m

    # material
    E = 200e9
    nu = 0.3

    plate = read_isotropic(thickness=h, E=E, nu=nu, calc_scf=True)

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]

    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    # creating global stiffness matrix
    K = np.zeros((DOF*nx*ny, DOF*nx*ny))

    # creating elements and populating global stiffness
    quads = []
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        pos3 = nid_pos[n3]
        pos4 = nid_pos[n4]
        r1 = ncoords[pos1]
        r2 = ncoords[pos2]
        r3 = ncoords[pos3]
        normal = np.cross(r2 - r1, r3 - r2)
        assert normal > 0 # guaranteeing that all elements have CCW positive normal
        quad = Quad4R()
        quad.n1 = n1
        quad.n2 = n2
        quad.n3 = n3
        quad.n4 = n4
        quad.scf13 = plate.scf_k13
        quad.scf23 = plate.scf_k23
        quad.h = h
        quad.ABDE = plate.ABDE
        update_K(quad, nid_pos, ncoords, K)
        quads.append(quad)

    print('elements created')


    # applying boundary conditions
    # simply supported
    bk = np.zeros(K.shape[0], dtype=bool) #array to store known DOFs
    check = np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0) | np.isclose(y, b)
    bk[2::DOF] = check

    # eliminating all u,v displacements
    bk[0::DOF] = True
    bk[1::DOF] = True

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # external force vector for point load at center
    f = np.zeros(K.shape[0])
    fmid = 1.
    # force at center node
    check = np.isclose(x, a/2) & np.isclose(y, b/2)
    f[2::DOF][check] = fmid
    assert f.sum() == fmid

    # sub-matrices corresponding to unknown DOFs
    Kuu = K[bu, :][:, bu]
    fu = f[bu]

    # solving static problem
    uu = solve(Kuu, fu)

    # vector u containing displacements for all DOFs
    u = np.zeros(K.shape[0])
    u[bu] = uu

    w = u[2::DOF].reshape(nx, ny).T

    # obtained with bfsplate2d element, nx=ny=29
    wmax_ref = 6.594931610258557e-05
    # obtained with Quad4R nx=7, ny=11
    wmax_ref = 5.752660593372991e-05
    assert np.isclose(wmax_ref, w.max(), rtol=0.02)
    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    test_static_plate_quad_point_load(plot=True)
