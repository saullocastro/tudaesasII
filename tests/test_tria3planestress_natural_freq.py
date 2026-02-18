import sys
sys.path.append('..')

import numpy as np
from scipy.spatial import Delaunay
from scipy.linalg import eigh

from tudaesasII.tria3planestress import Tria3PlaneStressIso, update_K_M, DOF



def test_tria3planestress_natural_freq():
    for lumped in [False, True]:
        # number of nodes in each direction
        nx = 40
        ny = 10

        # geometry
        L = 10
        h = 1
        w = h/10
        I = h**3*w/12

        # material properties
        E = 70e9
        nu = 0.33
        rho = 2.6e3

        # creating mesh
        xtmp = np.linspace(0, L, nx)
        ytmp = np.linspace(0, h, ny)
        xmesh, ymesh = np.meshgrid(xtmp, ytmp)
        ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T

        x = ncoords[:, 0]
        y = ncoords[:, 1]
        nid_pos = dict(zip(np.arange(len(ncoords)), np.arange(len(ncoords))))

        # triangulation to establish nodal connectivity
        d = Delaunay(ncoords)

        #NOTE using dense matrices
        K = np.zeros((DOF*nx*ny, DOF*nx*ny))
        M = np.zeros((DOF*nx*ny, DOF*nx*ny))

        elems = []
        # creating tria elements
        for s in d.simplices:
            elem = Tria3PlaneStressIso()
            elem.n1 = s[0]
            elem.n2 = s[1]
            elem.n3 = s[2]
            elem.E = E
            elem.nu = nu
            elem.h = h
            elem.rho = rho
            update_K_M(elem, nid_pos, ncoords, K, M, lumped=lumped)
            elems.append(elem)

        if lumped:
            assert np.count_nonzero(M-np.diag(np.diagonal(M))) == 0

        # applying boundary conditions
        bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
        check = np.isclose(x, 0.)
        bk[0::DOF] = check
        bk[1::DOF] = check
        bu = ~bk # defining unknown DOFs

        # sub-matrices corresponding to unknown DOFs
        Kuu = K[bu, :][:, bu]
        Muu = M[bu, :][:, bu]

        # solving
        nmodes = 3
        eigvals, U = eigh(a=Kuu, b=Muu, subset_by_index=(0, nmodes-1))
        wn = eigvals**0.5

        print(wn)
        assert np.allclose(wn,
                [56.43206469, 344.08967348, 817.09641692], rtol=0.01)

if __name__ == '__main__':
    test_tria3planestress_natural_freq()
