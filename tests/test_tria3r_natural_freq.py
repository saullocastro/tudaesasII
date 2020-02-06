import sys
sys.path.append('..')

import numpy as np
from scipy.spatial import Delaunay
from scipy.linalg import eigh
from composites.laminate import read_isotropic

from tudaesasII.tria3r import Tria3R, update_K, update_M, DOF


#def test_nat_freq_plate(plot=False, mode=0):
plot = False
if True:
    nx = 9
    ny = 9

    a = 0.3
    b = 0.5

    # Material Lastrobe Lescalloy
    E = 203.e9 # Pa
    nu = 0.33

    rho = 7.83e3 # kg/m3
    h = 0.001 # m

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T

    x = ncoords[:, 0]
    y = ncoords[:, 1]
    nid_pos = dict(zip(np.arange(len(ncoords)), np.arange(len(ncoords))))
    nids = np.asarray(list(nid_pos.keys()))

    # triangulation to establish nodal connectivity
    d = Delaunay(ncoords)
    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.triplot(ncoords[:, 0], ncoords[:, 1], d.simplices, lw=0.5)
        plt.plot(ncoords[:, 0], ncoords[:, 1], 'o', ms=2)
        plt.show()

    plate = read_isotropic(thickness=h, E=E, nu=nu, calc_scf=True)
    print('scf', plate.scf_k13, plate.scf_k23)

    K = np.zeros((DOF*nx*ny, DOF*nx*ny))
    M = np.zeros((DOF*nx*ny, DOF*nx*ny))
    trias = []
    for s in d.simplices:
        n1, n2, n3 = nids[s]
        n1, n2, n3 =  n2, n3, n1
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        pos3 = nid_pos[n3]
        r1 = ncoords[pos1]
        r2 = ncoords[pos2]
        r3 = ncoords[pos3]
        normal = np.cross(r2 - r1, r3 - r2)
        assert normal > 0 # guaranteeing that all elements have CCW positive normal
        tria = Tria3R()
        tria.rho = rho
        tria.n1 = n1
        tria.n2 = n2
        tria.n3 = n3
        tria.scf13 = plate.scf_k13
        tria.scf23 = plate.scf_k23
        tria.h = h
        tria.ABDE = plate.ABDE
        update_K(tria, nid_pos, ncoords, K)
        update_M(tria, nid_pos, ncoords, M)
        trias.append(tria)

    print('elements created')

    # applying boundary conditions
    # simply supported
    bk = np.zeros(K.shape[0], dtype=bool) #array to store known DOFs
    check = np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0) | np.isclose(y, b)
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
    mode = 0
    u[bu] = U[:, mode]

    # theoretical reference
    m = 1
    n = 1
    D = 2*h**3*E/(3*(1 - nu**2))
    wmn = (m**2/a**2 + n**2/b**2)*np.sqrt(D*np.pi**4/(2*rho*h))/2

    print('Theoretical omega123', wmn)
    print('Numerical omega123', omegan[0:10])

    #raise
    #if True:
        #plt.clf()
        #plt.contourf(xmesh, ymesh, u[2::DOF].reshape(xmesh.shape))
        #plt.show()


#if __name__ == '__main__':
    #test_nat_freq_plate(plot=True, mode=0)
