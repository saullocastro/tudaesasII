import sys
sys.path.append('..')

import numpy as np
from scipy.linalg import eigh

from tudaesasII.beam2d import (Beam2D, update_K, update_KG, update_KNL,
        update_M, DOF)


def test_NL_pre_stress_simply_supported_beam():
    for interpolation in ('legendre' , 'hermitian_cubic'):
        n = 10
        length = 3 # total size of the beam along x

        # Material Lastrobe Lescalloy
        E = 203.e9 # Pa
        rho = 7.83e3 # kg/m3

        x = np.linspace(0, length, n)
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
        KGunit = np.zeros((3*n, 3*n))
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
            beam.interpolation = interpolation
            update_K(beam, nid_pos, ncoords, K)
            update_KG(beam, 1., nid_pos, ncoords, KGunit)
            update_M(beam, nid_pos, M)
            beams.append(beam)

        # applying boundary conditions
        bk = np.zeros(K.shape[0], dtype=bool) #array to store known DOFs
        at_base = np.isclose(x, 0.) # locating node at base
        bk[0::DOF][at_base] = True
        bk[1::DOF][at_base] = True
        at_tip = np.isclose(x, length) # locating node at tip
        bk[1::DOF][at_tip] = True
        bu = ~bk # same as np.logical_not, defining unknown DOFs

        # sub-matrices corresponding to unknown DOFs
        Kuu = K[bu, :][:, bu]
        Muu = M[bu, :][:, bu]

        KGuu = KGunit[bu, :][:, bu]

        # linear buckling analysis
        num_modes = 3
        linbuck_eigvals, _ = eigh(a=Kuu, b=KGuu, subset_by_index=[0, num_modes-1])
        Ppreload = -0.9999999999*linbuck_eigvals[0]

        def calc_KT(u):
            KNL = np.zeros((3*n, 3*n))
            KG = np.zeros((3*n, 3*n))
            for beam in beams:
                update_KNL(beam, u, nid_pos, ncoords, KNL)
                update_KG(beam, u, nid_pos, ncoords, KG)
            assert np.allclose(K + KNL + KG, (K + KNL + KG).T)
            return K + KNL + KG

        u = np.zeros(K.shape[0])
        loads = np.abs(Ppreload)*np.linspace(0.1, 0.983, 10)
        for load in loads:
            fext = np.zeros(K.shape[0])
            fext[0::DOF][at_tip] = -load
            print('load', -load)
            if np.isclose(load, 0.1*np.abs(Ppreload)):
                KT = K
                uu = np.linalg.solve(Kuu, fext[bu])
                u[bu] = uu
            for i in range(100):
                R = (KT @ u) - fext
                check = np.abs(R[bu]).max()
                if check < 10.:
                    break
                duu = np.linalg.solve(KT[bu, :][:, bu], -R[bu])
                u[bu] += duu
                KT = calc_KT(u) #NOTE full Newton-Raphson since KT is calculated in every iteration
            assert i < 99

        nmodes = 3
        KT = calc_KT(u)[bu, :][:, bu]
        eigvals, U = eigh(a=KT, b=Muu, eigvals=(0, nmodes-1))
        omegan = np.sqrt(eigvals)
        print('Natural frequency [rad/s]', omegan)
        assert np.isclose(omegan[0], 1.9948, rtol=1e-3)

if __name__ == '__main__':
    test_NL_pre_stress_simply_supported_beam()
