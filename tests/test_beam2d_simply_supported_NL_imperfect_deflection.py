import sys
sys.path.append('..')

import numpy as np
from scipy.linalg import eigh

from tudaesasII.beam2d import (Beam2D, update_K, update_KG, update_KNL,
        calc_fint, update_M, DOF)


DOF = 3


def test_NL_imperfect_deflection():
    for interpolation in ('legendre' , 'hermitian_cubic'):
        n = 17
        length = 3 # total size of the beam along x

        # imperfection via bow imperfection
        w0 = length/100

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

        N = DOF*n
        K = np.zeros((N, N))
        KGunit = np.zeros((N, N))
        M = np.zeros((N, N))
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
            update_KG(beam, 1., None, nid_pos, ncoords, KGunit)
            update_M(beam, nid_pos, M)
            beams.append(beam)

        # applying boundary conditions
        at_middle = np.isclose(x, length/2) # locating node at the middle
        bk = np.zeros(N, dtype=bool) #array to store known DOFs
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
        assert np.isclose(linbuck_eigvals[0], 115947.38111518, rtol=0.01)
        Ppreload = -0.91*linbuck_eigvals[0]
        Ppreload = -0.5*linbuck_eigvals[0]

        def calc_KT(u, u0):
            KNL = np.zeros((N, N))
            KG = np.zeros((N, N))
            for beam in beams:
                update_KNL(beam, u, u0, nid_pos, ncoords, KNL)
                update_KG(beam, u, u0, nid_pos, ncoords, KG)
            assert np.allclose(KNL + KG, (KNL + KG).T)
            return KNL + KG

        u = np.zeros(N)
        u0 = np.zeros(N)
        u0[1::DOF] = w0*np.sin(np.pi*x/length)
        loads = np.abs(Ppreload)*np.linspace(0.1, 1., 10)
        for load in loads:
            fext = np.zeros(N)
            fext[0::DOF][at_tip] = -load
            if np.isclose(load, 0.1*np.abs(Ppreload)):
                KT = K
                uu = np.linalg.solve(Kuu, fext[bu])
                u[bu] = uu
            for i in range(100):
                KT = calc_KT(u, u0) #NOTE full Newton-Raphson since KT is updated at every iteration
                fint = calc_fint(beams, u, u0, nid_pos, ncoords)
                R = fint - fext
                check = np.abs(R[bu]).max()
                epsilon = 1e-6
                if check < epsilon:
                    # NOTE modified Newton-Raphson does not work
                    #KT = calc_KT(u, u0) #NOTE modified Newton-Raphson since KT is updated once at each load step
                    break
                duu = np.linalg.solve(KT[bu, :][:, bu], -R[bu])
                if np.any(np.isnan(duu)):
                    break
                u[bu] += duu
            assert i < 99

        print('linear buckling eigenvalues', linbuck_eigvals)
        print('maximum reached load', load)
        deflection_calc = u[1::DOF][at_middle][0]
        print('calculated deflection', deflection_calc)
        # NOTE reference deflection from the bow imperfection under axial
        # compressive load, taken from
        #      https://calcresource.com/statics-buckling-load.html
        k = np.sqrt(load/(E*Izz))
        PE = np.pi**2*E*Izz/length**2
        deflection_ref = w0*(load/(PE - load))
        print('reference deflection', deflection_ref)
        assert np.isclose(deflection_calc, deflection_ref, rtol=0.05)

if __name__ == '__main__':
    test_NL_imperfect_deflection()
