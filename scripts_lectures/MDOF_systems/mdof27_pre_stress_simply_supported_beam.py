import sys
# uncomment to run from the project root directory:
sys.path.append('.')

# uncomment to run from the scripts_lectures/MDOF_systems/ directory:
# sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, solve

from tudaesasII.beam2d import (Beam2D, update_K, update_KG, update_KNL,
        calc_fint, update_M, DOF)

# number of nodes along x
nx = 11 #NOTE keep nx an odd number to have a node in the middle

# geometry
length = 2
h = 0.03
w = h
Izz = h**3*w/12
A = w*h

# material properties
E = 70e9
Fcy = 200e6 # yield stress
nu = 0.33
rho = 2.6e3

# creating mesh
xmesh = np.linspace(0, length, nx)
ymesh = np.zeros_like(xmesh)
ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
x = ncoords[:, 0]
y = ncoords[:, 1]
nid_pos = dict(zip(np.arange(len(ncoords)), np.arange(len(ncoords))))

#NOTE using dense matrices
K = np.zeros((DOF*nx, DOF*nx))
KG = np.zeros((DOF*nx, DOF*nx))
M = np.zeros((DOF*nx, DOF*nx))

beams = []
# creating beam elements
nids = list(nid_pos.keys())
for n1, n2 in zip(nids[:-1], nids[1:]):
    beam = Beam2D()
    beam.n1 = n1
    beam.n2 = n2
    beam.E = E
    beam.A1 = beam.A2 = A
    beam.Izz1 = beam.Izz2 = Izz
    beam.rho = rho
    beam.interpolation = 'legendre'
    update_K(beam, nid_pos, ncoords, K)
    update_M(beam, nid_pos, M, lumped=False)
    beams.append(beam)

# boundary conditions for the dynamic problem
bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
at_base = np.isclose(x, 0.)
bk[0::DOF][at_base] = True
bk[1::DOF][at_base] = True
at_tip = np.isclose(x, length)
bk[1::DOF][at_tip] = True

bu = ~bk # defining unknown DOFs

# partitioned matrices
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

# geometric stiffness matrix
KG *= 0
for beam in beams:
    update_KG(beam, 1., nid_pos, ncoords, KG)
KGuu = KG[bu, :][:, bu]

# linear buckling analysis
num_modes = 3
linbuck_eigvals, _ = eigh(a=Kuu, b=KGuu, subset_by_index=[0, num_modes-1])

PCR = linbuck_eigvals[0]
Ppreload_list = np.linspace(-0.9999*PCR, +PCR, 200)

# pre-load effect on natural frequencies

#CASE 1, assuming KT = KC0 + KG
first_omegan = []
for Ppreload in Ppreload_list:
    # solving generalized eigenvalue problem
    num_modes = 3
    eigvals, Uu = eigh(a=Kuu + Ppreload*KGuu, b=Muu, subset_by_index=[0, num_modes-1])
    omegan = np.sqrt(eigvals)
    first_omegan.append(omegan[0])

#CASE 2, finding KT with Newton-Raphson
def calc_KT(u):
    KNL = np.zeros((DOF*nx, DOF*nx))
    KG = np.zeros((DOF*nx, DOF*nx))
    for beam in beams:
        update_KNL(beam, u, nid_pos, ncoords, KNL)
        update_KG(beam, u, nid_pos, ncoords, KG)
    assert np.allclose(K + KNL + KG, (K + KNL + KG).T)
    return K + KNL + KG

first_omegan_NL = []
for Ppreload in Ppreload_list:
    u = np.zeros(K.shape[0])
    load_steps = Ppreload*np.linspace(0.1, 1., 10)
    for load in load_steps:
        fext = np.zeros(K.shape[0])
        fext[0::DOF][at_tip] = load
        if np.isclose(load, load_steps[0]):
            KT = K
            uu = np.linalg.solve(Kuu, fext[bu])
            u[bu] = uu
        for i in range(100):
            fint = calc_fint(beams, u, nid_pos, ncoords)
            R = fint - fext
            check = np.abs(R[bu]).max()
            if check < 0.01:
                KT = calc_KT(u) #NOTE modified Newton-Raphson since KT is only updated after each load step
                break
            duu = np.linalg.solve(KT[bu, :][:, bu], -R[bu])
            u[bu] += duu
        assert i < 99

    KTuu = calc_KT(u)[bu, :][:, bu]
    num_modes = 3
    eigvals, Uu = eigh(a=KTuu, b=Muu, subset_by_index=[0, num_modes-1])
    omegan = np.sqrt(eigvals)
    first_omegan_NL.append(omegan[0])

plt.clf()
plt.plot(Ppreload_list, first_omegan, 'ko--', mfc='None',
    label=r'$K_T \approx K + K_G$')
plt.plot(Ppreload_list, first_omegan_NL, 'rs-', mfc='None',
    label=r'$K_T$ from NL equilibrium')
plt.title('Pre-stress effect for a simply-supported beam')
plt.xlabel('Pre-load [N]')
plt.ylabel('First natural frequency [rad/s]')
plt.yscale('linear')
plt.legend()
plt.show()
