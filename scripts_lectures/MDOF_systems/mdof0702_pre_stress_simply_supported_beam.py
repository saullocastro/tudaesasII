import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, cholesky, solve

from tudaesasII.beam2d import Beam2D, update_K, update_KG, update_M, DOF

# number of nodes along x
nx = 71 #NOTE keep nx an odd number to have a node in the middle

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

elems = []
# creating beam elements
nids = list(nid_pos.keys())
for n1, n2 in zip(nids[:-1], nids[1:]):
    elem = Beam2D()
    elem.n1 = n1
    elem.n2 = n2
    elem.E = E
    elem.A1 = elem.A2 = A
    elem.Izz1 = elem.Izz2 = Izz
    elem.rho = rho
    elem.interpolation = 'legendre'
    update_K(elem, nid_pos, ncoords, K)
    update_M(elem, nid_pos, M, lumped=False)
    elems.append(elem)

# calculating total mass of the system using the mass matrix
# unitary translation vector
unit_u = np.zeros(M.shape[0])
unit_u[0::DOF] = 1
mass = unit_u.T @ M @ unit_u

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
for elem in elems:
    update_KG(elem, 1., nid_pos, ncoords, KG)
KGuu = KG[bu, :][:, bu]

# linear buckling analysis
num_modes = 3
linbuck_eigvals, _ = eigh(a=Kuu, b=KGuu, subset_by_index=[0, num_modes-1])

Ppreload_list = np.linspace(-0.9999*linbuck_eigvals[0], +linbuck_eigvals[0], 200)
first_omegan = []
# pre-load effect on natural frequencies
for Ppreload in Ppreload_list:
    # solving generalized eigenvalue problem
    num_modes = 3
    eigvals, Uu = eigh(a=Kuu + Ppreload*KGuu, b=Muu, subset_by_index=[0, num_modes-1])
    omegan = np.sqrt(eigvals)
    first_omegan.append(omegan[0])
    print('Pre-load [N], Natural frequency [rad/s]', Ppreload, omegan[0])

plt.clf()
plt.plot(Ppreload_list, first_omegan, 'ko--', mfc='None')
plt.title('Pre-stress effect for a simply-supported beam')
plt.xlabel('Pre-load [N]')
plt.ylabel('First natural frequency [rad/s]')
plt.yscale('linear')
plt.show()
