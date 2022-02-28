import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, cholesky, solve

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF

# number of nodes along x
nx = 101 #NOTE keep nx an odd number to have a node in the middle

# geometry
length = 2
h = 0.03
w = h
Izz = h**3*w/12
A = w*h

# material properties
E = 70e9
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
M = np.zeros((DOF*nx, DOF*nx))
V = np.zeros((DOF*nx, DOF*nx))

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
    elem.rho = 1
    update_M(elem, nid_pos, V, lumped=False)
    elem.rho = rho
    elems.append(elem)

# calculating total mass of the system using the mass matrix
# unitary translation vector
unit_u = np.zeros(M.shape[0])
unit_u[0::DOF] = 1
mass = unit_u.T @ M @ unit_u
volume = unit_u.T @ V @ unit_u

# boundary conditions for the dynamic problem
bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
at_base = np.isclose(x, 0.)
bk[0::DOF] = at_base
bk[1::DOF] = at_base
at_tip = np.isclose(x, length)
bk[1::DOF] += at_tip

bu = ~bk # defining unknown DOFs

# partitioned matrices
Kuu = K[bu, :][:, bu]
Kuk = K[bu, :][:, bk]
Kku = K[bk, :][:, bu]
Kkk = K[bk, :][:, bk]
Muu = M[bu, :][:, bu]


# solving generalized eigenvalue problem
num_modes = 70
eigvals, Uu = eigh(a=Kuu, b=Muu, subset_by_index=[0, num_modes-1])
wn = np.sqrt(eigvals)
print('Natural frequencies [rad/s] =', wn)
U = np.zeros((K.shape[0], num_modes))
U[bu] = Uu

normalization = 1

ms = []
for mode in range(num_modes):
    U_I = U[:, mode]
    if normalization == 1:
        pass
        generalized_modal_mass = U_I.T @ M @ U_I
    elif normalization == 2:
        L_I2 = 1/volume * (U_I.T @ V @ U_I)
        print('modal length', L_I2)
        U_I /= np.sqrt(L_I2)
        assert np.isclose(1/volume * (U_I.T @ V @ U_I), 1.)
        generalized_modal_mass = U_I.T @ M @ U_I
        print('generalized modal mass', generalized_modal_mass)
        M_I = generalized_modal_mass / (1/volume * (U_I.T @ V @ U_I))
        print('M_I', M_I)
    elif normalization == 3:
        U_I[2::DOF] = 0 #NOTE using only translational DOFs
        L_I2 = (U_I.T @ U_I)/nx
        print('modal length', L_I2)
        U_I /= np.sqrt(L_I2)
        print('(U_I.T @ U_I)/nx', (U_I.T @ U_I)/nx)
        assert np.isclose((U_I.T @ U_I)/nx, 1.)
        generalized_modal_mass = U_I.T @ M @ U_I
        print('generalized modal mass', generalized_modal_mass)
        M_I = generalized_modal_mass / ((U_I.T @ U_I)/nx)
        print('M_I', M_I)
    elif normalization == 4:
        max_translation_DOF = max(np.abs(U_I[0::DOF]).max(), np.abs(U_I[1::DOF]).max())
        U_I[0::DOF] /= max_translation_DOF
        U_I[1::DOF] /= max_translation_DOF
        generalized_modal_mass = U_I.T @ M @ U_I
    ms.append(generalized_modal_mass)

plt.clf()
plt.bar(np.arange(num_modes)+1, ms, alpha=0.50)
if normalization == 2:
    plt.hlines([mass], xmin=0, xmax=num_modes, colors=['r', 'g'], linestyles=['--', '--'])
    plt.text(x=12, y=mass, s='total mass', va='bottom', ha='left')
if normalization == 3:
    plt.hlines([mass], xmin=0, xmax=num_modes, colors=['r', 'g'], linestyles=['--', '--'])
    plt.text(x=12, y=1.05*mass, s='total mass', va='bottom', ha='left')
plt.title('Modal mass for each mode')
plt.xlabel('mode')
plt.ylabel('Modal mass [kg]')
plt.xlim(1-0.5, num_modes+0.5)
plt.show()
