import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, cholesky, solve

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF

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


# creating the perturbation for different cases
u0 = np.zeros(K.shape[0])
case = 1
if case == 1:
    u0[0] = 1.
    u0[1] = 0.
    u0[-3] = 1.
    u0[-2] = 0.
elif case == 2:
    u0[0] = 0.
    u0[1] = 1.
    u0[-3] = 0.
    u0[-2] = 1.
elif case == 3:
    u0[0] = 1.
    u0[1] = 1.
    u0[-3] = 1.
    u0[-2] = 1.
else:
    raise RuntimeError('invalid case')
u0 /= (np.linalg.norm(u0.reshape(-1, 3), axis=1)).max()

uk = u0[bk]

fext = np.zeros(K.shape[0])
fext[bu] += -Kuk@uk
uu = solve(Kuu, fext[bu])
u0[bu] = uu

# influence vector
R = u0.copy()
R[np.isclose(R, 0)] = 0

# solving generalized eigenvalue problem
num_modes = 60
eigvals, Uu = eigh(a=Kuu, b=Muu, subset_by_index=[0, num_modes-1])
wn = np.sqrt(eigvals)
print('Natural frequencies [rad/s] =', wn)
U = np.zeros((K.shape[0], num_modes))
U[bu] = Uu

MPFs = []
EMMs = []
for mode in range(num_modes):
    U_I = U[:, mode]
    generalized_modal_mass = U_I.T @ M @ U_I
    Lbar = U_I.T @ M @ R # modal excitation factor
    MPF = Lbar/generalized_modal_mass # modal participation factor
    EMM = Lbar**2/generalized_modal_mass # effective modal mass
    MPFs.append(MPF)
    EMMs.append(EMM)

print('sum(MPFs)', sum(MPFs))
print('sum(EMMs)', sum(EMMs))
plt.clf()
plt.bar(np.arange(num_modes)+1, MPFs)
plt.title('Modal Participation Factor')
plt.xlabel('mode')
plt.ylabel('MPF')
plt.xlim(1-0.5, num_modes+0.5)
plt.show()
plt.clf()
plt.bar(np.arange(num_modes)+1, EMMs)
plt.title('Effective Modal Mass')
plt.xlabel('mode')
plt.ylabel('EMM [kg]')
plt.xlim(1-0.5, num_modes+0.5)
plt.hlines([0.01*mass, 0.02*mass], xmin=0, xmax=num_modes, colors=['r', 'g'], linestyles=['--', '--'])
plt.text(x=20, y=0.01*mass, s='1% of total mass', va='bottom', ha='left')
plt.text(x=30, y=0.02*mass, s='2% of total mass', va='bottom', ha='left')
plt.ylim(0, 0.2*mass)
plt.show()

plt.clf()
plt.bar(np.arange(num_modes)+1, np.cumsum(EMMs), alpha=0.50)
plt.hlines([0.8*mass, 0.95*mass], xmin=0, xmax=num_modes, colors=['r', 'g'], linestyles=['--', '--'])
plt.text(x=2, y=0.8*mass, s='80% of total mass', va='bottom', ha='left')
plt.text(x=12, y=0.95*mass, s='95% of total mass', va='bottom', ha='left')
plt.title('Cumulative Effective Modal Mass')
plt.xlabel('mode')
plt.ylabel('EMM cumulative sum [kg]')
plt.xlim(1-0.5, num_modes+0.5)
plt.show()
