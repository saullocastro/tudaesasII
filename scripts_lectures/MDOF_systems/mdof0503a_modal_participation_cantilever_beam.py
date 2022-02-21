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

mass = length*A*rho
print('mass = ', mass)

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

# applying boundary conditions for the initial static equilibrium
bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
at_clamp = np.isclose(x, 0.)
bk[0::DOF] = at_clamp
bk[1::DOF] = at_clamp
bk[2::DOF] = at_clamp

bu = ~bk # defining unknown DOFs

# partitioned matrices
Kuu = K[bu, :][:, bu]
Kuk = K[bu, :][:, bk]
Kku = K[bk, :][:, bu]
Kkk = K[bk, :][:, bk]
Muu = M[bu, :][:, bu]


u0 = np.zeros(K.shape[0])
case = 1
if case == 1:
    u0[0] = 1.
    u0[1] = 0.
elif case == 2:
    u0[0] = 0.
    u0[1] = 1.
elif case == 3:
    u0[0] = np.sqrt(2)/2
    u0[1] = np.sqrt(2)/2
else:
    raise RuntimeError('invalid case')
uk = u0[bk]

fext = np.zeros(K.shape[0])
fext[bu] += -Kuk@uk
uu = solve(Kuu, fext[bu])
u0[bu] = uu

R = u0.copy()


# solving generalized eigenvalue problem
num_modes = 60
eigvals, Uu = eigh(a=Kuu, b=Muu, subset_by_index=[0, num_modes-1])
wn = np.sqrt(eigvals)
print('Natural frequencies [rad/s] =', wn)
U = np.zeros((K.shape[0], num_modes))
U[bu] = Uu

MPFs = []
MPFs_u = []
MPFs_v = []
EMMs = []
EMMs_u = []
EMMs_v = []
for mode in range(num_modes):
    U_I = U[:, mode]
    generalized_modal_mass = U_I.T @ M @ U_I
    Lbar = U_I.T @ M @ R # modal excitation factor
    MPF = Lbar/generalized_modal_mass # modal participation factor
    EMM = Lbar**2/generalized_modal_mass # effective modal mass
    MPFs.append(MPF)
    EMMs.append(EMM)

    # obtaining MPF and EMM in each direction u,v (would be u,v,w for 3D # problems)
    U_I = U[0::DOF, mode]
    Lbar_u = U_I.T @ M[0::DOF, 0::DOF] @ R[0::DOF]
    MPF_u = Lbar_u/generalized_modal_mass
    EMM_u = Lbar_u**2/generalized_modal_mass
    MPFs_u.append(MPF_u)
    EMMs_u.append(EMM_u)

    U_I = U[1::DOF, mode]
    Lbar_v = U_I.T @ M[1::DOF, 1::DOF] @ R[1::DOF]
    MPF_v = Lbar_v/generalized_modal_mass
    EMM_v = Lbar_v**2/generalized_modal_mass
    MPFs_v.append(MPF_v)
    EMMs_v.append(EMM_v)

print('sum(MPFs)', sum(MPFs))
print('sum(MPFs_u)', sum(MPFs_u))
print('sum(MPFs_v)', sum(MPFs_v))
print('sum(EMMs)', sum(EMMs))
print('sum(EMMs_u)', sum(EMMs_u))
print('sum(EMMs_v)', sum(EMMs_v))
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
plt.show()

plt.clf()
plt.bar(np.arange(num_modes)+1, np.cumsum(EMMs), alpha=0.50)
plt.hlines([0.8*mass, 0.95*mass], xmin=0, xmax=num_modes, colors=['r', 'g'], linestyles=['--', '--'])
plt.text(x=2, y=0.8*mass, s='80% of total mass', va='bottom', ha='left')
plt.text(x=12, y=0.95*mass, s='95% of total mass', va='bottom', ha='left')
plt.title('Effective Modal Mass in Resultant Direction')
plt.xlabel('mode')
plt.ylabel('EMM cumulative sum [kg]')
plt.xlim(1-0.5, num_modes+0.5)
plt.show()

plt.clf()
plt.bar(np.arange(num_modes)+1, np.cumsum(EMMs_u), alpha=0.50)
plt.hlines([0.8*mass, 0.95*mass], xmin=0, xmax=num_modes, colors=['r', 'g'], linestyles=['--', '--'])
plt.text(x=2, y=0.8*mass, s='80% of total mass', va='bottom', ha='left')
plt.text(x=12, y=0.95*mass, s='95% of total mass', va='bottom', ha='left')
plt.title('Effective Modal Mass in u direction')
plt.xlabel('mode')
plt.ylabel('EMM cumulative sum [kg]')
plt.xlim(1-0.5, num_modes+0.5)
plt.show()

plt.clf()
plt.bar(np.arange(num_modes)+1, np.cumsum(EMMs_v), alpha=0.50)
plt.hlines([0.8*mass, 0.95*mass], xmin=0, xmax=num_modes, colors=['r', 'g'], linestyles=['--', '--'])
plt.text(x=2, y=0.8*mass, s='80% of total mass', va='bottom', ha='left')
plt.text(x=12, y=0.95*mass, s='95% of total mass', va='bottom', ha='left')
plt.title('Effective Modal Mass in v direction')
plt.xlabel('mode')
plt.ylabel('EMM cumulative sum [kg]')
plt.xlim(1-0.5, num_modes+0.5)
plt.show()

