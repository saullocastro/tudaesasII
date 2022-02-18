import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, cholesky, solve

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF

case = 1

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
nu = 0.33
rho = 2.6e3

print('mass = ', length*A*rho)

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
if case == 1:
    at_tip = np.isclose(x, length)
    bk[1::DOF] += at_tip
elif case == 2:
    at_mid = np.isclose(x, length/2)
    bk[1::DOF] += at_mid
    at_tip = np.isclose(x, length)
    bk[1::DOF] += at_tip
elif case == 3:
    pass
else:
    raise RuntimeError('Invalid case number')

bu = ~bk # defining unknown DOFs

# partitioned matrices
Kuu = K[bu, :][:, bu]
Kuk = K[bu, :][:, bk]
Kku = K[bk, :][:, bu]
Kkk = K[bk, :][:, bk]
Muu = M[bu, :][:, bu]

# calculating static equilibrium
u0 = np.zeros(K.shape[0])
if case == 1:
    u0[1::DOF][at_tip] = 0.05
elif case == 2:
    u0[1::DOF][at_mid] = -0.02
    u0[1::DOF][at_tip] += 0.05
elif case == 3:
    pass

uk = u0[bk] # initial displacement used to create the initial state

fext = np.zeros(K.shape[0])
if case == 3:
    fext[1::DOF] = -1
fext[bu] += -Kuk@uk
uu = solve(Kuu, fext[bu])
fext[bk] = Kku@uu + Kkk@uk

u0[bu] = uu

plt.ioff()
plt.clf()
plt.title('perturbation')
plt.plot(ncoords[:, 0], u0[1::DOF])
plt.show()

# boundary conditions for the dynamic problem
bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
at_clamp = np.isclose(x, 0.)
bk[0::DOF] = at_clamp
bk[1::DOF] = at_clamp
bk[2::DOF] = at_clamp

bu = ~bk

# partitioned matrices
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

# solving generalized eigenvalue problem
num_modes = 30
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
    MPF = (U_I.T @ fext)/generalized_modal_mass
    Ln_I = U_I.T @ M @ np.ones(K.shape[0])
    EMM = Ln_I**2/generalized_modal_mass
    MPFs.append(MPF)
    EMMs.append(EMM)

plt.bar(np.arange(num_modes)+1, MPFs)
plt.title('Modal Participation Factor')
plt.xlabel('mode')
plt.ylabel('MPF')
plt.show()
print(sum(EMMs))
plt.bar(np.arange(num_modes)+1, EMMs)
plt.title('Effective Modal Mass')
plt.xlabel('mode')
plt.ylabel('EMM')
plt.show()


