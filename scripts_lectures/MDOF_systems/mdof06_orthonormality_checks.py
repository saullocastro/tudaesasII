import sys
# uncomment to run from the project root directory:
sys.path.append('.')

# uncomment to run from the scripts_lectures/MDOF_systems/ directory:
# sys.path.append('../..')

import numpy as np
from scipy.linalg import eigh

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF

# number of nodes along x
nx = 100

# geometry
length = 10
h = 1
w = h/10
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
# creating and assemblying beam elements
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
    update_M(elem, nid_pos, M)
    elems.append(elem)

# applying boundary conditions
bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
check = np.isclose(x, 0.)
bk[0::DOF] = check
bk[1::DOF] = check
bk[2::DOF] = check
bu = ~bk # defining unknown DOFs

# sub-matrices corresponding to unknown DOFs
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

# solving
# NOTE: extracting ALL eigenvectors
num_modes = 3
eigvals, U = eigh(a=Kuu, b=Muu, subset_by_index=[0, num_modes])
wn = np.sqrt(eigvals)

print()
print('natural frequencies [rad/s] (wn)', wn[:num_modes])
print('wn**2', wn[:num_modes]**2)
print()
for I, J in [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]:
    print('I =', I, 'J =', J,
          '\tUI . UJ %1.3f' % (U[:, I] @ U[:, J]),
          '\tUI M UJ %1.3f' % (U[:, I] @ Muu @ U[:, J]),
          '\tUI K UJ %1.3f' % (U[:, I] @ Kuu @ U[:, J]))
