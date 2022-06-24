import sys
sys.path.append('../..')

import numpy as np
from scipy.linalg import eigh, cholesky

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

p = 4

# solving generalized eigenvalue problem
# NOTE: extracting only p eigenvectors
eigvals_g, U = eigh(a=Kuu, b=Muu, subset_by_index=(0, p-1))
wn_g = np.sqrt(eigvals_g)

# solving symmetric eigenvalue problem
L = cholesky(Muu, lower=True)
Linv = np.linalg.inv(L)
Kuutilde = (Linv @ Kuu) @ Linv.T

#NOTE asserting that Kuutilde is symmetric
assert np.allclose(Kuutilde, Kuutilde.T)

eigvals, V = eigh(Kuutilde, subset_by_index=(0, p-1))
wn = eigvals**0.5

print('omegan**2', wn**2)

P = V

print('P.T @ Kuutilde @ P')
print(np.round(P.T @ Kuutilde @ P, 2))
print()
print('P.T @ P')
print(np.round(P.T @ P, 2))

