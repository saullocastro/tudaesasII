import sys
# uncomment to run from the project root directory:
sys.path.append('.')

# uncomment to run from the scripts_lectures/MDOF_systems/ directory:
# sys.path.append('../..')

import numpy as np
from scipy.linalg import eigh, cholesky

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF


lumped = False
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

N = DOF*nx

#NOTE using dense matrices
K = np.zeros((N, N))
M = np.zeros((N, N))

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
    update_M(elem, nid_pos, M, lumped=lumped)
    elems.append(elem)

# applying boundary conditions
bk = np.zeros(N, dtype=bool) # defining known DOFs
check = np.isclose(x, 0.)
bk[0::DOF] = check
bk[1::DOF] = check
bk[2::DOF] = check
bu = ~bk # defining unknown DOFs

# sub-matrices corresponding to unknown DOFs
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

# solving generalized eigenvalue problem
num_modes = 3
U = np.zeros((N, num_modes))
eigvals_g, Uu = eigh(a=Kuu, b=Muu, subset_by_index=[0, num_modes-1])
U[bu] = Uu
omegan_g = np.sqrt(eigvals_g)

# solving symmetric eigenvalue problem
L = cholesky(M, lower=True)
Linv = np.linalg.inv(L)
Ktilde = (Linv @ K) @ Linv.T

if lumped:
    L_lumped = np.sqrt(M)
    Linv_lumped = np.zeros_like(L_lumped)
    Linv_lumped[np.diag_indices_from(Linv_lumped)] = 1/L_lumped.diagonal()

    assert np.allclose(L_lumped, L)
    assert np.allclose(Linv_lumped, Linv)

#NOTE asserting that Kuutilde is symmetric
assert np.allclose(Ktilde, Ktilde.T)

V = np.zeros((N, num_modes))
eigvals_s, Vu = eigh(Ktilde[bu, :][:, bu], subset_by_index=[0, num_modes-1])
V[bu] = Vu
omegan_s = eigvals_s**0.5

print('eigenvalues (wn_generalized**2)', omegan_g[:num_modes]**2)
print('eigenvalues (wn_symmetric**2)  ', omegan_s[:num_modes]**2)
print()
print('checks for U')
for I, J in [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]:
    print('I =', I, 'J =', J,
          '\tUI . UJ % 1.4f' % (U[:, I] @ U[:, J]),
          '\tUI M UJ % 1.4f' % (U[:, I] @ M @ U[:, J]),
          '\tUI K UJ % 1.4f' % (U[:, I] @ K @ U[:, J]))
print()
print('checks for V')
for I, J in [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]:
    print('I =', I, 'J =', J,
          '\tVI . VJ % 1.4f' % (V[:, I] @ V[:, J]),
          '\t\tVI K_tilde VJ % 1.4f' % (V[:, I] @ Ktilde @ V[:, J]))
