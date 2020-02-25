import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.linalg import eigh
from composites.laminate import read_isotropic

from tudaesasII.quad4r import Quad4R, update_K, update_M, DOF


plot = True
mode = 0

nx = 19
ny = 19

a = 0.3
b = 0.5

# Material Lastrobe Lescalloy
E = 203.e9 # Pa
nu = 0.33

rho = 7.83e3 # kg/m3
h = 0.01 # m

xtmp = np.linspace(0, a, nx)
ytmp = np.linspace(0, b, ny)
xmesh, ymesh = np.meshgrid(xtmp, ytmp)
ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
x = ncoords[:, 0]
y = ncoords[:, 1]

nids = 1 + np.arange(ncoords.shape[0])
nid_pos = dict(zip(nids, np.arange(len(nids))))
nids_mesh = nids.reshape(nx, ny)
n1s = nids_mesh[:-1, :-1].flatten()
n2s = nids_mesh[1:, :-1].flatten()
n3s = nids_mesh[1:, 1:].flatten()
n4s = nids_mesh[:-1, 1:].flatten()

plate = read_isotropic(thickness=h, E=E, nu=nu, calc_scf=True)

K = np.zeros((DOF*nx*ny, DOF*nx*ny))
M = np.zeros((DOF*nx*ny, DOF*nx*ny))
quads = []

for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
    pos1 = nid_pos[n1]
    pos2 = nid_pos[n2]
    pos3 = nid_pos[n3]
    pos4 = nid_pos[n4]
    r1 = ncoords[pos1]
    r2 = ncoords[pos2]
    r3 = ncoords[pos3]
    normal = np.cross(r2 - r1, r3 - r2)
    assert normal > 0 # guaranteeing that all elements have CCW positive normal
    quad = Quad4R()
    quad.rho = rho
    quad.n1 = n1
    quad.n2 = n2
    quad.n3 = n3
    quad.n4 = n4
    quad.scf13 = plate.scf_k13
    quad.scf23 = plate.scf_k23
    quad.h = h
    quad.ABDE = plate.ABDE
    update_K(quad, nid_pos, ncoords, K)
    update_M(quad, nid_pos, ncoords, M)
    quads.append(quad)

print('elements created')

# applying boundary conditions
# simply supported
bk = np.zeros(K.shape[0], dtype=bool) #array to store known DOFs
check = np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0) | np.isclose(y, b)
bk[2::DOF] = check

#eliminating all u,v displacements
bk[0::DOF] = True
bk[1::DOF] = True

bu = ~bk # same as np.logical_not, defining unknown DOFs

# sub-matrices corresponding to unknown DOFs
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

eigvals, U = eigh(a=Kuu, b=Muu)
omegan = eigvals**0.5

modes = np.asarray([[0, 1, 2], [3, 4, 5]])
fig, axes = plt.subplots(nrows=modes.shape[0], ncols=modes.shape[1])
for (i,j), mode in np.ndenumerate(modes):
    ax = axes[i, j]
    u = np.zeros(K.shape[0], dtype=float)
    u[bu] = U[:, mode]
    ax.contourf(xmesh, ymesh, u[2::DOF].reshape(xmesh.shape), cmap=cm.jet)
    ax.set_title('mode = %d\n$\omega=%1.2f rad/s$' % (mode+1, omegan[mode]))
plt.show()

