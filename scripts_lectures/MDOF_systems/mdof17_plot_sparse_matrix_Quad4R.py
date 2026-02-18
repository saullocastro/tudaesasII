import sys
# uncomment to run from the project root directory:
sys.path.append('.')

# uncomment to run from the scripts_lectures/MDOF_systems/ directory:
# sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np
from composites import isotropic_plate

from tudaesasII.quad4r import Quad4R, update_K, update_M, DOF
from tudaesasII.utils import plot_sparse_matrix


nx = 5
ny = 18

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

plate = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)

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
    quad.A = plate.A
    quad.B = plate.B
    quad.D = plate.D
    quad.Atrans = plate.Atrans
    update_K(quad, nid_pos, ncoords, K)
    update_M(quad, nid_pos, ncoords, M, lumped=False)
    quads.append(quad)

ax = plot_sparse_matrix(M)
plt.show()