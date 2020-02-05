import time

import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from truss2d_sparse import Truss2D, update_K_M

DOF = 2

lumped = True

# number of nodes in each direction
nx = 100
ny = 500

# geometry
a = 10
b = 1
A = 0.01**2

# material properties
E = 70e9
rho = 2.6e3

t0 = time.clock()
print()
print('Creating mesh')
xtmp = np.linspace(0, a, nx)
ytmp = np.linspace(0, b, ny)
xmesh, ymesh = np.meshgrid(xtmp, ytmp)
ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
x = ncoords[:, 0]
y = ncoords[:, 1]
nid_pos = dict(zip(np.arange(len(ncoords)), np.arange(len(ncoords))))

print('    Number of DOFs:', len(ncoords)*2)

# triangulation to establish nodal connectivity
td = time.clock()
print('    Delaunay')
d = Delaunay(ncoords)
print('    done (%f s)' % (time.clock()-td))

# extracting edges out of triangulation to form the truss elements
edges = {}
for s in d.simplices:
    edges[tuple(sorted([s[0], s[1]]))] = [s[0], s[1]]
    edges[tuple(sorted([s[1], s[2]]))] = [s[1], s[2]]
    edges[tuple(sorted([s[2], s[0]]))] = [s[2], s[0]]
nAnBs = np.array([list(edge) for edge in edges.values()], dtype=int)
print('done (%f s)' % (time.clock()-t0))

N = DOF*nx*ny

t0 = time.clock()
# creating truss elements
print('Computing K, M')
elems = []
for n1, n2 in nAnBs:
    elem = Truss2D()
    elem.n1 = n1
    elem.n2 = n2
    elem.E = E
    elem.A = A
    elem.rho = rho
    elems.append(elem)

#K 16 per element
#M 4 per element
rowK = np.zeros(16*len(elems))
colK = np.zeros(16*len(elems))
valK = np.zeros(16*len(elems))
rowM = np.zeros(4*len(elems))
colM = np.zeros(4*len(elems))
valM = np.zeros(4*len(elems))
for i, elem in enumerate(elems):
    update_K_M(i, elem.A, elem.E, elem.rho,
            nid_pos[elem.n1], nid_pos[elem.n2],
            ncoords, rowK, colK, valK, rowM, colM, valM,
            lumped=lumped)
K = coo_matrix((valK, (rowK, colK)), shape=(N, N)).tocsc()
M = coo_matrix((valM, (rowM, colM)), shape=(N, N)).tocsc()

print('done (%f s)' % (time.clock()-t0))

t0 = time.clock()
print('Partitioning due to boundary conditions')
# applying boundary conditions
bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
check = np.isclose(x, 0.)
bk[0::DOF] = check
bk[1::DOF] = check
bu = ~bk # defining unknown DOFs
# sub-matrices corresponding to unknown DOFs
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]
print('done (%f s)' % (time.clock()-t0))

nmodes = 4

t0 = time.clock()
print('Solving symmetric eigenvalue problem')
L = Muu.sqrt()
Linv = L.power(-1)
Kuutilde = (Linv * Kuu) * Linv.T
eigvals, V = eigsh(Kuutilde, k=nmodes, which='SM')
wn = eigvals**0.5
print(wn)
print('done (%f s)' % (time.clock()-t0))



