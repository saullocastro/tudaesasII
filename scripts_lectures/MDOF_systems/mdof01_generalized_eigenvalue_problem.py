import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.linalg import eigh

from truss2d import Truss2D, update_K_M

DOF = 2
plot_mesh = False
plot_result = True

lumped = False

# number of nodes in each direction
nx = 20
ny = 4

# geometry
a = 10
b = 1
A = 0.01**2

# material properties
E = 70e9
rho = 2.6e3

# creating mesh
xtmp = np.linspace(0, a, nx)
ytmp = np.linspace(0, b, ny)
xmesh, ymesh = np.meshgrid(xtmp, ytmp)
ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T

x = ncoords[:, 0]
y = ncoords[:, 1]
nid_pos = dict(zip(np.arange(len(ncoords)), np.arange(len(ncoords))))

# triangulation to establish nodal connectivity
d = Delaunay(ncoords)
if plot_mesh:
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.triplot(ncoords[:, 0], ncoords[:, 1], d.simplices, lw=0.5)
    plt.plot(ncoords[:, 0], ncoords[:, 1], 'o', ms=2)
    plt.show()

# extracting edges out of triangulation to form the truss elements
edges = {}
for s in d.simplices:
    edges[tuple(sorted([s[0], s[1]]))] = [s[0], s[1]]
    edges[tuple(sorted([s[1], s[2]]))] = [s[1], s[2]]
    edges[tuple(sorted([s[2], s[0]]))] = [s[2], s[0]]
nAnBs = np.array([list(edge) for edge in edges.values()], dtype=int)

#NOTE using dense matrices
K = np.zeros((DOF*nx*ny, DOF*nx*ny))
M = np.zeros((DOF*nx*ny, DOF*nx*ny))

# creating truss elements
elems = []
for n1, n2 in nAnBs:
    elem = Truss2D()
    elem.n1 = n1
    elem.n2 = n2
    elem.E = E
    elem.A = A
    elem.rho = rho
    update_K_M(elem, nid_pos, ncoords, K, M, lumped=lumped)
    elems.append(elem)
if lumped:
    assert np.count_nonzero(M-np.diag(np.diagonal(M))) == 0


# applying boundary conditions
bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
check = np.isclose(x, 0.)
bk[0::DOF] = check
bk[1::DOF] = check
bu = ~bk # defining unknown DOFs

# sub-matrices corresponding to unknown DOFs
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

# solving
# NOTE: extracting ALL eigenvectors
eigvals, U = eigh(a=Kuu, b=Muu)
wn = eigvals**0.5
print('wn', wn[:5])

if plot_result:
    u = np.zeros(K.shape[0], dtype=float)
    for mode in range(10):
        u[bu] = U[:, mode]
        u1 = u[0::DOF].reshape(nx, ny).T
        u2 = u[1::DOF].reshape(nx, ny).T
        plt.clf()
        plt.title('mode %02d, $\omega_n$ %1.2f rad/s' % (mode+1, wn[mode]))
        plt.gca().set_aspect('equal')
        mag = (u1**2 + u2**2)**0.5
        levels = np.linspace(mag.min(), mag.max(), 100)
        xplot = xmesh + u1
        yplot = ymesh + u2
        plt.contourf(xplot, yplot, mag, levels=levels, cmap=cm.jet)
        coords_plot = np.vstack((xplot.T.flatten(), yplot.T.flatten())).T
        plt.triplot(coords_plot[:, 0], coords_plot[:, 1], d.simplices, lw=0.5)
        #plt.colorbar()
        plt.show()
        #plt.savefig('plot_truss_mode_%02d.png' % (mode+1), bbox_inches='tight')
