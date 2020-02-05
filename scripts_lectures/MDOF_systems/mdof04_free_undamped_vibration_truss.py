import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.linalg import cholesky, eigh

from truss2d import Truss2D, update_K_M

DOF = 2

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

# solving symmetric eigenvalue problem
L = cholesky(Muu, lower=True)
Linv = np.linalg.inv(L)
Kuutilde = (Linv @ Kuu) @ Linv.T

eigvals, V = eigh(Kuutilde)
wn = eigvals**0.5


u0 = np.zeros(K.shape[0])
# linear
u0[1::DOF] = 0.5*(ncoords[:, 0]/a)
# quadratic
#u0[1::DOF] = 0.5*(ncoords[:, 0]/a)**2
u0u = u0[bu]
v0u = np.zeros_like(u0u)

plt.ioff()
plt.clf()
plt.title('perturbation')
plt.plot(ncoords[:, 0], u0[1::DOF])
plt.show()

nmodes = 50
print('wn', wn[:nmodes])

c1 = []
c2 = []
for I in range(nmodes):
    c1.append( V[:, I] @ L.T @ u0u )
    c2.append( V[:, I] @ L.T @ v0u / wn[I] )
print('c1', c1)
print('c2', c2)

def ufunc(t):
    tmp = 0
    for I in range(nmodes):
        tmp += (c1[I]*np.cos(wn[I]*t[:, None]) +
                c2[I]*np.sin(wn[I]*t[:, None]))*(Linv.T@V[:, I])
    return tmp

# to plot
num = 1000
t = np.linspace(0, 3, num)
uu = ufunc(t)
u_xt = np.zeros((t.shape[0], K.shape[0]))
u_xt[:, bu] = uu

plt.ion()
fig, axes = plt.subplots(nrows=2, figsize=(10, 5))
for s in axes[0].spines.values():
    s.set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].set_ylim(u_xt[:, 1::DOF].min(), u_xt[:, 1::DOF].max())

data = []
line = axes[1].plot([], [], '-')[0]
line2 = axes[1].plot([], [], 'ro')[0]
for i, ti in enumerate(t):
    ui = u_xt[i, :]
    u1 = ui[0::DOF].reshape(nx, ny).T
    u2 = ui[1::DOF].reshape(nx, ny).T

    axes[0].clear()
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlim(0, 1.1*a)
    axes[0].set_ylim(-2, 2)
    axes[0].set_title('t = %1.3f' % ti)
    xplot = xmesh + u1
    yplot = ymesh + u2
    coords_plot = np.vstack((xplot.T.flatten(), yplot.T.flatten())).T
    axes[0].triplot(coords_plot[:, 0], coords_plot[:, 1], d.simplices, lw=0.5)
    axes[0].scatter(xplot[-1, -1], yplot[-1, -1], c='r')

    axes[1].set_xlim(max(0, ti-0.5), ti+0.01)
    data.append([ti, u2[-1, -1]])
    line.set_data(np.asarray(data).T)
    line2.set_data(ti, u2[-1, -1])

    plt.pause(1e-15)

    if not plt.fignum_exists(fig.number):
        fig.savefig('plot_subplots.png', bbox_inches='tight')
        break

# frequency analysis
num = 100000
t = np.linspace(0, 10, num)
uu = ufunc(t)
u = np.zeros((t.shape[0], K.shape[0]))
u[:, bu] = uu

# get position of node corrsponding to the top right
pos = np.where((ncoords[:, 0] == xmesh[-1, -1]) & (ncoords[:, 1] == ymesh[-1, -1]))[0][0]
u1 = u[:, pos*DOF+1]

plt.ioff()
plt.clf()
plt.title('response for one DOF')
plt.plot(t, u1)
plt.xlim(0, 0.5)
plt.ylabel('$u_y$ top right')
plt.xlabel('$t$')
plt.show()


# Fast Fourier Transform to get the frequencies
uf = np.fft.fft(u1)
dt = t[1]-t[0]
xf = np.linspace(0.0, 2*np.pi*1.0/(2.0*dt), num//2)
yf = 2/num*np.abs(uf[0:num//2])

plt.xlabel('$rad/s$')
plt.plot(xf, yf)
plt.xlim(0, wn[4]+50)
plt.show()

