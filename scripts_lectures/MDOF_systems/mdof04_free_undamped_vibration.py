import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, cholesky

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF

# number of nodes along x
nx = 10

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

# solving symmetric eigenvalue problem
L = cholesky(Muu, lower=True)
Linv = np.linalg.inv(L)
Kuutilde = (Linv @ Kuu) @ Linv.T

p = 10
eigvals, V = eigh(Kuutilde, subset_by_index=(0, p-1))
wn = eigvals**0.5

u0 = np.zeros(K.shape[0])
# linear
u0[1::DOF] = 0.5*(ncoords[:, 0]/length)
# quadratic
#u0[1::DOF] = 0.5*(ncoords[:, 0]/length)**2
u0u = u0[bu]
v0u = np.zeros_like(u0u)

plt.ioff()
plt.clf()
plt.title('perturbation')
plt.plot(ncoords[:, 0], u0[1::DOF])
plt.show()


c1 = []
c2 = []
for I in range(p):
    c1.append( V[:, I] @ L.T @ u0u )
    c2.append( V[:, I] @ L.T @ v0u / wn[I] )

def ufunc(t):
    tmp = 0
    for I in range(p):
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
    u1 = ui[0::DOF]
    u2 = ui[1::DOF]

    axes[0].clear()
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlim(0, 1.1*length)
    axes[0].set_ylim(-2, 2)
    axes[0].set_title('t = %1.3f' % ti)
    xplot = xmesh + u1
    yplot = ymesh + u2
    axes[0].plot(xplot, yplot, 'k')
    axes[0].scatter(xplot[-1], yplot[-1], c='r')

    axes[1].set_xlim(max(0, ti-0.5), ti+0.01)
    data.append([ti, u2[-1]])
    line.set_data(np.asarray(data).T)
    line2.set_data(ti, u2[-1])

    plt.pause(1e-15)

    if not plt.fignum_exists(fig.number):
        fig.savefig('plot_subplots.png', bbox_inches='tight')
        break

# frequency analysis
num = 100000
t = np.linspace(0, 1, num)
uu = ufunc(t)
u = np.zeros((t.shape[0], K.shape[0]))
u[:, bu] = uu

# get position of node corrsponding to the top right
pos = np.where((ncoords[:, 0] == xmesh[-1]) & (ncoords[:, 1] == ymesh[-1]))[0][0]
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
plt.xlim(0, wn[p-1]+50)
plt.show()

