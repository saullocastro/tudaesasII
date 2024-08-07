import sys
# uncomment to run from the project root directory:
sys.path.append('.')

# uncomment to run from the scripts_lectures/MDOF_systems/ directory:
# sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import isclose
from scipy.linalg import cholesky, eigh, solve

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF


nx = 100

# geometry
length = 1
h = 0.05
w = h
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
N = DOF*nx
K = np.zeros((N, N))
M = np.zeros((N, N))

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

print('elements created')

# applying boundary conditions for static problem
# cantilever
bk = np.zeros(K.shape[0], dtype=bool) #array to store known DOFs
check = isclose(x, 0.)
bk[0::DOF] = check
bk[1::DOF] = check
bk[2::DOF] = check

u = np.zeros(K.shape[0])

bu = ~bk

Muu = M[bu, :][:, bu]
Mku = M[bk, :][:, bu]
Muk = M[bu, :][:, bk]
Mkk = M[bk, :][:, bk]

Kuu = K[bu, :][:, bu]
Kku = K[bk, :][:, bu]
Kuk = K[bu, :][:, bk]
Kkk = K[bk, :][:, bk]

# gravity acceleration
g = -9.81 #m/s^2

# acceleration vector
d2udt2 = np.zeros(N)
d2udt2[1::DOF] = g

# force due to gravity
Fg = M @ d2udt2

# force due to gravity at unknown DOFs
Fgu = Fg[bu]

# initial deflection due to gravity
uu0 = solve(Kuu, Fgu)
u0 = np.zeros(K.shape[0])
u0[bu] = uu0

# finding natural frequencies and orthonormal base
L = cholesky(M, lower=True)
Linv = np.linalg.inv(L)
Ktilde = Linv @ K @ Linv.T
p = 10
V = np.zeros((N, p))
gamma, Vu = eigh(Ktilde[bu, :][:, bu], subset_by_index=(0, p-1)) # already gives V[:, i] normalized to 1
V[bu] = Vu
omegan = gamma**0.5

print('modes calculated')

# calculating vibration modes from orthonormal base (remember U = L^(-T) V)
modes = np.zeros((N, len(gamma)))
for i in range(modes.shape[1]):
    modes[bu, i] =  (Linv.T @ V[:, i])[bu]

# performing time-domain analysis
tmax = 1
time_steps = 1000
plot_freq = 1

P = V

t = np.linspace(0, tmax, time_steps)

# assuming undamped case
zeta = 0

on = omegan
od = on*np.sqrt(1 - zeta**2)

# homogeneous solution for free damped 1DOF system using initial conditions
udot0 = np.zeros(N)

r0 = P.T @ L.T @ u0
rdot0 = P.T @ L.T @ udot0
phi = np.zeros_like(od)
check = r0 != 0
phi[check] = np.arctan(od[check]*r0[check]/(zeta*on[check]*r0[check] + rdot0[check]))
A0 = np.sqrt(r0**2 + (zeta*on/od*r0 + rdot0/od)**2)

# dynamic analysis
rpc = np.zeros((p, len(t)))

on = on[:, None]
od = od[:, None]

# convolution integral: general load as a sequence of impulse loads

def r_t(t, t1, t2, on, zeta, od, fmodaln):
    tn = (t1 + t2)/2
    dt = t2 - t1
    # damped function
    H = np.heaviside(t - tn, 1.)
    h = np.zeros((fmodaln.shape[0], t.shape[0]))
    check = t >= tn
    h[:, check] = 1/od*np.exp(-zeta*on*(t[check] - tn))*np.sin(od*(t[check] - tn))*H[check]
    return fmodaln*dt*h

# homogeneous solution
rh = A0[:, None]*np.exp(-zeta*on*t)*np.sin(od*t + phi[:, None])

F = np.zeros_like(Fg)
for t1, t2 in zip(t[:-1], t[1:]):
    tn = (t1 + t2)/2
    F[:] = Fg #NOTE keeping gravitational forces

    if tn >= 0.3 and tn <= 0.305:
        F[DOF*(nx//2)+1] += -1000

    # calculating modal forces
    fmodaln = (P.T @ Linv @ F)[:, None]

    # convolution
    rpc += r_t(t, t1, t2, on, zeta, od, fmodaln)

# superposition between homogeneous solution and forced solution
r = rh + rpc

# transforming from r-space to displacement
u = Linv.T @ P @ r

if True:
    plt.clf()
    fig = plt.gcf()
    v = u[1::DOF]
    scale = 1000
    ticks = scale*np.linspace(v.min(), v.max(), 5)
    ticklabels = [('%1.4f' % vi) for vi in np.linspace(v.min(), v.max(), 5)]
    ax = plt.gca()
    for i in range(1, len(t)):
        if i % plot_freq == 0:
            ax.clear()
            ax.set_ylim(scale*v.min(),scale*v.max())
            ax.set_title('$t = %1.3f s$' % t[i])
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.plot(x, scale*v[:, i])
            plt.pause(1e-9)
            if not plt.fignum_exists(fig.number):
                break
