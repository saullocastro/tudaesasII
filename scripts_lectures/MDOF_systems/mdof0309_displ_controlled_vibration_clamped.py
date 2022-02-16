import sys
sys.path.append(r'../..')

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.figsize'] = [10, 5]
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['font.size'] = 18
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, isclose
from scipy.linalg import cholesky, eigh

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF


nx = 100

# geometry
Lbeam = 2
h = 0.05
w = h
Izz = h**3*w/12
A = w*h

# material properties
E = 70e9
nu = 0.33
rho = 2.6e3

# creating mesh
xmesh = np.linspace(0, Lbeam, nx)
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
    update_M(elem, nid_pos, M, lumped=False)
    elems.append(elem)

print('elements created')

# applying boundary conditions for static problem
# cantilever
bk = np.zeros(K.shape[0], dtype=bool) #array to store known DOFs
check = isclose(x, 0.)
bk[0::DOF] = check
bk[1::DOF] = check
bk[2::DOF] = check

# removing all degrees-of-freedom from axial displacement
bk[0::DOF] = True

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

L = cholesky(M, lower=True)
Luu = L[bu, :][:, bu]
Linv = np.linalg.inv(L)
Linvuu = Linv[bu, :][:, bu]

Ktildeuu = Linvuu @ Kuu @ Linvuu.T

Nmodes = 12
eigenvalues, Vu = eigh(Ktildeuu, subset_by_index=(0, Nmodes-1)) # already gives V[:, i] normalized to 1

omegan = np.sqrt(eigenvalues)
print('Natural frequencies [rad/s]', omegan)

V = np.zeros((N, Nmodes))
V[bu, :] = Vu

P = V

# modal damping ratio
zeta = 0.02
# calculating damping matrix C from modal damping
Dm = np.zeros((Nmodes, Nmodes))
Dm[np.diag_indices_from(Dm)] = 2*zeta*omegan
C = L @ P @ Dm @ P.T @ L.T
Cuu = C[bu, :][:, bu]
Cuk = C[bu, :][:, bk]

on = omegan
od = on*np.sqrt(1 - zeta**2)
#NOTE adding new np.array axis to vectorize calculations
on = on[:, None]
od = od[:, None]

# base excitation
tmax = 1
dt = 1e-3
amplitude = 0.001
omegaf = 60 #[rad/s]
t = np.arange(0, tmax+dt, dt)
u = np.zeros((N, t.shape[0]))
udot = np.zeros((N, t.shape[0]))
uddot = np.zeros((N, t.shape[0]))
# prescribed displacement
u[1,:] = amplitude*np.sin(omegaf*t)
# prescribed velocity
udot[1,:] = amplitude*np.cos(omegaf*t)*omegaf
# prescribed acceleration
uddot[1,:] = -amplitude*np.sin(omegaf*t)*(omegaf)**2

# initial conditions in physical space
u0 = u[:, 0]
udot0 = udot[:, 0]
# initial conditions in modal space
r0 = P.T @ L.T @ u0
rdot0 = P.T @ L.T @ udot0

uk = u[bk]
udotk = udot[bk]
uddotk = uddot[bk]

# prescribed modal displacements
r = P.T @ L.T @ u
rdot = P.T @ L.T @ udot
rddot = P.T @ L.T @ uddot

# prescribed forces
F = np.zeros((N, t.shape[0])) #NOTE no external forces applied

# forces from prescribed displacements, velocities and accelerations
F[bu, :] += - Kuk @ uk - Cuk @ udotk - Muk @ uddotk
# modal forces
f = P.T @ Linv.T @ F

def r_t(t, t1, t2, on, zeta, od, fmodaln):
    """SDOF solution for a damped single impulse
    """
    tn = (t1 + t2)/2
    dt = t2 - t1
    H = np.heaviside(t - tn, 1.)
    h = 1/od*np.exp(-zeta*on*(t - tn))*np.sin(od*(t - tn))*H
    return fmodaln*dt*h

# homogeneous solution for a damped SDOF system
rh = np.exp(-zeta*on*t)*(r0[:, None]*np.cos(od*t) +
    (rdot0[:, None] + zeta*on*r0[:, None])*np.sin(od*t)/od)

# particular solution
# discrete approximation of Duhamel's convolution integral
rp = 0
for t1, t2, f1, f2 in zip(t[:-1], t[1:], f[:-1], f[1:]):
    rp += r_t(t, t1, t2, on, zeta, od, (f1+f2)/2)

# superposition between homogeneous and particular solutions
r = rh + rp

# transforming from modal space to physical space
u[bu] = (Linv.T @ P @ r)[bu]


# plotting animation
fig = plt.figure(figsize=(12, 5))
plt.xlabel('Beam length, [m]')
plt.ylabel('Deflection [mm]')
excitation = np.ones_like(x)*amplitude
plt.plot(x, np.zeros_like(x), 'k--')
line_excitation, = plt.plot(x[:1], u[1:2, 0], 'ro', ms=20)
line_deflection, = plt.plot(x, u[1::DOF, 0], 'b-')
plt.ylim(-0.01, 0.01)
ax = plt.gca()
lines = ax.get_lines()

def animate(i=0):
    lines[1].set_ydata(u[1:2, i])
    lines[2].set_ydata(u[1::DOF, i])
    return lines

anim =FuncAnimation(fig, animate, range(len(t)))
anim.save('mdof0309_movie.mp4', fps=20)
