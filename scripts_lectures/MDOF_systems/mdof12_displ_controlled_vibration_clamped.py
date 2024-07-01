import sys
# uncomment to run from the project root directory:
sys.path.append('.')

# uncomment to run from the scripts_lectures/MDOF_systems/ directory:
# sys.path.append('../..')

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
length = 2
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
    elem.interpolation = 'hermitian_cubic'
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
Linv = np.linalg.inv(L)

Ktilde = Linv @ K @ Linv.T

num_modes = 4
eigenvalues, Vu = eigh(Ktilde[bu, :][:, bu], subset_by_index=(0, num_modes-1)) # already gives V[:, i] normalized to 1

omegan = np.sqrt(eigenvalues)
print('Natural frequencies [rad/s]', omegan)

V = np.zeros((N, num_modes))
V[bu, :] = Vu

P = V

# modal damping ratio
zeta = 0.02
# calculating damping matrix C from modal damping
Dm = np.zeros((num_modes, num_modes))
Dm[np.diag_indices_from(Dm)] = 2*zeta*omegan
C = L @ P @ Dm @ P.T @ L.T
Cuu = C[bu, :][:, bu]
Cuk = C[bu, :][:, bk]

on = omegan
od = on*np.sqrt(1 - zeta**2)
print('Damped natural frequencies [rad/s]', od)

#NOTE adding new np.array axis to vectorize calculations
on = on[:, None]
od = od[:, None]

# base excitation
tmax = 3.
dt = 1e-3
amplitude = 0.001
omegaf = omegan[0] #[rad/s]
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

# prescribed modal displacements
r = P.T @ L.T @ u
rdot = P.T @ L.T @ udot
rddot = P.T @ L.T @ uddot

# prescribed forces
F = np.zeros((N, t.shape[0]))
#NOTE no external forces applied

# forces from prescribed displacements, velocities and accelerations
uk = u[bk]
udotk = udot[bk]
uddotk = uddot[bk]
F[bu, :] += - Kuk @ uk - Cuk @ udotk - Muk @ uddotk

# modal forces
f = P.T @ Linv @ F

def r_t(t, t1, t2, on, zeta, od, fmodaln):
    """SDOF solution for a damped single impulse
    """
    tn = (t1 + t2)/2
    dt = t2 - t1
    H = np.heaviside(t - tn, 1.)
    h = 1/od*np.exp(-zeta*on*(t - tn))*np.sin(od*(t - tn))*H
    return fmodaln[:, None]*dt*h

# homogeneous solution for a damped SDOF system
rh = np.exp(-zeta*on*t)*(r0[:, None]*np.cos(od*t) +
    (rdot0[:, None] + zeta*on*r0[:, None])*np.sin(od*t)/od)

# particular solution
# discrete approximation of Duhamel's convolution integral
rp = 0
for i in range(len(t)-1):
    t1 = t[i]
    t2 = t[i+1]
    f1 = f[:, i]
    f2 = f[:, i+1]
    fmodaln = (f1 + f2)/2
    rp += r_t(t, t1, t2, on, zeta, od, fmodaln)

# superposition between homogeneous and particular solutions
r = rh + rp

# transforming from modal space to physical space
u[bu] = (Linv.T @ P @ r)[bu]

#check = isclose(x, length)
#plt.plot(t, 1000*u[1::DOF, :][check][0])
#plt.show()
#raise

print('plotting...')
# plotting animation
tplot = t[::10]
uplot = u[:, ::10]

fig = plt.figure(figsize=(12, 6))
plt.xlabel('Beam length, [m]')
plt.ylabel('Deflection [mm]')
plt.plot(x, np.zeros_like(x), 'k--')
line_excitation, = plt.plot(x[:1], uplot[1:2, 0], 'ro', ms=20)
line_deflection, = plt.plot(x, uplot[1::DOF, 0], 'b-')
ax = plt.gca()
plt.ylim(-40, 40)
#plt.title('time = %1.2s')
lines = ax.get_lines()

m2mm = 1000
def animate(i=0):
    plt.title('$\\omega_f=$%1.2f [rad/s]\n$t = $%1.2f [s]' % (omegaf, tplot[i]))
    lines[1].set_ydata(uplot[1:2, i]*m2mm)
    u_base = uplot[1, i]*m2mm
    y_beam = u_base + uplot[1::DOF, i]*m2mm
    y_beam[0] = u_base
    lines[2].set_ydata(y_beam)
    return lines

anim = FuncAnimation(fig, animate, range(len(tplot)))
anim.save('mdof12_omegan1_zeta002_movie.mp4', fps=25)
