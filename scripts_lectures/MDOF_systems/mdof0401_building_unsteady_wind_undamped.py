import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, eigh

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF


m2mm = 1000
# number of nodes
n = 300

# Material steel
E = 200e9 # Pa
rho = 7e3 # kg/m^3

height = 50

y = np.linspace(0, height, n)
x = np.zeros_like(y)

# tapered properties
side_root = 20 # m
thick_root = 0.4 # m
side_tip = 20 # m
thick_tip = 0.1 # m
side = np.linspace(side_root, side_tip, n)
thick = np.linspace(thick_root, thick_tip, n)
#NOTE assuming hollow-square cross section
A = 4*side*thick - thick*thick*4
Izz = 1/12*side*side**3 - 1/12*(side-thick)*(side-thick)**3

# getting nodes
ncoords = np.vstack((x ,y)).T
nids = np.arange(ncoords.shape[0])
nid_pos = dict(zip(nids, np.arange(len(nids))))

n1s = nids[0:-1]
n2s = nids[1:]

K = np.zeros((DOF*n, DOF*n))
M = np.zeros((DOF*n, DOF*n))
elements = []
for n1, n2 in zip(n1s, n2s):
    pos1 = nid_pos[n1]
    pos2 = nid_pos[n2]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    A1 = A[nid_pos[n1]]
    A2 = A[nid_pos[n2]]
    Izz1 = Izz[nid_pos[n1]]
    Izz2 = Izz[nid_pos[n2]]
    beam = Beam2D()
    beam.n1 = n1
    beam.n2 = n2
    beam.E = E
    beam.rho = rho
    beam.A1, beam.A2 = A1, A2
    beam.Izz1, beam.Izz2 = Izz1, Izz2
    update_K(beam, nid_pos, ncoords, K)
    update_M(beam, nid_pos, M, lumped=True)
    elements.append(beam)

I = np.ones_like(M)

# applying boundary conditions
# uroot = 0
# vroot = unknown
# betaroot = 0
# utip = unknown
# vtip = prescribed displacement
# betatip = unknown
known_ind = [0, 2, (K.shape[0]-1)-1]
bu = np.logical_not(np.in1d(np.arange(M.shape[0]), known_ind))
bk = np.in1d(np.arange(M.shape[0]), known_ind)

Muu = M[bu, :][:, bu]
Mku = M[bk, :][:, bu]
Muk = M[bu, :][:, bk]
Mkk = M[bk, :][:, bk]

Kuu = K[bu, :][:, bu]
Kku = K[bk, :][:, bu]
Kuk = K[bu, :][:, bk]
Kkk = K[bk, :][:, bk]

# finding natural frequencies and orthonormal base
L = cholesky(Muu, lower=True)
Linv = np.linalg.inv(L)
Ktilde = Linv @ Kuu @ Linv.T
p = 20
gamma, V = eigh(Ktilde, subset_by_index=(0, p-1)) # already gives V[:, i] normalized to 1
omegan = gamma**0.5
print('First 5 natural frequencies', omegan[:5])

# calculating vibration modes from orthonormal base (remember U = L^(-T) V)
modes = np.zeros((DOF*n, len(gamma)))
for i in range(modes.shape[1]):
    modes[bu, i] = Linv.T @ V[:, i]

# ploting vibration modes
for i in range(5):
    plt.clf()
    plt.title(r'$\omega_n = %1.2f\ Hz$' % omegan[i])
    plt.plot(x+modes[0::DOF, i], y+modes[1::DOF, i])
    plt.savefig('mdof0307_plot_eigenmode_%02d.png' % i, bbox_inches='tight')

# performing dynamic analysis in time domain
tmax = 8
time_steps = 2000
plot_freq = 2

P = V

t = np.linspace(0, tmax, time_steps)

# gravity acceleration
g = -9.81 #m/s^2
# acceleration vector
uddot = np.zeros(DOF*n)
uddot[1::DOF] = g
# acceleration vector at known DOFs
uddotk = uddot[bk]
# acceleration vector at unknown DOFs
uddotug = uddot[bu]

# force due to gravity
fg = np.zeros(DOF*n)
fg[bu] = Muu @ uddotug + Muk @ uddotk

# force due to wind
# - using dynamic pressure q = rhoair*wind_speed**2/2
# - assuming pressure acting on one side of building
# - using log wind profile (https://en.wikipedia.org/wiki/Log_wind_profile)
#
rhoair = 1.225 # kg/m^3

def fwindspeed(y):
    y0 = 1.5 # m, roughness length for a dense urban area
    avg_building_height = 10 # m
    d = 2/3*avg_building_height # m
    ustar = 20/3.6 # m/s
    kappa = 0.41 # van Karman constant
    uy = ustar/kappa*np.log((y-d)/y0)
    uy[np.isnan(uy)] = 0
    uy[uy < 0] = 0
    return uy

wind_speed = fwindspeed(y)
plt.clf()
plt.title('Log Wind Profile')
plt.plot(wind_speed, y)
plt.xlabel('Lateral wind, $m/s$')
plt.ylabel('Height, $m$')
for yi, vx in zip(y[::n//10], wind_speed[::n//10]):
    plt.arrow(0, yi, vx, 0, width=0.3, length_includes_head=True)
plt.show()

# oscillatory wind parameters
wind_area = y*side
wind_freq = np.pi # rad/s

# initial conditions in physical space
u0 = np.zeros(DOF*n)
udot0 = np.zeros(DOF*n)
# initial conditions in modal space
r0 = P.T @ L.T @ u0[bu]
rdot0 = P.T @ L.T @ udot0[bu]

# dynamic analysis
# NOTE this can be further vectorized using NumPy bradcasting, but I kept this
# loop in order to make the code more understandable
f = np.zeros_like(fg)
u = np.zeros((DOF*n, len(t)))

#NOTE adding new np.array axis to vectorize calculations
on = omegan[:, None]

def r_t(t, t1, t2, on, fmodaln):
    """SDOF solution for an undamped single impulse
    """
    tn = (t1 + t2)/2
    dt = t2 - t1
    H = np.heaviside(t - tn, 1.)
    h = 1/on*np.sin(on*(t - tn))*H
    return fmodaln*dt*h

# homogeneous solution an undamped SDOF system
c1 = r0
c2 = rdot0/omegan
rh = c1[:, None]*np.sin(on*t) + c2[:, None]*np.cos(on*t)

# particular solution
# discrete approximation of Duhamel's convolution integral
rp = 0
for t1, t2 in zip(t[:-1], t[1:]):
    tn = (t1 + t2)/2
    f[:] = fg #gravitational forces
    wind_speed_total = wind_speed + wind_speed/10*np.sin(wind_freq*tn)
    f_wind = wind_area*rhoair*wind_speed_total**2/2
    f[0::DOF] = f_wind

    # calculating modal forces
    fmodaln = (P.T @ Linv @ f[bu])[:, None]
    # convolution
    rp += r_t(t, t1, t2, on, fmodaln)

# superposition between homogeneous and particular solutions
r = rh + rp

# transforming from r-space to displacement
u[bu] = Linv.T @ P @ r

plt.clf()
fig = plt.gcf()
for i in range(len(t)):
    if i % plot_freq == 0:
        plt.cla()
        plt.title('Oscillating building, t=%1.3f s' % t[i])
        plt.xlim(-60, 60)
        plt.ylim(0, y.max()*1.1)
        plt.plot(u[0::DOF, i]*m2mm, y)
        plt.xlabel('Lateral displacement, $mm$')
        plt.ylabel('Height, $m$')
        utip = u[0::DOF, i][-1]
        plt.text(0.0075, 1.05*y.max(), '%1.2f mm' % (utip*m2mm))
        plt.pause(1e-9)
        if not plt.fignum_exists(fig.number):
            break

plt.clf()
plt.plot(t, u[0::DOF, :][-1]*m2mm)
plt.ylabel('Lateral displacement, $mm$')
plt.xlabel('Time, $s$')
plt.show()
