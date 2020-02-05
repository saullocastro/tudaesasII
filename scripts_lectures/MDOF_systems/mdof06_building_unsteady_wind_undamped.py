# cover forced vibrations (slide 206)
# study ressonance
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, pi
from scipy.linalg import cholesky
from numpy.linalg import eigh
from numba import njit

from beam2D import Beam2D, update_K_M


DOF = 3

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
    update_K_M(beam, nid_pos, ncoords, K, M, lumped=True)
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
Ktilde = dot(dot(Linv, Kuu), Linv.T)
gamma, V = eigh(Ktilde) # already gives V[:, i] normalized to 1
omegan = gamma**0.5
print('First 5 natural frequencies', omegan[:5])

# calculating vibration modes from orthonormal base (remember U = L^(-T) V)
modes = np.zeros((DOF*n, len(gamma)))
for i in range(modes.shape[1]):
    modes[bu, i] = dot(Linv.T, V[:, i])

# ploting vibration modes
for i in range(5):
    plt.clf()
    plt.title(r'$\omega_n = %1.2f\ Hz$' % omegan[i])
    plt.plot(x+modes[0::DOF, i], y+modes[1::DOF, i])
    plt.savefig('exercise12_plot_eigenmode_%02d.png' % i, bbox_inches='tight')

# performing dynamic analysis in time domain
nmodes = 20
tmax = 10
time_steps = 10000
plot_freq = 10

P = V[:, :nmodes]

times = np.linspace(0, tmax, time_steps)

# gravity acceleration
g = -9.81 #m/s^2
# acceleration vector
d2udt2 = np.zeros(DOF*n)
d2udt2[1::DOF] = g
# acceleration vector at known DOFs
d2ukgdt2 = d2udt2[bk]
# acceleration vector at unknown DOFs
d2uugdt2 = d2udt2[bu]

# force due to gravity (explained Assignment documents)
fg = np.zeros(DOF*n)
fg[bu] = dot(Muu, d2uugdt2) + dot(Muk, d2ukgdt2)

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

# - oscillatory wind
wind_area = y*side
wind_freq = 2*pi/5 # rad/s

deltat = times[1] - times[0]

# homogeneous solution using initial conditions
u0 = np.zeros(DOF*n)
v0 = np.zeros(DOF*n)
r0 = P.T @ L.T @ u0[bu]
rdot0 = P.T @ L.T @ v0[bu]
c1 = r0
c2 = rdot0/omegan[:nmodes]

# dynamic analysis for each time tc
# NOTE this can be further vectorized using NumPy bradcasting, but I kept this
# loop in order to make the code more understandable
f = np.zeros_like(fg)
u = np.zeros(DOF*n)
rpc = np.zeros(nmodes)

on = omegan[:nmodes]

# convolution integral: general load as a sequence of impulse loads
# NOTE using NumBa's njit to accelerate here
@njit(nogil=True)
def calc_rp(c, deltat, tc, on, fmodal, times, rpc):
    rpc *= 0
    for i, tn in enumerate(times[:c]):
        # undamped heaviside function
        h = 1/on*np.sin(on*(tc - tn))
        rpc += fmodal[i]*h*deltat

fmodal = np.zeros((len(times), nmodes))
for c, tc in enumerate(times):
    f[:] = fg
    wind_speed_total = wind_speed + wind_speed/10*np.sin(wind_freq*tc)
    f_wind = wind_area*rhoair*wind_speed_total**2/2
    f[0::DOF] = f_wind

    # calculating modal forces
    fmodal[c] = dot(P.T, dot(Linv, f[bu]))

    calc_rp(c, deltat, tc, on, fmodal, times, rpc)

    # superposition with homogeneous solution (using initial conditions)
    rh = c1*np.sin(on*tc) + c2*np.cos(on*tc)
    r = rh + rpc
    # transforming from r-space to displacement
    u[bu] = Linv.T @ P @ r

    if c % plot_freq == 0:
        plt.clf()
        plt.title('Oscillating building, t=%1.3f s' % tc)
        plt.xlim(-0.1, 0.1)
        plt.plot(u[0::DOF], y)
        plt.xlabel('Lateral displacement, $m$')
        plt.ylabel('Height, $m$')
        utip = u[0::DOF][-1]
        plt.text(0, y.max(), '%1.2f mm' % (utip*1000))
        plt.pause(1e-9)
