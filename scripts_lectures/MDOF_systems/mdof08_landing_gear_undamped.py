# cover forced vibrations (slide 206)
# study ressonance
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy import dot, pi

from beam2D import Beam2D, update_K_M

create_animation = True


DOF = 3

# number of nodes
n = 300
# Material Lastrobe Lescalloy
E = 203.e9 # Pa

rho = 7.83e3 # kg/m3
m_wheel = 20 # kg
r_wheel = 0.3 # m
m_airplane = 250 # kg

I0_wheel = 1/12*m_wheel*(2*r_wheel)**2 # mass moment of inertia for wheel

thetas = np.linspace(pi/2, 0, n)
a = 2.4
b = 1.2
R = a*b/np.sqrt((b*np.cos(thetas))**2 + (a*np.sin(thetas))**2)
x = R*np.cos(thetas)
y = R*np.sin(thetas)

# tapered properties
b_root = h_root = 0.10 # m
b_tip = h_tip = 0.05 # m
b = np.linspace(b_root, b_tip, n)
h = np.linspace(h_root, h_tip, n)
A = b*h
Izz = 1/12*b*h**3

# getting nodes
ncoords = np.vstack((x ,y)).T
nids = 1 + np.arange(ncoords.shape[0])
nid_pos = dict(zip(nids, np.arange(len(nids))))

n1s = nids[0:-1]
n2s = nids[1:]

K = np.zeros((DOF*n, DOF*n))
M = np.zeros((DOF*n, DOF*n))
beams = []
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
    beam.interpolation = 'legendre'
    beam.n1 = n1
    beam.n2 = n2
    beam.E = E
    beam.rho = rho
    beam.A1, beam.A2 = A1, A2
    beam.Izz1, beam.Izz2 = Izz1, Izz2
    update_K_M(beam, nid_pos, ncoords, K, M, lumped=True)
    beams.append(beam)

# adding effect of concentrated aircraft mass into M
M[0, 0] += m_airplane/2 # u
M[1, 1] += m_airplane/2 # v
M[2, 2] += 0 # beta (no mass moment of inertia here)

# adding effect of concentrated wheel mass into M
M[-3, -3] += m_wheel # u
M[-2, -2] += m_wheel # v
M[-1, -1] += I0_wheel # beta

from scipy.linalg import cholesky
from numpy.linalg import eigh

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

L = cholesky(Muu, lower=True)

Linv = np.linalg.inv(L)
Ktilde = dot(dot(Linv, Kuu), Linv.T)

gamma, V = eigh(Ktilde) # already gives V[:, i] normalized to 1
omegan = gamma**0.5

print('Natural frequencies', omegan[:5])
eigmodes = np.zeros((DOF*n, len(gamma)))
for i in range(eigmodes.shape[1]):
    eigmodes[bu, i] = dot(Linv.T, V[:, i])

for i in range(5):
    plt.clf()
    plt.title(r'$\omega_n = %1.2f\ Hz$' % omegan[i])
    plt.plot(x+eigmodes[0::DOF, i], y+eigmodes[1::DOF, i])
    plt.savefig('exercise14_plot_eigenmode_%02d.png' % i, bbox_inches='tight')

nmodes = 10

P = V[:, :nmodes]

# creating function for vB(t)
v0h = 20
a0 = 0.01
Lr = 100
ac = 1/(2*Lr)*(-v0h**2) # constant deceleration rate
nbumps = 1000

tmax = 10
#import sympy
#t = sympy.Symbol('t', positive=True)
#tmax = float(sympy.solve(Lr - v0h*t - 1/2*ac*t**2, t)[0])

# times to plot vBt
t = np.linspace(0, tmax, 10000)

# displacements at B as a function of time
s = v0h*t + 1/2*ac*t**2
vBt = a0 * np.sin(2*nbumps*pi*s/Lr)
# displacement vector of known dofs
uk = np.zeros((len(known_ind), len(t)))
uk[-1] = vBt # -1 corresponds to vertical displ at B

# computing r(t) numerically using convolution

# accelerations at B as a function of time
d2vBt_dt2 = pi*a0*nbumps*(2.0*Lr*ac*np.cos(pi*nbumps*t*(ac*t + 2*v0h)/Lr) -
        4*pi*nbumps*(1.0*ac*t + v0h)**2*np.sin(pi*nbumps*t*(ac*t +
            2*v0h)/Lr))/Lr**2

# gravity acceleration
# known_ind = [0, 2, (K.shape[0]-1)-1]
# into uu
g = -9.81 #m/s**2
d2ualldt2 = np.zeros((DOF*n, len(t)))
d2ualldt2[1::DOF] = g
d2ukgdt2 = d2ualldt2[bk]
d2uugdt2 = d2ualldt2[bu]

# acceleration vector of known dofs
d2ukdt2 = np.zeros_like(d2ukgdt2)
d2ukdt2[-1] = d2vBt_dt2

# force due to gravity
fu = np.zeros(Kuu.shape[0])
fu = dot(Muu, d2uugdt2) + dot(Muk, d2ukgdt2)

# force from prescribed displacements
f = fu - dot(Muk, d2ukdt2) - dot(Kuk, uk)

# calculating modal forces
fmodal = dot(P.T, dot(Linv, f))

deltat = np.diff(t)[0]
on = omegan[:nmodes, None]

# particular solution
rp = np.zeros((nmodes, len(t)))
fm = fmodal[:nmodes, :]

# convolution loop
for i, tc in enumerate(t):
    # undamped heaviside function
    h = 1/on * np.heaviside(tc - t, 1)*np.sin(on*(tc - t))
    rp[:, i] = np.sum(fm * deltat * h, axis=1)

# homogeneous solution
u0 = np.zeros(DOF*n)
v0 = np.zeros(DOF*n)
# initial velocity
v0[1::DOF] = -3 # m/s
v0[-2] = 0
r0 = P.T @ L.T @ u0[bu]
rdot0 = P.T @ L.T @ v0[bu]
c1 = r0
c2 = rdot0/omegan[:nmodes]
rh = c1[:, None]*np.cos(on*t) + c2[:, None]*np.sin(on*t)

# total solution
r = rh + rp

u = np.zeros((DOF*n, len(t)))
u[bu] = Linv.T @ P @ r

if create_animation:
    tplot = t[0::10]
    uplot = u[:, 0::10]
    plt.clf()
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 2.5+uplot[-3, :].max())
    ax.set_ylim(-0.1, 1.3+uplot[1, :].max())
    ax.plot(x, y, '-k')
    ax.plot(x, y, '--r')
    lines = ax.get_lines()
    print('Creating animation')
    def animate(i):
        global lines
        plt.title('t=%1.2f s' % tplot[i])
        ui = uplot[:, i].reshape(n, 3)
        lines[1].set_data(*[x+ui[:, 0], y+ui[:, 1]])
        return lines
    ani = FuncAnimation(fig, animate, range(len(tplot)))
    ani.save('exercise14_plot_animation_reponse_undamped.html', fps=25)

