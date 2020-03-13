import sys
sys.path.append('../..')

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.linalg import cholesky, eigh
from composites.laminate import read_isotropic

from tudaesasII.quad4r import Quad4R, update_K, update_M, DOF


nx = 25
ny = 25

a = 300/1000 # [m]
b = 274/1000 # [m]

E = 70.e9 # Pa
nu = 0.33
rho = 2.7e3 # kg/m3
h = 0.001 # m

xtmp = np.linspace(0, a, nx)
ytmp = np.linspace(0, b, ny)
xmesh, ymesh = np.meshgrid(xtmp, ytmp)
ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
x = ncoords[:, 0]
y = ncoords[:, 1]

nids = 1 + np.arange(ncoords.shape[0])
nid_pos = dict(zip(nids, np.arange(len(nids))))
nids_mesh = nids.reshape(nx, ny)
n1s = nids_mesh[:-1, :-1].flatten()
n2s = nids_mesh[1:, :-1].flatten()
n3s = nids_mesh[1:, 1:].flatten()
n4s = nids_mesh[:-1, 1:].flatten()

plate = read_isotropic(thickness=h, E=E, nu=nu, calc_scf=True)

N = DOF*nx*ny
K = np.zeros((N, N))
M = np.zeros((N, N))
quads = []

for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
    pos1 = nid_pos[n1]
    pos2 = nid_pos[n2]
    pos3 = nid_pos[n3]
    pos4 = nid_pos[n4]
    r1 = ncoords[pos1]
    r2 = ncoords[pos2]
    r3 = ncoords[pos3]
    normal = np.cross(r2 - r1, r3 - r2)
    assert normal > 0 # guaranteeing that all elements have CCW positive normal
    quad = Quad4R()
    quad.rho = rho
    quad.n1 = n1
    quad.n2 = n2
    quad.n3 = n3
    quad.n4 = n4
    quad.scf13 = plate.scf_k13
    quad.scf23 = plate.scf_k23
    quad.h = h
    quad.ABDE = plate.ABDE
    update_K(quad, nid_pos, ncoords, K)
    update_M(quad, nid_pos, ncoords, M, lumped=False)
    quads.append(quad)

print('elements created')

bk = np.zeros(N, dtype=bool) # constrained DOFs, can be used to prescribe displacements
#bk[2::DOF] = check
# eliminating u and v
bk[0::DOF] = True
bk[1::DOF] = True

# unknown DOFs
bu = ~bk

# sub-matrices corresponding to unknown DOFs
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

L = cholesky(Muu, lower=True)
Linv = np.linalg.inv(L)
Ktilde = Linv @ Kuu @ Linv.T

Nmodes = 13
gamma, V = eigh(Ktilde, eigvals=(0, Nmodes-1)) # already gives V[:, i] normalized to 1
dummy = 3
V = V[:, dummy:]
gamma = gamma[dummy:]

omegan = gamma**0.5

P = V

# performing time-domain analysis
tmax = 4
time_steps = 100000
plot_freq = 10
t = np.linspace(0, tmax, time_steps)

zeta = 0.02

on = omegan
od = on*np.sqrt(1 - zeta**2)

# dynamic analysis
u = np.zeros((N, len(t)))

# initial conditions
v0 = np.zeros(N)
u0 = np.zeros(N)

# modal space
r0 = P.T @ L.T @ u0[bu]
rdot0 = P.T @ L.T @ v0[bu]

on = on[:, None]
od = od[:, None]

# homogeneous solution
rh = np.exp(-zeta*on*t)*(r0[:, None]*np.cos(od*t) +
    (rdot0[:, None] + zeta*on*r0[:, None])*np.sin(od*t)/od)

# convolution integral: general load as a sequence of impulse loads
def r_t(t, t1, t2, on, zeta, od, fmodaln):
    tn = (t1 + t2)/2
    dt = t2 - t1
    # damped function
    H = np.heaviside(t - tn, 1.)
    h = np.zeros((Nmodes-dummy, t.shape[0]))
    check = t >= tn
    h[:, check] = 1/od*np.exp(-zeta*on*(t[check] - tn))*np.sin(od*(t[check] - tn))*H[check]
    return fmodaln*dt*h

force_pos1 = np.where(np.isclose(x, a/4) & np.isclose(y, b/4))[0][0]
acc_pos5 = np.where(np.isclose(x, a/2) & np.isclose(y, b/2))[0][0]


# hammer loads from Excel polynomial curve-fit
f_hammer = lambda t: -87745588352.1543*t**4 + 1415919808.3273*t**3 - 7623386.1429*t**2 + 13762.9340*t - 0.4145

rpc = np.zeros((Nmodes-dummy, len(t)))
fu = np.zeros(N)[bu]

for t1, t2 in zip(t[:-1], t[1:]):
    tn = (t1 + t2)/2

    if tn <= 0.006:
        fu[:] = 0
        fu[DOF*force_pos1+2] += f_hammer(tn)
    else:
        break

    # calculating modal forces
    fmodaln = (P.T @ Linv @ fu)[:, None]

    # convolution
    rpc += r_t(t, t1, t2, on, zeta, od, fmodaln)

# superposition between homogeneous solution and forced solution
r = rh + rpc

# transforming from r-space to displacement
u[bu] = Linv.T @ P @ r

if False:
    plt.clf()
    plt.ion()
    fig = plt.gcf()
    w = u[2::DOF]
    for i in range(1, len(t)):
        if i % plot_freq == 0:
            plt.clf()
            plt.gca().set_aspect('equal')
            plt.title('$t = %1.3f s$' % t[i])
            tmp = u[2::DOF, i].reshape(xmesh.T.shape).T
            lev = np.linspace(tmp.min(), tmp.max(), 200)
            plt.contourf(xmesh, ymesh,  tmp,
                    cmap=cm.jet, levels=lev)
            #plt.colorbar()
            plt.show()
            plt.pause(1e-9)
            if not plt.fignum_exists(fig.number):
                break

else:
    plt.clf()
    wcenter = u[DOF*acc_pos5+2]
    plt.plot(t, wcenter)
    plt.show()
