import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import eigh

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF

plot_mesh = True
plot_result = True

lumped = False

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
    update_M(elem, nid_pos, M, lumped=lumped)
    elems.append(elem)

if lumped:
    assert np.count_nonzero(M-np.diag(np.diagonal(M))) == 0

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

# solving
# NOTE: extracting ALL eigenvectors
num_modes = 5
eigvals, U = eigh(a=Kuu, b=Muu, subset_by_index=(0, num_modes))
wn = eigvals**0.5
print('wn', wn[:5])

print('omega1 ref', 1.875**2*np.sqrt(E*Izz/(rho*A*length**4)))
print('omega2 ref', 4.694**2*np.sqrt(E*Izz/(rho*A*length**4)))
print('omega3 ref', 7.855**2*np.sqrt(E*Izz/(rho*A*length**4)))

if plot_mesh:
    plt.plot(x, y, 'o-')
    plt.show()

if plot_result:
    u = np.zeros(K.shape[0], dtype=float)
    for mode in range(num_modes):
        u[bu] = U[:, mode]
        scale = 10
        u1 = scale*u[0::DOF]
        u2 = scale*u[1::DOF]
        plt.clf()
        plt.ylim(-scale/10, scale/10)
        plt.title('mode %02d, $\omega_n$ %1.2f rad/s' % (mode+1, wn[mode]))
        plt.gca().set_aspect('equal')
        mag = u2
        levels = np.linspace(mag.min(), mag.max(), 100)
        xplot = xmesh + u1
        yplot = ymesh + u2
        plt.plot(xplot, yplot)
        plt.show()
