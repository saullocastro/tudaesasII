import sys
# uncomment to run from the project root directory:
sys.path.append('.')

# uncomment to run from the scripts_lectures/MDOF_systems/ directory:
# sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import eig

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF

plot_result = True

# number of nodes along x
nx = 20

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

alpha = 0.03
beta = 6.4e-6
C = alpha*M + beta*K

bubu = np.concatenate((bu, bu))
I = np.identity(M.shape[0])
ZERO = np.zeros_like(M)

A = np.vstack((np.column_stack((ZERO,   -I)),
               np.column_stack((   K, 1j*C))))

B = np.vstack((np.column_stack((   I, ZERO)),
               np.column_stack((ZERO,   -M))))*(1 + 0j)

# sub-matrices corresponding to unknown DOFs
Auu = A[bubu, :][:, bubu]
Buu = B[bubu, :][:, bubu]

# solving
# NOTE: extracting ALL eigenvectors
num_modes = 5
eigvals, Uphiuu = eig(a=Auu, b=Buu)

size = Uphiuu.shape[0]//2
Uu = Uphiuu[:size, :]
wn = -eigvals

Uu = Uu[:, wn>0]
wn = wn[wn > 0]

asort = np.argsort(wn)
wn = wn[asort]
Uu = Uu[:, asort]
print('wn', wn[:num_modes])


if plot_result:
    for mode in range(num_modes):
        plt.figure(mode+1)
        u = np.zeros(DOF*nx)
        u[bu] = Uu[:, mode].real
        u1 = u[0::DOF]
        u2 = u[1::DOF]
        plt.clf()
        plt.title('mode %02d, $\\omega_n$ %1.2f rad/s' % (mode+1, wn[mode].real))
        #plt.gca().set_aspect('equal')
        mag = u2
        levels = np.linspace(mag.min(), mag.max(), 100)
        xplot = xmesh + u1
        yplot = ymesh + u2
        plt.plot(xplot, yplot, 's--')
        plt.ylim(yplot.min(), yplot.max())

    plt.show()
