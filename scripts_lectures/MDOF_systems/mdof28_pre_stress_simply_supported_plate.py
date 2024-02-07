import sys
# uncomment to run from the project root directory:
sys.path.append('.')

# uncomment to run from the scripts_lectures/MDOF_systems/ directory:
# sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import isclose
from scipy.linalg import eigh
from scipy.sparse.linalg import spsolve, eigsh
from composites import isotropic_plate

from tudaesasII.quad4r import (Quad4R, update_K, update_KG, update_KNL,
        calc_fint, update_M, DOF)


nx = 19
ny = 19

a = 1.
b = 0.5

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

plate = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)

N = DOF*nx*ny
K = np.zeros((N, N))
KG = np.zeros((N, N))
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
    update_M(quad, nid_pos, ncoords, M)
    quads.append(quad)

print('elements created')

# applying boundary conditions
# simply supported
bk = np.zeros(N, dtype=bool) #array to store known DOFs
edges = isclose(x, 0.) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
bk[2::DOF][edges] = True
u_constr = isclose(x, a/2) & (isclose(y, 0) | isclose(y, b))
bk[0::DOF][u_constr] = True
v_constr = isclose(y, b/2) & (isclose(x, 0) | isclose(x, a))
bk[1::DOF][v_constr] = True

bu = ~bk # same as np.logical_not, defining unknown DOFs

# sub-matrices corresponding to unknown DOFs
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

# static case to calculate linear buckling load
def calc_fext(load):
    fext = np.zeros(N)
    ftotal = load
    # at x=0
    check = (isclose(x, 0) & ~isclose(y, 0) & ~isclose(y, b))
    fext[0::DOF][check] = -ftotal/(ny - 1)
    check = ((isclose(x, 0) & isclose(y, 0))
            |(isclose(x, 0) & isclose(y, b)))
    fext[0::DOF][check] = -ftotal/(ny - 1)/2
    assert isclose(fext.sum(), -ftotal)
    # at x=a
    check = (isclose(x, a) & ~isclose(y, 0) & ~isclose(y, b))
    fext[0::DOF][check] = ftotal/(ny - 1)
    check = ((isclose(x, a) & isclose(y, 0))
            |(isclose(x, a) & isclose(y, b)))
    fext[0::DOF][check] = ftotal/(ny - 1)/2
    assert isclose(fext.sum(), 0)
    return fext

fext = calc_fext(1.)
u0 = np.zeros(N)
u0u = np.linalg.solve(Kuu, fext[bu])
u0[bu] = u0u

# geometric stiffness matrix
KG *= 0
for quad in quads:
    update_KG(quad, u0, nid_pos, ncoords, KG)
KGuu = KG[bu, :][:, bu]

# linear buckling analysis
num_modes = 2
linbuck_eigvals, linbuck_eigvecsu = eigh(a=-KGuu, b=Kuu, subset_by_index=[0, num_modes-1])
lambda_CR = -0.999999*1/linbuck_eigvals[0]
linbuck_eigvecs = np.zeros((N, num_modes))
linbuck_eigvecs[bu] = linbuck_eigvecsu

# pre-load effect on natural frequencies
print('lambda_CR', lambda_CR)
preload_list = np.linspace(-0.9999*lambda_CR, +lambda_CR, 50)

#CASE 1, assuming KT = KC0 + KG
first_omegan = []
for lambda_i in preload_list:
    # solving generalized eigenvalue problem
    num_modes = 2
    eigvals, Uu = eigsh(A=Kuu + lambda_i*KGuu, M=Muu, k=num_modes, sigma=-1.,
            which='LM')
    omegan = np.sqrt(eigvals)
    first_omegan.append(omegan[0])

#CASE 2, finding KT with Newton-Raphson
def calc_KT(u):
    KNL = np.zeros((N, N))
    KG = np.zeros((N, N))
    for quad in quads:
        update_KNL(quad, u, nid_pos, ncoords, KNL)
        update_KG(quad, u, nid_pos, ncoords, KG)
    assert np.allclose(K + KNL + KG, (K + KNL + KG).T)
    return K + KNL + KG

first_omegan_NL = []
preload_list_NL = np.linspace(-0.995*lambda_CR, +lambda_CR, 15)
for lambda_i in preload_list_NL:
    print('lambda_i', lambda_i)
    u = np.zeros(K.shape[0])
    load_steps = lambda_i*np.linspace(0.1, 1., 3)
    for load in load_steps:
        fext = calc_fext(load)
        if np.isclose(load, 0.1*lambda_i):
            KT = K
            uu = spsolve(Kuu, fext[bu])
            u[bu] = uu
        for i in range(100):
            fint = calc_fint(quads, u, nid_pos, ncoords)
            R = fint - fext
            check = np.abs(R[bu]).max()
            if check < 0.1:
                KT = calc_KT(u) #NOTE modified Newton-Raphson since KT is calculated only after each load step
                break
            duu = spsolve(KT[bu, :][:, bu], -R[bu])
            u[bu] += duu
        assert i < 99

    num_modes = 3
    eigvals, Uu = eigsh(A=KT[bu, :][:, bu], M=Muu, k=num_modes, sigma=-1., which='LM')
    omegan = np.sqrt(eigvals)
    first_omegan_NL.append(omegan[0])

plt.clf()
plt.plot(preload_list, first_omegan, 'ko--', mfc='None',
    label=r'$K_T \approx K + K_G$')
plt.plot(preload_list_NL, first_omegan_NL, 'rs-', mfc='None',
    label=r'$K_T$ from NL equilibrium')
plt.title('Pre-stress effect for a simply-supported plate')
plt.xlabel('Pre-load [N]')
plt.ylabel('First natural frequency [rad/s]')
plt.yscale('linear')
plt.legend()
plt.show()
