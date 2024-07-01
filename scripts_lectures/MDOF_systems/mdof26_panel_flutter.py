import sys
# uncomment to run from the project root directory:
sys.path.append('.')

# uncomment to run from the scripts_lectures/MDOF_systems/ directory:
# sys.path.append('../..')

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
import numpy as np
from composites import isotropic_plate

from tudaesasII.quad4r import Quad4R, update_K, update_M, update_KA, DOF


plot = True
# number of nodes
nx = 21 # along x
ny = 21 # along y

#material properties (Aluminum)
E = 70e9
nu = 0.3
rho = 7.8e3

# design variables
h = 0.0035 # thickness in [m]
a = 0.8 # length along x in [m]
b = 0.5 # length along y in [m]

# creating mesh
xtmp = np.linspace(0, a, nx)
ytmp = np.linspace(0, b, ny)
xmesh, ymesh = np.meshgrid(xtmp, ytmp)

# node coordinates and position in the global matrix
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

num_elements = len(n1s)
print('num_elements', num_elements)

N = DOF*nx*ny

K = np.zeros((N, N))
M = np.zeros((N, N))
KA = np.zeros((N, N))

quads = []
for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
    pos1 = nid_pos[n1]
    pos2 = nid_pos[n2]
    pos3 = nid_pos[n3]
    pos4 = nid_pos[n4]
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
    update_KA(quad, nid_pos, ncoords, KA)
    update_M(quad, nid_pos, ncoords, M)
    quads.append(quad)

print('elements created')

# applying boundary conditions
# simply supported
bk = np.zeros(N, dtype=bool)
# constraining w at all edges
check = (np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0.) | np.isclose(y, b))
bk[2::DOF] = check
#NOTE uncomment for clamped
#bk[3::DOF] = check
#bk[4::DOF] = check
# removing u,v
bk[0::DOF] = True
bk[1::DOF] = True

# unconstrained nodes
bu = ~bk # logical_not

Kuu = csc_matrix(K[bu, :][:, bu])
Muu = csc_matrix(M[bu, :][:, bu])
KAuu = csc_matrix(KA[bu, :][:, bu])

num_eigenvalues = 6

def MAC(mode1, mode2):
    return (mode1@mode2)**2/((mode1@mode1)*(mode2@mode2))

MACmatrix = np.zeros((num_eigenvalues, num_eigenvalues))
rho_air = 1.225 # kg/m^3
v_sound = 343 # m/s
v_air = np.linspace(1.1*v_sound, 10*v_sound, 200)
Mach = v_air/v_sound
betas = rho_air*v_air**2/np.sqrt(Mach**2 - 1)
omegan_vec = []
for i, beta in enumerate(betas):
    print('analysis i', i)
    # solving generalized eigenvalue problem
    eigvals, eigvecsu = eigs(A=Kuu + beta*KAuu, M=Muu,
            k=num_eigenvalues, which='LM', sigma=-1., tol=1e-10)
    eigvecs = np.zeros((N, num_eigenvalues), dtype=float)
    eigvecs[bu, :] = eigvecsu

    if i == 0:
        eigvecs_ref = eigvecs

    corresp = []
    for j in range(num_eigenvalues):
        for k in range(num_eigenvalues):
            MACmatrix[j, k] = MAC(eigvecs_ref[:, j], eigvecs[:, k])
        if np.isclose(np.max(MACmatrix[j, :]), 1.):
            corresp.append(np.argmax(MACmatrix[j, :]))
        else:
            corresp.append(j)
    omegan_vec.append(eigvals[corresp]**0.5)
    print(np.round(MACmatrix, 2))

    eigvecs_ref = eigvecs[:, corresp].copy()


omegan_vec = np.array(omegan_vec)

if plot:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    for i in range(num_eigenvalues):
        plt.plot(Mach, omegan_vec[:, i])
    plt.ylabel('$\\omega_n\ [rad/s]$')
    plt.xlabel('Mach')
    plt.show()
