import sys
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np

from tudaesasII.beam2d import Beam2D, update_K, update_M, DOF
from tudaesasII.utils import plot_sparse_matrix


m2mm = 1000
# number of nodes
n = 150

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
    update_M(beam, nid_pos, M, lumped=False)
    elements.append(beam)

ax = plot_sparse_matrix(M)
ax.get_figure().show()


