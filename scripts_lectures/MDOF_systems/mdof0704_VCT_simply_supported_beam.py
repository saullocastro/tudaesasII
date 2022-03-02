import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, cholesky, solve
from scipy.optimize import curve_fit

from tudaesasII.beam2d import (Beam2D, update_K, update_KG, update_KNL,
        update_M, DOF)

# number of nodes along x
nx = 11 #NOTE keep nx an odd number to have a node in the middle

# geometry
length = 2
h = 0.03
w = h
Izz = h**3*w/12
A = w*h

# material properties
E = 70e9
Fcy = 200e6 # yield stress
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
KG = np.zeros((DOF*nx, DOF*nx))
M = np.zeros((DOF*nx, DOF*nx))

beams = []
# creating beam elements
nids = list(nid_pos.keys())
for n1, n2 in zip(nids[:-1], nids[1:]):
    beam = Beam2D()
    beam.n1 = n1
    beam.n2 = n2
    beam.E = E
    beam.A1 = beam.A2 = A
    beam.Izz1 = beam.Izz2 = Izz
    beam.rho = rho
    beam.interpolation = 'legendre'
    update_K(beam, nid_pos, ncoords, K)
    update_M(beam, nid_pos, M, lumped=False)
    beams.append(beam)

# boundary conditions for the dynamic problem
bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
at_base = np.isclose(x, 0.)
bk[0::DOF][at_base] = True
bk[1::DOF][at_base] = True
at_tip = np.isclose(x, length)
bk[1::DOF][at_tip] = True

bu = ~bk # defining unknown DOFs

# partitioned matrices
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

num_modes = 3
eigvals, Uu = eigh(a=Kuu, b=Muu, subset_by_index=[0, num_modes-1])
omegan0 = np.sqrt(eigvals[0])

# geometric stiffness matrix
KG *= 0
for beam in beams:
    update_KG(beam, 1., nid_pos, ncoords, KG)
KGuu = KG[bu, :][:, bu]

# linear buckling analysis
num_modes = 3
linbuck_eigvals, _ = eigh(a=Kuu, b=KGuu, subset_by_index=[0, num_modes-1])

PCR = -linbuck_eigvals[0]
print('PCR', PCR)
Ppreload_list = np.linspace(0.1*PCR, 0.9999*PCR, 100)

#CASE 1, assuming KT = KC0 + KG
first_omegan = []
for Ppreload in Ppreload_list:
    # solving generalized eigenvalue problem
    num_modes = 3
    eigvals, Uu = eigh(a=Kuu + Ppreload*KGuu, b=Muu, subset_by_index=[0, num_modes-1])
    omegan = np.sqrt(eigvals)
    first_omegan.append(omegan[0])

#CASE 2, finding KT with Newton-Raphson
def calc_KT(u):
    KNL = np.zeros((DOF*nx, DOF*nx))
    KG = np.zeros((DOF*nx, DOF*nx))
    for beam in beams:
        update_KNL(beam, u, nid_pos, ncoords, KNL)
        update_KG(beam, u, nid_pos, ncoords, KG)
    assert np.allclose(K + KNL + KG, (K + KNL + KG).T)
    return K + KNL + KG

first_omegan_NL = []
estimated_PCR = []
one_minus_P_PCR_square = []
one_minus_omegan_omegan0_square = []
one_minus_omegan_omegan0_fourth = []
for Ppreload in Ppreload_list:
    u = np.zeros(K.shape[0])
    load_steps = Ppreload*np.linspace(0.1, 1., 10)
    for load in load_steps:
        fext = np.zeros(K.shape[0])
        fext[0::DOF][at_tip] = load
        if np.isclose(load, 0.1*Ppreload):
            KT = K
            uu = np.linalg.solve(Kuu, fext[bu])
            u[bu] = uu
        for i in range(100):
            R = (KT @ u) - fext
            check = np.abs(R[bu]).max()
            if check < 0.1:
                break
            duu = np.linalg.solve(KT[bu, :][:, bu], -R[bu])
            u[bu] += duu
            KT = calc_KT(u) #NOTE full Newton-Raphson since KT is calculated in every iteration
        assert i < 99

    KTuu = calc_KT(u)[bu, :][:, bu]
    num_modes = 3
    eigvals, Uu = eigh(a=KTuu, b=Muu, subset_by_index=[0, num_modes-1])
    omegan = np.sqrt(eigvals)
    estimated_PCR.append(Ppreload/(1 - (omegan[0]/omegan0)**2))
    one_minus_P_PCR_square.append((1 - Ppreload/PCR)**2)
    one_minus_omegan_omegan0_square.append(1 - (omegan[0]/omegan0)**2)
    one_minus_omegan_omegan0_fourth.append(1 - (omegan[0]/omegan0)**4)
    first_omegan_NL.append(omegan[0])

VCT_omegan = omegan0*np.sqrt(1 - Ppreload_list/PCR)

plt.clf()
plt.plot(Ppreload_list, first_omegan, 'ko', mfc='None',
    label=r'$\omega_n$ using $K_T \approx K + K_G$')
plt.plot(Ppreload_list, first_omegan_NL, 'rs', mfc='None',
    label=r'$\omega_n$ using $K_T$ from NL equilibrium')
plt.plot(Ppreload_list, VCT_omegan, 'b-', mfc='None',
    label=r'$\omega_n$ from VCT equation')
plt.title('Pre-stress effect for a simply-supported beam')
plt.xlabel('Pre-load [N]')
plt.ylabel('First natural frequency [rad/s]')
plt.yscale('linear')
plt.legend()
plt.show()

plt.clf()
plt.plot(Ppreload_list, first_omegan, 'ko', mfc='None',
    label=r'$\omega_n$ using $K_T \approx K + K_G$')
plt.plot(Ppreload_list, VCT_omegan, 'b-', mfc='None',
    label=r'$\omega_n$ from VCT equation')
plt.title('Pre-stress effect for a simply-supported beam')
plt.xlabel('Pre-load [N]')
plt.ylabel('First natural frequency [rad/s]')
plt.yscale('linear')
plt.legend()
plt.show()

plt.clf()
plt.plot(Ppreload_list, estimated_PCR, 'ks--', mfc='None')
plt.title('VCT to estimate $P_{CR}$ of a simply-supported beam')
plt.ylabel('Estimated $P_{CR}$ [N]')
plt.xlabel('Pre-load $P$ [N]')
plt.show()


def first_order_fit(x, a, b):
    return a + b*x

def second_order_fit(x, a, b, c):
    return a + b*x + c*x**2

load_ratios = [0.2, 0.5, 0.8, 0.96]
symbols = ['s', 'o', '^', '1']
colors = ['blue', 'orange', 'green', 'red']

for load_ratio, symbol, color in zip(load_ratios, symbols, colors):
    plt.clf()
    threshold = load_ratio*PCR
    valid_values = Ppreload_list > threshold
    xdata = np.asarray(one_minus_omegan_omegan0_fourth)[valid_values]
    ydata = np.asarray(one_minus_P_PCR_square)[valid_values]
    (a, b), corr = curve_fit(first_order_fit, xdata, ydata)
    xi2_omegan_0 = first_order_fit(1, a, b)
    xplot = np.asarray(one_minus_omegan_omegan0_fourth[:-4])
    plt.plot(xdata, ydata, marker=symbol, mec=color, mfc='None')
    plt.plot(xplot, first_order_fit(xplot, a, b), '-', color=color,
        label=r'First-order curve fit with data up to {0:1.0f}%'.format(load_ratio*100) + ' of $P_{CR}$')
    plt.xlim(0.15, 1.1)
    plt.ylim(-0.05, 0.85)
    plt.legend(loc='upper right')
    plt.text(0.16, 0.01, r'Nonlinear buckling load ($\omega_n \to 0$) = %1.2f' % (PCR*(1 - xi2_omegan_0**0.5)), ha='left', va='bottom')
    plt.xlabel(r'$1 - (\omega_n/{\omega_n}_0)^4$')
    plt.ylabel(r'$(1 - P/P_{CR})^2$')
    plt.savefig('mdof0704_VCT_Sosa_%02d.png' % (load_ratio*100), bbox_inches='tight')

for load_ratio, symbol, color in zip(load_ratios, symbols, colors):
    plt.clf()
    threshold = load_ratio*PCR
    valid_values = Ppreload_list > threshold
    xdata = np.asarray(one_minus_omegan_omegan0_square)[valid_values]
    ydata = np.asarray(one_minus_P_PCR_square)[valid_values]
    (a, b, c), corr = curve_fit(second_order_fit, xdata, ydata)
    xmin = -b/(2*c)
    xi2 = second_order_fit(xmin, a, b, c)
    xi2_omegan_0 = second_order_fit(1, a, b, c)
    plt.hlines([xi2], 0, 2, colors=color, linestyles='--', lw=0.5)
    xplot = np.asarray(one_minus_omegan_omegan0_square[:-4])
    plt.plot(xdata, ydata, marker=symbol, mec=color, mfc='None')
    plt.plot(xplot, second_order_fit(xplot, a, b, c), '-', color=color,
        label=r'Second-order curve fit with data up to {0:1.0f}%'.format(load_ratio*100) + ' of $P_{CR}$')
    plt.xlim(0, 1.0)
    plt.ylim(-0.05, 0.85)
    plt.legend()
    plt.text(0.01, 0.1, r'Nonlinear buckling load ($\omega_n \to 0$) = %1.2f' % (PCR*(1 - xi2_omegan_0**0.5)), ha='left', va='bottom')
    plt.text(0.01, 0.05, 'Nonlinear buckling load (Arbelo et al.) = %1.2f' % (PCR*(1 -
        xi2**0.5)), ha='left', va='bottom')
    #plt.xlabel(r'$\left(1 - \left(\frac{\omega_n}{{\omega_n}_0}\right)^2\right)$')
    #plt.ylabel(r'$\left(1 - \frac{P}{P_{CR}}\right)^2$')
    plt.xlabel(r'$1 - (\omega_n/{\omega_n}_0)^2$')
    plt.ylabel(r'$(1 - P/P_{CR})^2$')
    plt.savefig('mdof0704_VCT_Arbelo_%02d.png' % (load_ratio*100), bbox_inches='tight')

