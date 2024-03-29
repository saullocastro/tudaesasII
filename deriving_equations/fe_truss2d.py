import numpy as np
import sympy
from sympy import Matrix, integrate

DOF = 3

def name_ind(i):
    if i >=0 and i < DOF:
        return 'c1'
    elif i >= DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    else:
        raise

sympy.var('Izz, A, le, xi')
sympy.var('rho, E, nu, c, s')

N1 = (1-xi)/2
N2 = (1+xi)/2

Nu = Matrix([[N1, 0, 0, N2, 0, 0]])
Nv = Matrix([[0, N1, 0, 0, N2, 0]])
Nbeta = Matrix([[0, 0, N1, 0, 0, N2]])

# Constitutive linear stiffness matrix
#c
R = Matrix([[c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [ 0,  0 , 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [ 0,  0 , 0, 0, 0, 1]])

num_nodes = 2

BLu = Matrix([[(2/le)*N1.diff(xi), 0, (2/le)*N2.diff(xi), 0]])

Ke = A*E/le*Matrix([[1, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [-1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])

Me = (le/2)*sympy.integrate(rho*A*(Nu.T*Nu + Nv.T*Nv) + Izz*rho*Nbeta.T*Nbeta,
        (xi, -1, +1))
mA = integrate((le/2)*rho*A, (xi, -1, 0))
mB = integrate((le/2)*rho*A, (xi, 0, +1))
x = (xi + 1)*le/2
MMIA = integrate((le/2)*rho*(Izz + x**2*A), (xi, -1, 0))
MMIB = integrate((le/2)*rho*(Izz + (le-x)**2*A), (xi, 0, +1))
Me_lumped = Matrix([[mA, 0, 0, 0, 0, 0],
                    [0, mA, 0, 0, 0, 0],
                    [0, 0, MMIA, 0, 0, 0],
                    [0, 0, 0, mB, 0, 0],
                    [0, 0, 0, 0, mB, 0],
                    [0, 0, 0, 0, 0, MMIB]])

print('printing Me')
for ind, val in np.ndenumerate(Me):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    Me[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', Me[ind])
print('printing Me_lumped')
for ind, val in np.ndenumerate(Me_lumped):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    Me[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', Me_lumped[ind])

K = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
M = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
M_lumped = sympy.zeros(num_nodes*DOF, num_nodes*DOF)

# global
K[:, :] = R.T*Ke*R
M[:, :] = R.T*Me*R
M_lumped[:, :] = R.T*Me_lumped*R

# K represents the global stiffness matrix
# in case we want to apply coordinate transformations

print('printing K')
for ind, val in np.ndenumerate(K):
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    K[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', K[ind])

print('printing M')
for ind, val in np.ndenumerate(M):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M[ind])

print('printing M_lumped')
for ind, val in np.ndenumerate(M_lumped):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M_lumped[ind])
