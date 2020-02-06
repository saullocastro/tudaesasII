import numpy as np
import sympy
from sympy import simplify, Matrix, integrate
from sympy.vector import CoordSys3D, cross

r"""

   3
   |\
   | \    positive normal in CCW
   |  \   ----------------------
   |___\
   1    2

"""
DOF = 2
num_nodes = 3

sympy.var('h, A', positive=True, real=True)
sympy.var('x1, y1, x2, y2, x3, y3, x, y', real=True, positive=True)
sympy.var('rho, E, nu', real=True, positive=True)

ONE = sympy.Integer(1)

R = CoordSys3D('R')
r1 = x1*R.i + y1*R.j
r2 = x2*R.i + y2*R.j
r3 = x3*R.i + y3*R.j
r = x*R.i + y*R.j

Aexpr = cross(r1 - r3, r2 - r3).components[R.k]/2
print('A =', Aexpr)

AN1 = cross(r2 - r, r3 - r).components[R.k]/2
AN2 = cross(r3 - r, r1 - r).components[R.k]/2
N1 = simplify(AN1/A)
N2 = simplify(AN2/A)
N3 = simplify((A - AN1 - AN2)/A)

N1x = N1.diff(x)
N2x = N2.diff(x)
N3x = N3.diff(x)
N1y = N1.diff(y)
N2y = N2.diff(y)
N3y = N3.diff(y)

# Jacobian
# N1 = N1(x, y)
# N2 = N2(x, y)
# dN1 = [dN1/dx  dN1/dy] dx
# dN2   [dN2/dx  dN2/dy] dy
#
Jinv = Matrix([[N1x, N1y],
               [N2x, N2y]])
detJ = Jinv.inv().det().simplify()
detJ = 2*A # it can be easily shown

print('N1x =', N1x)
print('N2x =', N2x)
print('N3x =', N3x)
print('N1y =', N1y)
print('N2y =', N2y)
print('N3y =', N3y)

N1x, N2x, N3x = sympy.var('N1x, N2x, N3x')
N1y, N2y, N3y = sympy.var('N1y, N2y, N3y')

# d/dx = dN1/dx*d/dN1 + dN2/dx*d/dN2 + dN3/dx*d/dN3

#NOTE evaluating at 1 integration point at the centre
Ke = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
C = E/((1+nu)*(1-2*nu))*sympy.Matrix(
        [[1-nu, nu, 0],
         [nu, 1-nu, 0],
         [0, 0, 1-2*nu]])
BL = Matrix(
        # u v (node 1, node2, node3)
        [[N1x, 0, N2x, 0, N3x, 0],         #exx = u,x
         [0, N1y, 0, N2y, 0, N3y],         #eyy = v,u
         [N1y, N1x, N2y, N2x, N3y, N3x],   #gxy = u,y + v,x
        ])
N1, N2 = sympy.var('N1, N2')
N3 = 1 - N1 - N2

Nu = Matrix([[N1, 0, N2, 0, N3, 0]])
Nv = Matrix([[0, N1, 0, N2, 0, N3]])

Ke[:, :] += detJ*integrate(integrate(h*BL.T*C*BL, (N2, 0, 1-N1)), (N1, 0, 1))
Me[:, :] += detJ*rho*h*integrate(integrate(Nu.T*Nu + Nv.T*Nv, (N2, 0, 1-N1)), (N1, 0, 1))

Melumped = rho*h*A/3*sympy.Matrix(
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]])

#TODO nonlinear terms

print('Integrating Ke (not really because it is constant)')
integrands = []
for ind, integrand in np.ndenumerate(Ke):
    integrands.append(integrand)
for i, (ind, integrand) in enumerate(np.ndenumerate(Ke)):
    Ke[ind] = sympy.simplify(integrands[i])

# K represents the global stiffness matrix
# in case we want to apply coordinate transformations
K = Ke
M = Me
Mlumped = Melumped

def name_ind(i):
    if i >=0 and i < DOF:
        return 'c1'
    elif i >= DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    else:
        raise

print('printing K')
for ind, val in np.ndenumerate(K):
    if val == 0:
        continue
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

print('printing Mlumped')
for ind, val in np.ndenumerate(Mlumped):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    Mlumped[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', Mlumped[ind])

