import numpy as np
import sympy
from sympy import simplify, integrate, Matrix
from sympy.vector import CoordSys3D, cross

r"""

   3
   |\
   | \    positive normal in CCW
   |  \   ----------------------
   |___\
   1    2

"""
DOF = 5
num_nodes = 3

sympy.var('h, A', positive=True, real=True)
sympy.var('x1, y1, x2, y2, x3, y3, x, y', real=True, positive=True)
sympy.var('rho')
sympy.var('A11, A12, A16, A22, A26, A66')
sympy.var('B11, B12, B16, B22, B26, B66')
sympy.var('D11, D12, D16, D22, D26, D66')
sympy.var('E44, E45, E55')
sympy.var('d')
#NOTE shear correction factor should be applied to E44, E45 and E55
#     in the finite element code

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
N1, N2 = sympy.var('N1, N2')
N3 = 1 - N1 - N2

Nu = Matrix(   [[N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3, 0, 0, 0, 0]])
Nv = Matrix(   [[0, N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3, 0, 0, 0]])
Nw = Matrix(   [[0, 0, N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3, 0, 0]])
Nphix = Matrix([[0, 0, 0, N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3, 0]])
Nphiy = Matrix([[0, 0, 0, 0, N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3]])

BL = Matrix(
        # u v w  phix  phy (node 1, node2, node3)
        [[N1x, 0, 0, 0, 0, N2x, 0, 0, 0, 0, N3x, 0, 0, 0, 0],        #exx = u,x
         [0, N1y, 0, 0, 0, 0, N2y, 0, 0, 0, 0, N3y, 0, 0, 0],        #eyy = v,u
         [N1y, N1x, 0, 0, 0, N2y, N2x, 0, 0, 0, N3y, N3x, 0, 0, 0],  #gxy = u,y + v,x
         [0, 0, 0, N1x, 0, 0, 0, 0, N2x, 0, 0, 0, 0, N3x, 0],        #kxx = phix,x
         [0, 0, 0, 0, N1y, 0, 0, 0, 0, N2y, 0, 0, 0, 0, N3y],        #kyy = phiy,y
         [0, 0, 0, N1y, N1x, 0, 0, 0, N2y, N2x, 0, 0, 0, N3y, N3x],  #kxy = phix,y + phiy,x
         [0, 0, N1y, 0, N1, 0, 0, N2y, 0, N2, 0, 0, N3y, 0, N3],     #gyz = w,y + phiy
         [0, 0, N1x, N1, 0, 0, 0, N2x, N2, 0, 0, 0, N3x, N3, 0],     #gxz = w,x + phix
        ])

# Constitutive linear stiffness matrix
Ke = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Melumped = sympy.zeros(num_nodes*DOF, num_nodes*DOF)

ABDE = Matrix(
        [[A11, A12, A16, B11, B12, B16, 0, 0],
         [A12, A22, A26, B12, B22, B26, 0, 0],
         [A16, A26, A66, B16, B26, B66, 0, 0],
         [B11, B12, B16, D11, D12, D16, 0, 0],
         [B12, B22, B26, D12, D22, D26, 0, 0],
         [B16, B26, B66, D16, D26, D66, 0, 0],
         [0, 0, 0, 0, 0, 0, E44, E45],
         [0, 0, 0, 0, 0, 0, E45, E55]])

Ke[:, :] = detJ*simplify(integrate(integrate(
    BL.T*ABDE*BL, (N2, 0, 1-N1)), (N1, 0, 1)))

#reduced integration for stiffness
Ke[:, :] = detJ*simplify((BL.T*ABDE*BL).subs({N1: ONE/3, N2: ONE/3}))

# Mass matrix adapted from (#TODO offset yet to be checked):
d = 0
# Flutter of stiffened composite panels considering the stiffener's base as a structural element
# Saullo G.P. Castro, Thiago A.M. Guimarães et al.
# Composite Structures, 140, 4 2016
# https://www.sciencedirect.com/science/article/pii/S0263822315011460
#maux = Matrix([[ 1, 0, 0, -d, 0],
               #[ 0, 1, 0, 0, -d],
               #[ 0, 0, 1, 0, 0],
               #[-d, 0, 0, (h**2/12 + d**2), 0],
               #[0, -d, 0, 0, (h**2/12 + d**2)]])
#tmp = Matrix([Nu, Nv, Nw, Nphix, Nphiy])
#Meintegrand = rho*h*tmp.T*maux*tmp

Meintegrand = rho*(h*Nu.T*Nu + h*Nv.T*Nv + h*Nw.T*Nw +
    h**3/12*Nphix.T*Nphix + h**3/12*Nphiy.T*Nphiy)
Me[:, :] = detJ*integrate(integrate(Meintegrand, (N2, 0, 1-N1)), (N1, 0, 1))

#TODO Melumped, figure out how to compute mass moment of inertias

print('Integrating Ke')
integrands = []
for ind, integrand in np.ndenumerate(Ke):
    integrands.append(integrand)
for i, (ind, integrand) in enumerate(np.ndenumerate(Ke)):
    Ke[ind] = simplify(integrands[i])

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

if False:
    print('printing Mlumped')
    for ind, val in np.ndenumerate(Mlumped):
        if val == 0:
            continue
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('    Mlumped[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', Mlumped[ind])

