import numpy as np
import sympy
from sympy import simplify, integrate, Matrix
from sympy.vector import CoordSys3D, cross

r"""

    4 ____ 3
     /   /
    /   /   positive normal in CCW
   /___/
   1    2

"""

DOF = 5
num_nodes = 4

sympy.var('h', positive=True, real=True)
sympy.var('x1, y1, x2, y2, x3, y3, x4, y4', real=True, positive=True)
sympy.var('rho, xi, eta, A')
sympy.var('A11, A12, A16, A22, A26, A66')
sympy.var('B11, B12, B16, B22, B26, B66')
sympy.var('D11, D12, D16, D22, D26, D66')
sympy.var('E44, E45, E55')
#NOTE shear correction factor should be applied to E44, E45 and E55
#     in the finite element code

ONE = sympy.Integer(1)

R = CoordSys3D('R')
r1 = x1*R.i + y1*R.j
r2 = x2*R.i + y2*R.j
r3 = x3*R.i + y3*R.j
r4 = x4*R.i + y4*R.j

rbottom = r1 + (r2 - r1)*(xi + 1)/2
rtop = r4 + (r3 - r4)*(xi + 1)/2
r = rbottom + (rtop - rbottom)*(eta + 1)/2
xfunc = r.components[R.i]
yfunc = r.components[R.j]

# Jacobian
# http://kis.tu.kielce.pl/mo/COLORADO_FEM/colorado/IFEM.Ch17.pdf
#NOTE for linear element these derivatives are constant
# xi = xi(x, y)
# eta = eta(x, y)
#J = [dx/dxi  dy/dxi ]
#    [dx/deta dy/deta]
# dx   = J.T dxi
# dy         deta
#
# dxi   = Jinv.T dx
# deta           dy
#
# Jinv:
# d/dx = d/dxi*dxi/dx + d/deta*deta/dx = [dxi/dx   deta/dx] d/dxi  =  [j11  j12] d/dxi
# d/dy   d/dxi*dxi/dy + d/deta*deta/dy   [dxi/dy   deta/dy] d/deta =  [j21  j22] d/deta
#
J = Matrix([[xfunc.diff(xi),  yfunc.diff(xi)],
            [xfunc.diff(eta), yfunc.diff(eta)]])
detJ = J.det().simplify()
print('detJ =', detJ)

j = J.inv()
j11 = j[0, 0].simplify()
j12 = j[0, 1].simplify()
j21 = j[1, 0].simplify()
j22 = j[1, 1].simplify()

print('j11 =', j11.simplify())
print('j12 =', j12.simplify())
print('j21 =', j21.simplify())
print('j22 =', j22.simplify())

j11, j12, j21, j22 = sympy.var('j11, j12, j21, j22')

N1 = (eta*xi - eta - xi + 1)/4
N2 = -(eta*xi + eta - xi - 1)/4
N3 = (eta*xi + eta + xi + 1)/4
N4 = -(eta*xi - eta + xi - 1)/4

N1xi = N1.diff(xi)
N2xi = N2.diff(xi)
N3xi = N3.diff(xi)
N4xi = N4.diff(xi)

N1eta = N1.diff(eta)
N2eta = N2.diff(eta)
N3eta = N3.diff(eta)
N4eta = N4.diff(eta)

N1x = j11*N1xi + j12*N1eta
N2x = j11*N2xi + j12*N2eta
N3x = j11*N3xi + j12*N3eta
N4x = j11*N4xi + j12*N4eta

N1xxi = N1x.diff(xi)
N1xeta = N1x.diff(eta)
N2xxi = N2x.diff(xi)
N2xeta = N2x.diff(eta)
N3xxi = N3x.diff(xi)
N3xeta = N3x.diff(eta)
N4xxi = N4x.diff(xi)
N4xeta = N4x.diff(eta)

N1xy = j21*N1xxi + j22*N1xeta
N2xy = j21*N2xxi + j22*N2xeta
N3xy = j21*N3xxi + j22*N3xeta
N4xy = j21*N4xxi + j22*N4xeta

N1y = j21*N1xi + j22*N1eta
N2y = j21*N2xi + j22*N2eta
N3y = j21*N3xi + j22*N3eta
N4y = j21*N4xi + j22*N4eta

N1yxi = N1y.diff(xi)
N1yeta = N1y.diff(eta)
N2yxi = N2y.diff(xi)
N2yeta = N2y.diff(eta)
N3yxi = N3y.diff(xi)
N3yeta = N3y.diff(eta)
N4yxi = N4y.diff(xi)
N4yeta = N4y.diff(eta)

N1yx = j11*N1yxi + j12*N1yeta
N2yx = j11*N2yxi + j12*N2yeta
N3yx = j11*N3yxi + j12*N3yeta
N4yx = j11*N4yxi + j12*N4yeta

print('N1 =', N1)
print('N2 =', N2)
print('N3 =', N3)
print('N4 =', N4)

print('N1x =', N1x.simplify())
print('N2x =', N2x.simplify())
print('N3x =', N3x.simplify())
print('N4x =', N4x.simplify())

print('N1y =', N1y.simplify())
print('N2y =', N2y.simplify())
print('N3y =', N3y.simplify())
print('N4y =', N4y.simplify())
print('')
print('N1xy =', N1xy.simplify())
print('N2xy =', N2xy.simplify())
print('N3xy =', N3xy.simplify())
print('N4xy =', N4xy.simplify())
print('')
print('Niyx only for checking purposes')
print('')
print('N1yx =', N1yx.simplify())
print('N2yx =', N2yx.simplify())
print('N3yx =', N3yx.simplify())
print('N4yx =', N4yx.simplify())
print('')

detJ = sympy.var('detJ')
N1, N2, N3, N4 = sympy.var('N1, N2, N3, N4')
N1x, N2x, N3x, N4x = sympy.var('N1x, N2x, N3x, N4x')
N1y, N2y, N3y, N4y = sympy.var('N1y, N2y, N3y, N4y')

Nu = Matrix(   [[N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3, 0, 0, 0, 0, N4, 0, 0, 0, 0]])
Nv = Matrix(   [[0, N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3, 0, 0, 0, 0, N4, 0, 0, 0]])
Nw = Matrix(   [[0, 0, N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3, 0, 0, 0, 0, N4, 0, 0]])
Nphix = Matrix([[0, 0, 0, N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3, 0, 0, 0, 0, N4, 0]])
Nphiy = Matrix([[0, 0, 0, 0, N1, 0, 0, 0, 0, N2, 0, 0, 0, 0, N3, 0, 0, 0, 0, N4]])

# u v w  phix  phy (node 1, node2, node3, node4)

#exx = u,x = (dxi/dx)*u,xi + (deta/dx)*u,eta = j11 u,xi + j12 u,eta
BLexx = Matrix([[N1x, 0, 0, 0, 0,
                 N2x, 0, 0, 0, 0,
                 N3x, 0, 0, 0, 0,
                 N4x, 0, 0, 0, 0]])
#eyy = v,y = (dxi/dy)*v,xi + (deta/dy)*v,eta = j21 v,xi + j22 v,eta
BLeyy = Matrix([[0, N1y, 0, 0, 0,
                 0, N2y, 0, 0, 0,
                 0, N3y, 0, 0, 0,
                 0, N4y, 0, 0, 0]])
#gxy = u,y + v,x = (dxi/dy)*u,xi + (deta/dy)*u,eta + (dxi/dx)*v,xi + (deta/dy)*v,eta
BLgxy = Matrix([[N1y, N1x, 0, 0, 0,
                 N2y, N2x, 0, 0, 0,
                 N3y, N3x, 0, 0, 0,
                 N4y, N4x, 0, 0, 0]])
#kxx = phix,x = (dxi/dx)*phix,xi + (deta/dx)*phix,eta
BLkxx = Matrix([[0, 0, 0, N1x, 0,
                 0, 0, 0, N2x, 0,
                 0, 0, 0, N3x, 0,
                 0, 0, 0, N4x, 0]])
BLkyy = Matrix([[0, 0, 0, 0, N1y,
                 0, 0, 0, 0, N2y,
                 0, 0, 0, 0, N3y,
                 0, 0, 0, 0, N4y]])
BLkxy = Matrix([[0, 0, 0, N1y, N1x,
                 0, 0, 0, N2y, N2x,
                 0, 0, 0, N3y, N3x,
                 0, 0, 0, N4y, N4x]])
BLgyz = Matrix([[0, 0, N1y, 0, N1,
                 0, 0, N2y, 0, N2,
                 0, 0, N3y, 0, N3,
                 0, 0, N4y, 0, N4]])
BLgxz = Matrix([[0, 0, N1x, N1, 0,
                 0, 0, N2x, N2, 0,
                 0, 0, N3x, N3, 0,
                 0, 0, N4x, N4, 0]])

BL = Matrix([BLexx, BLeyy, BLgxy, BLkxx, BLkyy, BLkxy, BLgyz, BLgxz])

# hourglass control as per Brockman 1987
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620241208
print('gamma1 = N1xy')
print('gamma2 = N2xy')
print('gamma3 = N3xy')
print('gamma4 = N4xy')
Eu = Ev = 0.10*A11/(1 + 1/A)
Ephix = 0.10*D11/(1 + 1/A)
Ephiy = 0.10*D22/(1 + 1/A)
Ew = (Ephix + Ephiy)/2
print('Eu =', Eu)
print('Ev =', Ev)
print('Ew =', Ew)
print('Ephix =', Ephix)
print('Ephiy =', Ephiy)
gamma1, gamma2, gamma3, gamma4 = sympy.var('gamma1, gamma2, gamma3, gamma4')
Eu, Ev, Ew, Ephix, Ephiy = sympy.var('Eu, Ev, Ew, Ephix, Ephiy')
Bgamma = Matrix([[gamma1, 0, 0, 0, 0,
                  gamma2, 0, 0, 0, 0,
                  gamma3, 0, 0, 0, 0,
                  gamma4, 0, 0, 0, 0],
                 [0, gamma1, 0, 0, 0,
                  0, gamma2, 0, 0, 0,
                  0, gamma3, 0, 0, 0,
                  0, gamma4, 0, 0, 0],
                 [0, 0, gamma1, 0, 0,
                  0, 0, gamma2, 0, 0,
                  0, 0, gamma3, 0, 0,
                  0, 0, gamma4, 0, 0],
                 [0, 0, 0, gamma1, 0,
                  0, 0, 0, gamma2, 0,
                  0, 0, 0, gamma3, 0,
                  0, 0, 0, gamma4, 0],
                 [0, 0, 0, 0, gamma1,
                  0, 0, 0, 0, gamma2,
                  0, 0, 0, 0, gamma3,
                  0, 0, 0, 0, gamma4],
                 ])
Egamma = Matrix([[Eu, 0, 0, 0, 0],
                 [0, Ev, 0, 0, 0],
                 [0, 0, Ew, 0, 0],
                 [0, 0, 0, Ephix, 0],
                 [0, 0, 0, 0, Ephiy]])

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

sympy.var('wij')

Ke[:, :] = wij*detJ*(BL.T*ABDE*BL + Bgamma.T*Egamma*Bgamma)

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

Me[:, :] = wij*detJ*rho*(h*Nu.T*Nu + h*Nv.T*Nv + h*Nw.T*Nw +
    h**3/12*Nphix.T*Nphix + h**3/12*Nphiy.T*Nphiy)

#TODO Melumped

# K represents the global stiffness matrix
# in case we want to apply coordinate transformations
K = Ke
M = Me
Mlumped = Melumped

def name_ind(i):
    if i >= 0*DOF and i < 1*DOF:
        return 'c1'
    elif i >= 1*DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    elif i >= 3*DOF and i < 4*DOF:
        return 'c4'
    else:
        raise

print()
for ind, val in np.ndenumerate(K):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    K[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', K[ind])


print()
for ind, val in np.ndenumerate(M):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M[ind])

if False:
    print()
    for ind, val in np.ndenumerate(Mlumped):
        if val == 0:
            continue
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('    Mlumped[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', Mlumped[ind])

