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
#NOTE for linear element these derivatives are constant
# xi = xi(x, y)
# eta = eta(x, y)
# dx   = [dx/dxi  dx/deta ] dxi
# dy     [dy/dxi  dy/deta ] deta
#
J = Matrix([[xfunc.diff(xi), xfunc.diff(eta)],
            [yfunc.diff(xi), yfunc.diff(eta)]])
detJ = J.det().simplify()
print('detJ =', detJ)

# Jinv:
# d/dx = d/dxi*dxi/dx + d/deta*deta/dx = [dxi/dx   deta/dx] d/dxi
# d/dy   d/dxi*dxi/dy + d/deta*deta/dy   [dxi/dy   deta/dy] d/deta
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

N1y = j21*N1xi + j22*N1eta
N2y = j21*N2xi + j22*N2eta
N3y = j21*N3xi + j22*N3eta
N4y = j21*N4xi + j22*N4eta

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

#TODO hourglass control
# From: https://abaqus-docs.mit.edu/2017/English/SIMACAEELMRefMap/simaelm-c-sectioncontrol.htm#simaelm-c-esection-hourglass
sympy.var('hhg')
G = 1/3*(A12 + E44 + E55)/h
rFG = 0.005 * integrate(G/h, (h, -hhg/2, +hhg/2))
rthetaG = 0.00375 * 12 * integrate(G*h**2/h**3, (h, -hhg/2, +hhg/2))

hhg = (12*(E44 + E55 + A66)/(A11 + A22 + 0))**0.5
rthetaG = -0.18*(0.333333333333333*A12 + 0.333333333333333*E44 + 0.333333333333333*E55)/hhg
rFG = -0.02*(0.333333333333333*A12 + 0.333333333333333*E44 + 0.333333333333333*E55)/hhg

B1P = Matrix([[N1x, 0, 0, 0, 0,
               N2x, 0, 0, 0, 0,
               N3x, 0, 0, 0, 0,
               N4x, 0, 0, 0, 0]])
B2P = Matrix([[0, N1y, 0, 0, 0,
               0, N2y, 0, 0, 0,
               0, N3y, 0, 0, 0,
               0, N4y, 0, 0, 0]])
B1Pr = Matrix([[0, 0, 0, N1x, 0,
                0, 0, 0, N2x, 0,
                0, 0, 0, N3x, 0,
                0, 0, 0, N4x, 0]])
B2Pr = Matrix([[0, 0, 0, 0, N1y,
                0, 0, 0, 0, N2y,
                0, 0, 0, 0, N3y,
                0, 0, 0, 0, N4y]])


BalphaP = Matrix([B1P, B2P, B1Pr, B2Pr])
Khse = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
ss = 1
sr = 1
Khse[:, :] = ss*rFG*(BalphaP.T*BalphaP)*h*A + sr*rthetaG*(BalphaP.T*BalphaP)*h**3*A


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

Ke[:, :] = wij*detJ*BL.T*ABDE*BL

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
#Khs = Khse
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

#print()
#for ind, val in np.ndenumerate(Khs):
    #if val == 0:
        #continue
    #i, j = ind
    #si = name_ind(i)
    #sj = name_ind(j)
    #print('    K[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', Khs[ind])

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

