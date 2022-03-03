import numpy as np
import sympy
from sympy import simplify, integrate, Matrix, symbols
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
sympy.var('Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qx, Qy')
sympy.var('ux, uy, vx, vy, wx, wy')
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

detJfunc = detJ
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
BNLexx = Matrix([[ux*N1x, vx*N1x, wx*N1x, 0, 0,
                  ux*N2x, vx*N2x, wx*N2x, 0, 0,
                  ux*N3x, vx*N3x, wx*N3x, 0, 0,
                  ux*N4x, vx*N4x, wx*N4x, 0, 0]])
#eyy = v,y = (dxi/dy)*v,xi + (deta/dy)*v,eta = j21 v,xi + j22 v,eta
BLeyy = Matrix([[0, N1y, 0, 0, 0,
                 0, N2y, 0, 0, 0,
                 0, N3y, 0, 0, 0,
                 0, N4y, 0, 0, 0]])
BNLeyy = Matrix([[uy*N1y, vy*N1y, wy*N1y, 0, 0,
                  uy*N2y, vy*N2y, wy*N2y, 0, 0,
                  uy*N3y, vy*N3y, wy*N3y, 0, 0,
                  uy*N4y, vy*N4y, wy*N4y, 0, 0]])
#gxy = u,y + v,x = (dxi/dy)*u,xi + (deta/dy)*u,eta + (dxi/dx)*v,xi + (deta/dy)*v,eta
BLgxy = Matrix([[N1y, N1x, 0, 0, 0,
                 N2y, N2x, 0, 0, 0,
                 N3y, N3x, 0, 0, 0,
                 N4y, N4x, 0, 0, 0]])
BNLgxy = Matrix([[uy*N1x + ux*N1y, vy*N1x + vx*N1y, wy*N1x + wx*N1y, 0, 0,
                  uy*N2x + ux*N2y, vy*N2x + vx*N2y, wy*N2x + wx*N2y, 0, 0,
                  uy*N3x + ux*N3y, vy*N3x + vx*N3y, wy*N3x + wx*N3y, 0, 0,
                  uy*N4x + ux*N4y, vy*N4x + vx*N4y, wy*N4x + wx*N4y, 0, 0]])
#kxx = phix,x = (dxi/dx)*phix,xi + (deta/dx)*phix,eta
BLkxx = Matrix([[0, 0, 0, N1x, 0,
                 0, 0, 0, N2x, 0,
                 0, 0, 0, N3x, 0,
                 0, 0, 0, N4x, 0]])
#kyy = phiy,y = (dxi/dy)*phiy,xi + (deta/dy)*phiy,eta
BLkyy = Matrix([[0, 0, 0, 0, N1y,
                 0, 0, 0, 0, N2y,
                 0, 0, 0, 0, N3y,
                 0, 0, 0, 0, N4y]])
#kxy = phix,y + phiy,x = (dxi/dy)*phix,xi + (deta/dy)*phix,eta
#                       +(dxi/dx)*phiy,xi + (deta/dx)*phiy,eta
BLkxy = Matrix([[0, 0, 0, N1y, N1x,
                 0, 0, 0, N2y, N2x,
                 0, 0, 0, N3y, N3x,
                 0, 0, 0, N4y, N4x]])
#gyz = phiy + w,y
BLgyz = Matrix([[0, 0, N1y, 0, N1,
                 0, 0, N2y, 0, N2,
                 0, 0, N3y, 0, N3,
                 0, 0, N4y, 0, N4]])
#gxz = phix + w,x
BLgxz = Matrix([[0, 0, N1x, N1, 0,
                 0, 0, N2x, N2, 0,
                 0, 0, N3x, N3, 0,
                 0, 0, N4x, N4, 0]])

Nux = Matrix([[N1x, 0, 0, 0, 0, N2x, 0, 0, 0, 0, N3x, 0, 0, 0, 0, N4x, 0, 0, 0, 0]])
Nuy = Matrix([[N1y, 0, 0, 0, 0, N2y, 0, 0, 0, 0, N3y, 0, 0, 0, 0, N4y, 0, 0, 0, 0]])
Nvx = Matrix([[0, N1x, 0, 0, 0, 0, N2x, 0, 0, 0, 0, N3x, 0, 0, 0, 0, N4x, 0, 0, 0]])
Nvy = Matrix([[0, N1y, 0, 0, 0, 0, N2y, 0, 0, 0, 0, N3y, 0, 0, 0, 0, N4y, 0, 0, 0]])
Nwx = Matrix([[0, 0, N1x, 0, 0, 0, 0, N2x, 0, 0, 0, 0, N3x, 0, 0, 0, 0, N4x, 0, 0]])
Nwy = Matrix([[0, 0, N1y, 0, 0, 0, 0, N2y, 0, 0, 0, 0, N3y, 0, 0, 0, 0, N4y, 0, 0]])

BL = Matrix([BLexx, BLeyy, BLgxy, BLkxx, BLkyy, BLkxy, BLgyz, BLgxz])
ZERO = 0*BNLexx
BNL = Matrix([BNLexx, BNLeyy, BNLgxy, ZERO, ZERO, ZERO, ZERO, ZERO])

# hourglass control as per Brockman 1987
# adapted to composites replacing E*h by A11 and E*h**3 by 12*D11
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620241208
print('gamma1 = N1xy')
print('gamma2 = N2xy')
print('gamma3 = N3xy')
print('gamma4 = N4xy')
Eu = Ev = 0.10*A11/(1 + 1/A)
Ephix = 12*0.10*D11/(1 + 1/A)
Ephiy = 12*0.10*D22/(1 + 1/A)
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
KNLe = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
KGe = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
Me_lumped = sympy.zeros(num_nodes*DOF, num_nodes*DOF)

ABDE = Matrix(
        [[A11, A12, A16, B11, B12, B16, 0, 0],
         [A12, A22, A26, B12, B22, B26, 0, 0],
         [A16, A26, A66, B16, B26, B66, 0, 0],
         [B11, B12, B16, D11, D12, D16, 0, 0],
         [B12, B22, B26, D12, D22, D26, 0, 0],
         [B16, B26, B66, D16, D26, D66, 0, 0],
         [0, 0, 0, 0, 0, 0, E44, E45],
         [0, 0, 0, 0, 0, 0, E45, E55]])

#NOTE this 2D library does not allow rotation from global to local coordinates
#     thus, we assume that u = ue
u = Matrix([symbols(r'u[%d]' % i) for i in range(0, BL.shape[1])])
print('Nxx =', (ABDE*(BL + BNL/2)*u)[0, 0])
print('Nyy =', (ABDE*(BL + BNL/2)*u)[1, 0])
print('Nxy =', (ABDE*(BL + BNL/2)*u)[2, 0])
print('Mxx =', (ABDE*(BL + BNL/2)*u)[3, 0])
print('Myy =', (ABDE*(BL + BNL/2)*u)[4, 0])
print('Mxy =', (ABDE*(BL + BNL/2)*u)[5, 0])
print('Qx =', (ABDE*(BL + BNL/2)*u)[6, 0])
print('Qy =', (ABDE*(BL + BNL/2)*u)[7, 0])
stress = Matrix([[Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qx, Qy]]).T
print('ux =', (Nux*u)[0, 0])
print('uy =', (Nuy*u)[0, 0])
print('vx =', (Nvx*u)[0, 0])
print('vy =', (Nvy*u)[0, 0])
print('wx =', (Nwx*u)[0, 0])
print('wy =', (Nwy*u)[0, 0])

sympy.var('wij, offset')

Ke[:, :] = wij*detJ*(BL.T*ABDE*BL + Bgamma.T*Egamma*Bgamma)

fint = wij*detJ*(BL.T + BNL.T)*stress

KNLe[:, :] = wij*detJ*(BL.T*ABDE*BNL + BNL.T*ABDE*BL + BNL.T*ABDE*BNL)

KGe[:, :] = wij*detJ*(
          Nxx*(Nux.T*Nux + Nvx.T*Nvx + Nwx.T*Nwx)
        + Nyy*(Nuy.T*Nuy + Nvy.T*Nvy + Nwy.T*Nwy)
        + Nxy*(Nux.T*Nuy + Nuy.T*Nux + Nvx.T*Nvy + Nvy.T*Nvx + Nwx.T*Nwy + Nwy.T*Nwx)
        )

Me[:, :] = wij*detJ*rho*(h*Nu.T*Nu + h*Nv.T*Nv + h*Nw.T*Nw +
    h*(h**2/12 + offset**2)*Nphix.T*Nphix +
    h*(h**2/12 + offset**2)*Nphiy.T*Nphiy)

BA = Nwx
KAe = wij*detJ*(Nw.T*BA)

calc_lumped = False

if calc_lumped:
    m1 = simplify(integrate(rho*h*detJfunc, (xi, -1, 0), (eta, -1, 0)))
    m2 = simplify(integrate(rho*h*detJfunc, (xi, 0, +1), (eta, -1, 0)))
    m3 = simplify(integrate(rho*h*detJfunc, (xi, 0, +1), (eta, 0, +1)))
    m4 = simplify(integrate(rho*h*detJfunc, (xi, -1, 0), (eta, 0, +1)))

    Iyy1 = simplify(integrate((xfunc-x1)**2*rho*h*detJfunc, (xi, -1, 0), (eta, -1, 0)))
    Iyy2 = simplify(integrate((xfunc-x2)**2*rho*h*detJfunc, (xi, 0, +1), (eta, -1, 0)))
    Iyy3 = simplify(integrate((xfunc-x3)**2*rho*h*detJfunc, (xi, 0, +1), (eta, 0, +1)))
    Iyy4 = simplify(integrate((xfunc-x4)**2*rho*h*detJfunc, (xi, -1, 0), (eta, 0, +1)))

    Ixx1 = simplify(integrate((yfunc-y1)**2*rho*h*detJfunc, (xi, -1, 0), (eta, -1, 0)))
    Ixx2 = simplify(integrate((yfunc-y2)**2*rho*h*detJfunc, (xi, 0, +1), (eta, -1, 0)))
    Ixx3 = simplify(integrate((yfunc-y3)**2*rho*h*detJfunc, (xi, 0, +1), (eta, 0, +1)))
    Ixx4 = simplify(integrate((yfunc-y4)**2*rho*h*detJfunc, (xi, -1, 0), (eta, 0, +1)))

    diag = (
            m1, m1, m1, Iyy1, Ixx1,
            m2, m2, m2, Iyy2, Ixx2,
            m3, m3, m3, Iyy3, Ixx3,
            m4, m4, m4, Iyy4, Ixx4
            )

    for i in range(Me.shape[0]):
        Me_lumped[i, i] = diag[i]

# K represents the global stiffness matrix
# in case we want to apply coordinate transformations
K = Ke
KNL = KNLe
KG = KGe
KA = KAe
M = Me
M_lumped = Me_lumped

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
for ind, val in np.ndenumerate(KNL):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    KNL[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', KNL[ind])

print()
for ind, val in np.ndenumerate(KG):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    KG[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', KG[ind])

print()
for i, fi in enumerate(fint):
    if fi == 0:
        continue
    si = name_ind(i)
    print('fint[%d + %s] +=' % (i%DOF, si), fi)

print()
for ind, val in np.ndenumerate(M):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M[ind])

print()
for ind, val in np.ndenumerate(KA):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    KA[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', KA[ind])

if calc_lumped:
    print()
    print('M_lumped')
    print()
    for ind, val in np.ndenumerate(M_lumped):
        if val == 0:
            continue
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=',
                M_lumped[ind])

