import numpy as np
import sympy
sympy.var('xi, le, E, cosr, sinr, rho, A1, A2, Izz1, Izz2')

A = A1 + (A2 - A1)*(xi - (-1))/(1 - (-1))
Izz = Izz1 + (Izz2 - Izz1)*(xi - (-1))/(1 - (-1))
Nu1 = (1-xi)/2
Nu2 = (1+xi)/2

R = sympy.Matrix([[  cosr,  sinr , 0, 0, 0, 0],
                  [ -sinr,  cosr , 0, 0, 0, 0],
                  [ 0,  0 , 1, 0, 0, 0],
                  [ 0, 0, 0,  cosr,  sinr , 0],
                  [ 0, 0, 0, -sinr,  cosr , 0],
                  [ 0, 0, 0, 0,  0 , 1]])

#for leg_poly in [False, True]:
ONE = sympy.Integer(1)
for leg_poly in [True, False]:
    if leg_poly:
        Nv1 = ONE/2 - 3*xi/4 + 1*xi**3/4
        Nv2 = le*(ONE/8 - 1*xi/8 - 1*xi**2/8 + 1*xi**3/8)
        Nv3 = ONE/2 + 3*xi/4 - 1*xi**3/4
        Nv4 = le*(-ONE/8 - 1*xi/8 + 1*xi**2/8 + 1*xi**3/8)
    else: # Hermitian cubic functions
        Nv1 = ONE/4*(1-xi)**2*(2+xi)
        Nv2 = le*1/8*(1-xi)**2*(1+xi)
        Nv3 = ONE/4*(1+xi)**2*(2-xi)
        Nv4 = le*1/8*(1+xi)**2*(xi-1)
    Nu = sympy.Matrix([[Nu1, 0,  0, Nu2,  0,  0]])
    Nv = sympy.Matrix([[0, Nv1, Nv2,  0, Nv3, Nv4]])
    Nbeta = -(2/le)*sympy.diff(Nv, xi)

    Nuxi = sympy.diff(Nu, xi)
    Nbetaxi = Nbeta.diff(xi)

    Ke = sympy.zeros(6, 6)
    Me = sympy.zeros(6, 6)

    Ke[:, :] = (2/le)*E*Izz*Nbetaxi.T*Nbetaxi + (2/le)*E*A*Nuxi.T*Nuxi
    Me[:, :] = (le/2)*rho*(A*Nu.T*Nu + A*Nv.T*Nv + Izz*Nbeta.T*Nbeta)

    for ind, val in np.ndenumerate(Ke):
        Ke[ind] = sympy.simplify(sympy.integrate(val, (xi, -1, 1)))

    for ind, val in np.ndenumerate(Me):
        Me[ind] = sympy.simplify(sympy.integrate(val, (xi, -1, 1)))

    if leg_poly:
        print('K integrated with Legendre polynomial')
    else:
        print('K integrated with Hermitian cubic polynomial')
    K = R.T*Ke*R
    for ind, val in np.ndenumerate(K):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        K[%d+%s, %d+%s]' % (i%3, si, j%3, sj), '+=', K[ind])

    if leg_poly:
        print('M integrated with Legendre polynomial')
    else:
        print('M integrated with Hermitian polynomial')
    M = R.T*Me*R
    for ind, val in np.ndenumerate(M):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        M[%d+%s, %d+%s]' % (i%3, si, j%3, sj), '+=', M[ind])

