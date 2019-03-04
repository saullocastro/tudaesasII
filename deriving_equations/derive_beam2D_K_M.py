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
for leg_poly in [True]:
    if leg_poly:
        Nv1 = sympy.Integer(1)/2 - 3*xi/4 + 1*xi**3/4
        Nv2 = le*(sympy.Integer(1)/8 - 1*xi/8 - 1*xi**2/8 + 1*xi**3/8)
        Nv3 = sympy.Integer(1)/2 + 3*xi/4 - 1*xi**3/4
        Nv4 = le*(-sympy.Integer(1)/8 - 1*xi/8 + 1*xi**2/8 + 1*xi**3/8)
    else: # Hermitian cubic functions
        Nv1 = 1/4*(1-xi)**2*(2+xi)
        Nv2 = le*1/8*(1-xi)**2*(1+xi)
        Nv3 = 1/4*(1+xi)**2*(2-xi)
        Nv4 = le*1/8*(1+xi)**2*(xi-1)
    Nu = sympy.Matrix([[Nu1, 0,  0, Nu2,  0,  0]])
    Nv = sympy.Matrix([[0, Nv1, Nv2,  0, Nv3, Nv4]])
    Nuxi = sympy.diff(Nu, xi)
    Nvxixi = sympy.diff(Nv, xi, xi)

    Ke = sympy.zeros(6, 6)
    Me = sympy.zeros(6, 6)

    Ke[:, :] = ((2/le)**4)*(le/2)*E*Izz*Nvxixi.T*Nvxixi + ((2/le)**2)*(le/2)*E*A*Nuxi.T*Nuxi
    Me[:, :] = (le/sympy.Integer(2))*A*rho*(Nu.T*Nu + Nv.T*Nv)

    for ind, val in np.ndenumerate(Ke):
        Ke[ind] = sympy.simplify(sympy.integrate(val, (xi, -1, 1)))

    for ind, val in np.ndenumerate(Me):
        Me[ind] = sympy.simplify(sympy.integrate(val, (xi, -1, 1)))

    if leg_poly:
        print('K integrated with Legendre polynomial')
    else:
        print('K integrated with Hermitian polynomial')
    K = R.T*Ke*R
    for ind, val in np.ndenumerate(K):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('    K[%d+%s, %d+%s]' % (i%3, si, j%3, sj), '+=', K[ind])

    if leg_poly:
        print('M integrated with Legendre polynomial')
    else:
        print('M integrated with Hermitian polynomial')
    M = R.T*Me*R
    for ind, val in np.ndenumerate(M):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('    M[%d+%s, %d+%s]' % (i%3, si, j%3, sj), '+=', M[ind])

