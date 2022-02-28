import numpy as np
import sympy
from sympy import Matrix, simplify, integrate
sympy.var('xi, le, E, cosr, sinr, rho, A1, A2, Izz1, Izz2, Ppreload')

A = A1 + (A2 - A1)*(xi - (-1))/(1 - (-1))
Izz = Izz1 + (Izz2 - Izz1)*(xi - (-1))/(1 - (-1))
Nu1 = (1-xi)/2
Nu2 = (1+xi)/2

R = Matrix([[  cosr,  sinr , 0, 0, 0, 0],
            [ -sinr,  cosr , 0, 0, 0, 0],
            [ 0,  0 , 1, 0, 0, 0],
            [ 0, 0, 0,  cosr,  sinr , 0],
            [ 0, 0, 0, -sinr,  cosr , 0],
            [ 0, 0, 0, 0,  0 , 1]])

ONE = sympy.Integer(1)
for leg_poly in [True, False]:
    if leg_poly:
        Nv1 = ONE/2 - 3*xi/4 + 1*xi**3/4
        Nb1 = le*(ONE/8 - 1*xi/8 - 1*xi**2/8 + 1*xi**3/8)
        Nv2 = ONE/2 + 3*xi/4 - 1*xi**3/4
        Nb2 = le*(-ONE/8 - 1*xi/8 + 1*xi**2/8 + 1*xi**3/8)
    else: # Hermitian cubic functions
        Nv1 = ONE/4*(1-xi)**2*(2+xi)
        Nb1 = le*1/8*(1-xi)**2*(1+xi)
        Nv2 = ONE/4*(1+xi)**2*(2-xi)
        Nb2 = le*1/8*(1+xi)**2*(xi-1)
    Nu = Matrix([[Nu1, 0,  0, Nu2,  0,  0]])
    Nv = Matrix([[0, Nv1, Nb1,  0, Nv2, Nb2]])
    Nbeta = -(2/le)*sympy.diff(Nv, xi)

    Nuxi = sympy.diff(Nu, xi)
    Nvxi = sympy.diff(Nv, xi)
    Nbetaxi = Nbeta.diff(xi)

    Ke = sympy.zeros(6, 6)
    KGe = sympy.zeros(6, 6)
    Me = sympy.zeros(6, 6)

    BL = sympy.Matrix([(2/le)*Nuxi,
                       (2/le)*Nbetaxi])
    print('BL (nodal displacements in global coordinates) =', BL*R)

    Ke[:, :] = (2/le)*E*Izz*Nbetaxi.T*Nbetaxi + (2/le)*E*A*Nuxi.T*Nuxi
    KGe[:, :] = (le/2)*Ppreload*(2/le)**2*(Nuxi.T*Nuxi + Nvxi.T*Nvxi)
    Me[:, :] = (le/2)*rho*(A*Nu.T*Nu + A*Nv.T*Nv + Izz*Nbeta.T*Nbeta)

    #NOTE procedure to compute lumped matrix when cross section changes
    x = (xi + 1)*le/2
    mA = integrate((le/2)*rho*A, (xi, -1, 0))
    mB = integrate((le/2)*rho*A, (xi, 0, +1))
    MMIA = integrate((le/2)*rho*(Izz + x**2*A), (xi, -1, 0))
    MMIB = integrate((le/2)*rho*(Izz + (le-x)**2*A), (xi, 0, +1))
    Me_lumped = Matrix([[mA, 0, 0, 0, 0, 0],
                        [0, mA, 0, 0, 0, 0],
                        [0, 0, MMIA, 0, 0, 0],
                        [0, 0, 0, mB, 0, 0],
                        [0, 0, 0, 0, mB, 0],
                        [0, 0, 0, 0, 0, MMIB]])

    # integrating matrices in natural coordinates

    for ind, val in np.ndenumerate(Ke):
        Ke[ind] = simplify(integrate(val, (xi, -1, 1)))

    for ind, val in np.ndenumerate(KGe):
        KGe[ind] = simplify(integrate(val, (xi, -1, 1)))

    for ind, val in np.ndenumerate(Me):
        Me[ind] = simplify(integrate(val, (xi, -1, 1)))

    Me_lumped = simplify(Me_lumped)

    print('printing Me')
    print(Me)
    print('printing Me_lumped')
    print(Me_lumped)

    K = simplify(R.T*Ke*R)

    if leg_poly:
        print('K integrated with Legendre polynomial')
    else:
        print('K integrated with Hermitian cubic polynomial')

    print('printing K for code')
    for ind, val in np.ndenumerate(K):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        K[%d+%s, %d+%s]' % (i%3, si, j%3, sj), '+=', K[ind])

    if leg_poly:
        print('M integrated with Legendre polynomial')
    else:
        print('M integrated with Hermitian polynomial')

    print('printing Me for assignment report')
    for ind, val in np.ndenumerate(Me):
        i, j = ind
        if val == 0:
            continue
        print('M_e[%d, %d] =' % (i+1, j+1), end='')
        sympy.print_latex(Me[ind].subs('le', 'l_e'))
        print(r'\\')

    print('printing Me_lumped for assignment report')
    for ind, val in np.ndenumerate(Me_lumped):
        i, j = ind
        if val == 0:
            continue
        print('M_e[%d, %d] =' % (i+1, j+1), end='')
        sympy.print_latex(Me_lumped[ind].subs('le', 'l_e'))
        print(r'\\')

    M = simplify(R.T*Me*R)
    M_lumped = simplify(R.T*Me_lumped*R)

    print('printing M for code')
    for ind, val in np.ndenumerate(M):
        if val == 0:
            continue
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        M[%d+%s, %d+%s]' % (i%3, si, j%3, sj), '+=', M[ind])

    print('printing M_lumped for code')
    for ind, val in np.ndenumerate(M_lumped):
        if val == 0:
            continue
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        M[%d+%s, %d+%s]' % (i%3, si, j%3, sj), '+=', M_lumped[ind])

    KG = simplify(R.T*KGe*R)

    print('printing KG for code')
    for ind, val in np.ndenumerate(KG):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        KG[%d+%s, %d+%s]' % (i%3, si, j%3, sj), '+=', KG[ind])

