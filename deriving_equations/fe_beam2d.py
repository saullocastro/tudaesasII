import numpy as np
import sympy
from sympy import zeros, symbols, Matrix, simplify, integrate
sympy.var('wi, xi, le, E, cosr, sinr, rho, h1, h2, A1, A2, Izz1, Izz2, Ppreload')
sympy.var('Pxx, Mxx, ux, vx, vxx, v0x')

DOF = 3

Nu1 = (1-xi)/2
Nu2 = (1+xi)/2

R = Matrix([[  cosr,  sinr , 0, 0, 0, 0],
            [ -sinr,  cosr , 0, 0, 0, 0],
            [ 0,  0 , 1, 0, 0, 0],
            [ 0, 0, 0,  cosr,  sinr , 0],
            [ 0, 0, 0, -sinr,  cosr , 0],
            [ 0, 0, 0, 0,  0 , 1]])

ONE = sympy.Integer(1)
for leg_poly in [False, True]:
    A = A1 + (A2 - A1)*(xi - (-1))/(1 - (-1))
    h = h1 + (h2 - h1)*(xi - (-1))/(1 - (-1))
    Izz = Izz1 + (Izz2 - Izz1)*(xi - (-1))/(1 - (-1))
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
    Nvxixi = sympy.diff(Nv, xi, xi)
    Nbetaxi = Nbeta.diff(xi)

    Nux = (2/le)*Nuxi
    Nvx = (2/le)*Nvxi
    Nvxx = (2/le)*(2/le)*Nvxixi

    Ke = zeros(2*DOF, 2*DOF)
    KNLe = zeros(2*DOF, 2*DOF)
    KGconste = zeros(2*DOF, 2*DOF)
    KGe = zeros(2*DOF, 2*DOF)
    Me = sympy.zeros(2*DOF, 2*DOF)

    ZERO = 0*Nuxi
    BL = Matrix([(2/le)*Nuxi, (2/le)*Nbetaxi])
    BNL = Matrix([vx*(2/le)*Nvxi + ux*(2/le)*Nuxi, ZERO])
    BNL0 = Matrix([v0x*(2/le)*Nvxi, ZERO])

    print('BL (nodal displacements in global coordinates) =', BL*R)
    ucon = Matrix([symbols(r'ucon[%d]' % i) for i in range(0, BL.shape[1])])
    u0con = Matrix([symbols(r'u0con[%d]' % i) for i in range(0, BL.shape[1])])
    print('ux =', Nux*R*ucon)
    print('vx =', Nvx*R*ucon)
    print('vxx =', Nvxx*R*ucon)
    print('v0x =', Nvx*R*u0con)

    #print('exx =', ((BL + BNL/2 + BNL0)*R*ucon)[0, 0])
    #print('Pxx =', (E*A*(BL + BNL/2 + BNL0)*R*ucon)[0, 0])
    #print('Mxx =', (E*Izz*(BL + BNL/2 + BNL0)*R*ucon)[1, 0])

    # Pxx_Mxx terms E*h**3*vxx**2*width/24 + E*h*ux**2*width/2 + E*h*ux*width + E*h*vx**2*width/2
    #print('Pxx attempt =', ((E*Izz*vxx/2*Nvxx + E*A*(ux*Nux/2 + Nux + vx*Nvx/2))*R*ucon)[0, 0])

    Ke[:, :] = (2/le)*E*Izz*Nbetaxi.T*Nbetaxi + (2/le)*E*A*Nuxi.T*Nuxi

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

    KGconste[:, :] = (le/2)*Ppreload*(2/le)**2*(Nuxi.T*Nuxi + Nvxi.T*Nvxi)


    #KGe[:, :] = (le/2)*wi*Pxx*(2/le)**2*(Nuxi.T*Nuxi + Nvxi.T*Nvxi)

    sympy.var('A, h, Izz') # redefining these as variables

# KG terms (
#         + E*Nux**2*h**3*vxx**2*width/24
#         + E*Nux**2*h*ux**2*width/2
#         + E*Nux**2*h*ux*width
#         + E*Nux**2*h*vx**2*width/2
#         + E*Nux*Nvxx*h**3*ux*vxx*width/6
#         + E*Nux*Nvxx*h**3*vxx*width/6
#         + E*Nvx**2*h**3*vxx**2*width/24
#         + E*Nvx**2*h*ux**2*width/2
#         + E*Nvx**2*h*ux*width
#         + E*Nvx**2*h*vx**2*width/2
#         + E*Nvxx**2*h**5*vxx**2*width/160
#         + E*Nvxx**2*h**3*ux**2*width/24
#         + E*Nvxx**2*h**3*ux*width/12
#         + E*Nvxx**2*h**3*vx**2*width/24

    KGe[:, :] = (le/2)*wi*(
                    Nux.T*Nux*(
                            E*Izz*vxx**2/2
                          + E*A*ux**2/2
                          + E*A*ux
                          + E*A*vx**2/2
                              )
                + (Nux.T*Nvxx + Nvxx.T*Nux)*(
                            E*Izz*ux*vxx
                          + E*Izz*vxx
                              )
                  + Nvx.T*Nvx*(
                            E*Izz*vxx**2/2
                          + E*A*ux**2/2
                          + E*A*ux
                          + E*A*vx**2/2
                               )
                  + Nvxx.T*Nvxx*(
                            E**h**4*vxx**2*A/160
                          + E**Izz*ux**2/2
                          + E**Izz*ux
                          + E**Izz*vx**2/2
                           )
                )


# fint terms (
# + E*Nux*h**3*ux*vxx**2*width/8
# + E*Nux*h**3*vxx**2*width/8
# + E*Nux*h*ux**3*width/2
# + 3*E*Nux*h*ux**2*width/2
# + E*Nux*h*ux*vx**2*width/2
# + E*Nux*h*ux*width
# + E*Nux*h*vx**2*width/2
# + E*Nvx*h**3*vx*vxx**2*width/24
# + E*Nvx*h*ux**2*vx*width/2
# + E*Nvx*h*ux*vx*width
# + E*Nvx*h*vx**3*width/2
# + E*Nvxx*h**5*vxx**3*width/160
# + E*Nvxx*h**3*ux**2*vxx*width/8
# + E*Nvxx*h**3*ux*vxx*width/4
# + E*Nvxx*h**3*vx**2*vxx*width/24
# + E*Nvxx*h**3*vxx*width/12

# )

    finte = (le/2)*wi*(
          12*E*Nux*Izz*ux*vxx**2/8 +
          12*E*Nux*Izz*vxx**2/8 +
             E*Nux*A*ux**3/2 +
             3*E*Nux*A*ux**2/2 +
             E*Nux*A*ux*vx**2/2 +
             E*Nux*A*ux +
             E*Nux*A*vx**2/2 +
             E*Nvx*Izz*vx*vxx**2/2 +
             E*Nvx*A*ux**2*vx/2 +
             E*Nvx*A*ux*vx +
             E*Nvx*A*vx**3/2 +
             E*Nvxx*h**4*vxx**3*A/160 +
          12*E*Nvxx*Izz*ux**2*vxx/8 +
          12*E*Nvxx*Izz*ux*vxx/4 +
             E*Nvxx*Izz*vx**2*vxx/2 +
             E*Nvxx*Izz*vxx
    ).T

# KNL terms (
#            E*Nux**2*h**3*vxx**2*width/12
#            E*Nux**2*h*ux**2*width
#          2*E*Nux**2*h*ux*width
#            E*Nux**2*h*width
#          2*E*Nux*Nvx*h*ux*vx*width
#          2*E*Nux*Nvx*h*vx*width
#            E*Nux*Nvxx*h**3*ux*vxx*width/3
#            E*Nux*Nvxx*h**3*vxx*width/3
#            E*Nvx**2*h*vx**2*width
#            E*Nvx*Nvxx*h**3*vx*vxx*width/6
#            E*Nvxx**2*h**5*vxx**2*width/80
#            E*Nvxx**2*h**3*ux**2*width/12
#            E*Nvxx**2*h**3*ux*width/6
#            E*Nvxx**2*h**3*width/12

    KNLe[:, :] = (le/2)*wi*(
                E*A*(Nux.T*Nux)
              + 2*E*A*ux*(Nux.T*Nux)
              + E*A*ux**2*(Nux.T*Nux)
              + E*A*vx*(Nux.T*Nvx + Nvx.T*Nux)
              + E*A*vx**2*(Nvx.T*Nvx)
              + E*A*ux*vx*(Nux.T*Nvx + Nvx.T*Nux)

              + E*Izz*(Nvxx.T*Nvxx)
              + 2*E*Izz*ux*(Nvxx.T*Nvxx)
              + 2*E*Izz*vxx*(Nux.T*Nvxx + Nvxx.T*Nux)
              + E*Izz*ux**2*(Nvxx.T*Nvxx)
              + E*Izz*vxx**2*(Nux.T*Nux)
              + 2*E*Izz*ux*vxx*(Nux.T*Nvxx + Nvxx.T*Nux)
              + E*Izz*vx*vxx*(Nvx.T*Nvxx + Nvxx.T*Nvx)

              + E*h**4*vxx**2*A/80*(Nvxx.T*Nvxx)
    )

    # integrating matrices in natural coordinates

    for ind, val in np.ndenumerate(Ke):
        Ke[ind] = simplify(integrate(val, (xi, -1, 1)))

    for ind, val in np.ndenumerate(KNLe):
        #NOTE to be integrated numerically
        KNLe[ind] = val

    for ind, val in np.ndenumerate(KGconste):
        KGconste[ind] = simplify(integrate(val, (xi, -1, 1)))

    for ind, val in np.ndenumerate(KGe):
        #NOTE to be integrated numerically
        KGe[ind] = val

    for ind, val in np.ndenumerate(Me):
        Me[ind] = simplify(integrate(val, (xi, -1, 1)))

    Me_lumped = simplify(Me_lumped)

    print('printing Me')
    print(Me)
    print('printing Me_lumped')
    print(Me_lumped)

    K = simplify(R.T*Ke*R)
    KNL = simplify(R.T*KNLe*R)
    fint = R.T*finte

    if leg_poly:
        print('K integrated with Legendre polynomial')
    else:
        print('K integrated with Hermitian cubic polynomial')

    print('printing K for code')
    for ind, val in np.ndenumerate(K):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        K[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', K[ind])

    if leg_poly:
        print('KNL integrated with Legendre polynomial')
    else:
        print('KNL integrated with Hermitian cubic polynomial')

    print('printing KNL for code')
    for ind, val in np.ndenumerate(KNL):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        KNL[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', sympy.simplify(KNL[ind]))

    if leg_poly:
        print('fint integrated with Legendre polynomial')
    else:
        print('fint integrated with Hermitian cubic polynomial')

    print('printing fint for code')
    for i, fi in enumerate(fint):
        if fi == 0:
            continue
        si = 'c1' if i < 3 else 'c2'
        print('fint[%d + %s] +=' % (i%DOF, si), sympy.simplify(fi))

    if leg_poly:
        print('KG integrated with Legendre polynomial')
    else:
        print('KG integrated with Hermitian cubic polynomial')

    KGconst = simplify(R.T*KGconste*R)

    print('printing KGconst for code')
    for ind, val in np.ndenumerate(KGconst):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        KG[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', KGconst[ind])

    KG = simplify(R.T*KGe*R)

    print('printing KG for code')
    for ind, val in np.ndenumerate(KG):
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        KG[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', sympy.simplify(KG[ind]))

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
        print('        M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M[ind])

    print('printing M_lumped for code')
    for ind, val in np.ndenumerate(M_lumped):
        if val == 0:
            continue
        i, j = ind
        si = 'c1' if i < 3 else 'c2'
        sj = 'c1' if j < 3 else 'c2'
        print('        M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M_lumped[ind])

