import numpy as np
import sympy

DOF = 2

def main():
    sympy.var('h, A')
    sympy.var('x1, y1, x2, y2, x3, y3')
    sympy.var('rho, E, nu')

    ONE = sympy.Integer(1)

    BL = sympy.Matrix(
            [[y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
             [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
             [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]])

    # Constitutive linear stiffness matrix
    num_nodes = 3
    Ke = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
    Me = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
    Melumped = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
    C = E/(1-nu**2)*sympy.Matrix(
            [[1, nu, 0],
             [nu, 1, 0],
             [0, 0, 1-nu]])

    Ke[:, :] = h/(4.*A)*(BL.T*C*BL)
    Me[:, :] = rho*h*A/12.*sympy.Matrix(
            [[2, 0, 1, 0, 1, 0],
             [0, 2, 0, 1, 0, 1],
             [1, 0, 2, 0, 1, 0],
             [0, 1, 0, 2, 0, 1],
             [1, 0, 1, 0, 2, 0],
             [0, 1, 0, 1, 0, 2]])
    Melumped[:, :] = rho*h*A/3*sympy.Matrix(
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
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('    K[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', K[ind])

    print('printing M')
    for ind, val in np.ndenumerate(M):
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M[ind])

    print('printing Mlumped')
    for ind, val in np.ndenumerate(Mlumped):
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('    Mlumped[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', Mlumped[ind])


if __name__ == '__main__':
    main()
