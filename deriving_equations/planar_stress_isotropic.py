from IPython.display import display
import sympy

sympy.var('E')
nu = sympy.Symbol(r'\nu')
G = E/(2*(1 + nu))

S = sympy.Matrix([[1/E, -nu/E, 0, 0, 0],
                  [-nu/E, 1/E, 0, 0, 0],
                  [0, 0, 1/G, 0, 0],
                  [0, 0, 0, 1/G, 0],
                  [0, 0, 0, 0, 1/G]])
display(sympy.simplify(S.inv()))

sympy.var('z, h, d')

display(sympy.simplify(sympy.integrate(1, (z, -h/2+d, +h/2+d))))

