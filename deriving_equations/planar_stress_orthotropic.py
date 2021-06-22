#FIXME not checked yet
from IPython.display import display
import sympy

sympy.var('E11, E22, G12, G23, G13')
nu21 = sympy.Symbol(r'\nu21')
nu12 = sympy.Symbol(r'\nu12')

S = sympy.Matrix([[1/E11, -nu21/E22, 0, 0, 0],
                  [-nu12/E11, 1/E22, 0, 0, 0],
                  [0, 0, 1/G12, 0, 0],
                  [0, 0, 0, 1/G23, 0],
                  [0, 0, 0, 0, 1/G13]])
display(sympy.simplify(S.inv()))

sympy.var('z, h, d')

display(sympy.simplify(sympy.integrate(1, (z, -h/2+d, +h/2+d))))

