from IPython.display import display
import sympy

sympy.var('offset, rho, h, z, z1, z2')

display(sympy.simplify(sympy.integrate(1, (z, -h/2+offset, h/2+offset))))

display(sympy.simplify(sympy.integrate(1, (z, z1, z2))))

display(sympy.simplify(sympy.integrate(z**2, (z, -h/2+offset, h/2+offset))))

display(sympy.simplify(sympy.integrate(z**2, (z, z1, z2))))
