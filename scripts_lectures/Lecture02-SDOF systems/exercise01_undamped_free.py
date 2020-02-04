from IPython.display import display
import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
m = Symbol('m', positive=True)
k = Symbol('k', positive=True)

# unknown function
u = Function('u')(t)

# solving ODE
sol = dsolve(m*u.diff(t, t) + k*u)

# sol.lhs ==> u(t)
# sol.rhs ==> solution
display(sol.rhs)

omegan = sympy.Symbol('\omega_n')
expr = sol.rhs.subs(sympy.sqrt(k/m), omegan)
display(expr)


