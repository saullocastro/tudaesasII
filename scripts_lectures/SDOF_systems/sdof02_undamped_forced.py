from IPython.display import display
import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
m = Symbol('m', positive=True)
k = Symbol('k', positive=True)
F0 = Symbol('F_0', positive=True)
omegaf = Symbol('\omega_f', positive=True)

# unknown function
u = Function('u')(t)

# solving ODE
F = F0*sympy.sin(omegaf*t)
sol = dsolve(m*u.diff(t, t) + k*u - F)

# sol.lhs ==> u(t)
# sol.rhs ==> solution
display(sol.rhs)

omegan = sympy.Symbol('\omega_n')
expr = sol.rhs.subs(sympy.sqrt(k/m), omegan)
display(expr)


