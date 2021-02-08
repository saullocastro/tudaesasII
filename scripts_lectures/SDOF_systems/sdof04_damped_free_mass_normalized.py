from IPython.display import display
import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
omegan = Symbol('\omega_n', positive=True)
zeta = Symbol('\zeta')

# unknown function
u = Function('u')(t)

# solving ODE
sol = dsolve(u.diff(t, t) + 2*zeta*omegan*u.diff(t) + omegan**2*u)

# sol.lhs ==> u(t)
# sol.rhs ==> solution
display(sol.rhs.simplify())

