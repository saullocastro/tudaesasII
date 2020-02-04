from IPython.display import display
import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
omegan = Symbol('\omega_n', positive=True)
zeta = Symbol('\zeta')
f0 = Symbol('f_0', positive=True)
wf = Symbol('\omega_f', positive=True)
u0 = Symbol('u_0', constant=True)
v0 = Symbol('v_0', constant=True)

# unknown function
u = Function('u')(t)

# solving ODE
f = f0*sympy.cos(wf*t)
ics = {u.subs(t, 0): u0,
       u.diff(t).subs(t, 0): v0,
       }
sol = dsolve(u.diff(t, t) + 2*zeta*omegan*u.diff(t) + omegan**2*u - f, ics=ics)

# sol.lhs ==> u(t)
# sol.rhs ==> solution
display(sol.rhs.simplify())

