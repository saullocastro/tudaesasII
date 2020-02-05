from IPython.display import display
import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
k = Symbol('k', positive=True)
m = Symbol('m', positive=True)
tn = Symbol('t_n', positive=True)
fi = Symbol('f_i', positive=True)
u0 = Symbol('u_0')
v0 = Symbol('v_0')
omegan = Symbol('\omega_n')

# unknown function
u = Function('u')(t)

# solving ODE
f = fi*sympy.DiracDelta(t - tn)
ics = {u.subs(t, 0): u0,
       u.diff(t).subs(t, 0): v0,
       }
sol = dsolve(m*u.diff(t, t) + k*u - f, ics=ics)
display(sympy.simplify(sol.rhs))

#expr = sympy.sqrt(k/m)
#display(sympy.simplify(sol.rhs.expand().subs({expr: omegan})))

#expr2 = sympy.sqrt(k*m)
#expr3 = 1/expr
#display(sympy.simplify(sol.rhs.expand().subs({expr: omegan, expr2: m*omegan, expr3: 1/omegan})))

