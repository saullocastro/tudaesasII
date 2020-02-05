from IPython.display import display
import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
zeta = Symbol('\zeta', positive=True)
omegan = Symbol('\omega_n', positive=True)
omegad = Symbol('\omega_d', positive=True)
epsilon = Symbol(r'\varepsilon', positive=True)
tn = Symbol('t_n', positive=True)
fi = Symbol('f_i')
m = Symbol('m', positive=True)
u0 = 0
v0 = 0

# unknown function
u = Function('u')(t)

# solving ODE (mass-normalized EOM)
f = fi*sympy.DiracDelta(t-tn)
ics = {u.subs(t, 0): u0,
       u.diff(t).subs(t, 0): v0,
       }
sol = dsolve(u.diff(t, t) + 2*zeta*omegan*u.diff(t) + omegan**2*u - f/m, ics=ics)
display(sympy.simplify(sol.rhs))

from sympy.plotting import plot
plot(sol.rhs.subs({omegan: 10, zeta: 0.1, tn: 3, fi: 1, m: 3}), (t, 0, 10),
     adaptive=False,
     nb_of_points=1000,
     ylabel='$u(t)$')

