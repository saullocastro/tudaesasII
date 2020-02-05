import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
ratio = Symbol('ratio', positive=True)
zeta = Symbol('zeta', positive=True)

# unknown function
u = Function('u')(t)

# assumed values
omegan = 5.
u0 = 1
v0 = 0
omegaf = omegan*ratio
f0 = 10.

# solving ODE
f = f0*sympy.cos(omegaf*t)
ics = {u.subs(t, 0): u0,
       u.diff(t).subs(t, 0): v0,
       }
sol = dsolve(u.diff(t, t) + 2*zeta*omegan*u.diff(t) + omegan**2*u - f, ics=ics)

import matplotlib
matplotlib.use('TkAgg')
from sympy.plotting import plot3d

subs = {zeta: 0.2}
p1 = plot3d(sol.rhs.subs(subs), (t, 0, 10), (ratio, 0.1, 2),
        show=False,
        nb_of_points_x=250,
        nb_of_points_y=250,
        xlabel='$t$',
        ylabel='$\omega_f/\omega_n$',
        zlabel='$u(t)$',
        )
p1.show()

