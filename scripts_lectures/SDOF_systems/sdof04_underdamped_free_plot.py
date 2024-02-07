import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
zeta = Symbol('\\zeta', constant=True, positive=True)

# unknown function
u = Function('u')(t)

# assumed values
u0 = 1
v0 = 0
omegan = 4.
wd = omegan*sympy.sqrt(1-zeta**2)

ics = {u.subs(t, 0): u0, u.diff(t).subs(t, 0): v0}
sol = dsolve(u.diff(t, t) + 2*zeta*omegan*u.diff(t) + omegan**2*u, ics=ics)

#import matplotlib
#matplotlib.use('TkAgg')
from sympy.plotting import plot3d
p1 = plot3d(sol.rhs, (t, 0, 10), (zeta, 0.05, 0.7),
    show=False,
    nb_of_points_x=500,
    nb_of_points_y=10,
    xlabel='$t$',
    ylabel='$\\zeta$',
    zlabel='$u(t)$',
    )
p1.show()









