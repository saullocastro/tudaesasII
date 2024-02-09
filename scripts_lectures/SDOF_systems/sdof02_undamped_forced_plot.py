import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
wf = Symbol('wf', positive=True)

# unknown function
u = Function('u')(t)

# solving ODE with initial conditions
u0 = 0.4
v0 = 2
k = 150
m = 2
F0 = 10
wn = sympy.sqrt(k/m)
#wf = 2*sympy.sqrt(k/m)
F = F0*sympy.sin(wf*t)
ics = {u.subs(t, 0): u0, u.diff(t).subs(t, 0): v0}

sol = dsolve(m*u.diff(t, t) + k*u - F, ics=ics)

#import matplotlib
#matplotlib.use('TkAgg')
from sympy.plotting import plot3d
p1 = plot3d(sol.rhs, (t, 0, 10), (wf, 0.8*wn, 0.99*wn),
        xlabel='$t$', ylabel='$\\omega_f$', zlabel='$u(t)$',
        nb_of_points_x=250, nb_of_points_y=25)
