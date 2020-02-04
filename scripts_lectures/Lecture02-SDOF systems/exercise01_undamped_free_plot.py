import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)

# unknown function
u = Function('u')(t)

# solving ODE with initial conditions
u0 = 0.4
v0 = 2
k = 150
m = 2
ics = {u.subs(t, 0): u0, u.diff(t).subs(t, 0): v0}

sol = dsolve(m*u.diff(t, t) + k*u, ics=ics)

#import matplotlib
#matplotlib.use('TkAgg')
from sympy.plotting import plot
p1 = plot(sol.rhs, (t, 0, 1), xlabel='$t$', ylabel='$u(t)$')
