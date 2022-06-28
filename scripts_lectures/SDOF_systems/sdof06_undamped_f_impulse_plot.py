import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
k = Symbol('k', positive=True)
m = Symbol('m', positive=True)
tn = Symbol('t_n', positive=True)
P0 = Symbol('P_0', positive=True)
u0 = Symbol('u_0')
v0 = Symbol('v_0')

# unknown function
u = Function('u')(t)

# solving ODE
f = P0*sympy.DiracDelta(t - tn)
ics = {u.subs(t, 0): u0,
       u.diff(t).subs(t, 0): v0,
       }
sol = dsolve(m*u.diff(t, t) + k*u - f, ics=ics)

#import matplotlib
#matplotlib.use('TkAgg')
from sympy.plotting import plot

res = sol.rhs.subs({
k: 15.,
m: 3.,
u0: 0,
v0: 0,
P0: 1,
tn: 1.5,
})

p1 = plot(res, (t, 0, 5), xlabel='$t$', ylabel='$u(t)$', nb_of_points_x=10000)

