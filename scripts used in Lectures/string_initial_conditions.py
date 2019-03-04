import numpy as np
import sympy
sympy.var('A, An, T, rho')
t = sympy.Symbol('t')
L = 7
a = 3*L/4
n = sympy.Symbol('n', integer=True, positive=True)
x = sympy.Symbol('x', positive=True)

y0 = sympy.Piecewise((A*x/a, ((x>=0) & (x<=a))),
        (A*(L-x)/(L-a), ((x>a) & (x<=L))))
v0 = 0

omegan = n*sympy.pi*sympy.sqrt(T/(rho*L**2))
Yn = An*sympy.sin(n*sympy.pi*x/L)

exp = sympy.integrate(rho*Yn**2, (x, 0, L)) - 1
An = sympy.solve(exp, An)[1].simplify()

Yn = An*sympy.sin(n*sympy.pi*x/L)

eta0 = sympy.integrate(rho*Yn*y0, (x, 0, L)).simplify()

etan = eta0*sympy.cos(omegan*t)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

nmaxs = [1, 2, 3, 4, 5, 10, 20, 40]
for nmax in nmaxs:
    yxt = 0
    #yxt2 = 0
    for ni in range(1, nmax+1):
        yxt += (Yn*etan).subs(n, ni)
        #s = (1/n**2*sympy.sin(n*sympy.pi*a/L)*
            #sympy.sin(n*sympy.pi*x/L)*
            #sympy.cos(omegan*t))
        #yxt2 += 2*A*L**2/(sympy.pi**2*a*(L-a))*s.subs(n, ni)
    subs = dict(
            rho=7e3,
            T=100e2,
            A=0.01,
            )
    yxt = yxt.subs(subs)
    #yxt2 = yxt2.subs(subs)
    fyxt = sympy.lambdify('x, t', yxt, modules='numpy')
    #fyxt2 = sympy.lambdify('x, t', yxt2, modules='numpy')
    ts = np.linspace(0, 20, 1000)
    plt.plot(ts, fyxt(L/2, ts), label='$n_{max}=%d$' % nmax)
plt.legend()
plt.savefig('string_initial_conditions.png', bbox_inches='tight')

if True:
    fig = plt.gcf()
    ax = plt.gca()
    plt.cla()
    x = np.linspace(0, L, 500)
    ax.plot(x, x*0, '-k')
    ax.plot(x, x*0, '--k')
    lines = ax.get_lines()
    ts = np.linspace(0, 160, 1000)
    fps = 25

    def animate(i):
        global lines
        y = fyxt(x, ts[i])
        lines[1].set_data(*[x, y])
        return lines

    ani = FuncAnimation(fig, animate, range(len(ts)))
    ani.save('string_initial_conditions.gif', fps=fps)



