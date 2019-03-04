import numpy as np
import sympy
from sympy import cos, sin, sqrt, pi

sympy.var('A, An, T, rho')
t = sympy.Symbol('t')
n = sympy.Symbol('n', integer=True, positive=True)
x = sympy.Symbol('x', positive=True)
L = 7
a = 3*L/4
y0 = sympy.Piecewise((A*x/a, ((x>=0) & (x<=a))),
        (A*(L-x)/(L-a), ((x>a) & (x<=L))))
Yn = An*sympy.sin(n*pi*x/L)
exp = sympy.integrate(rho*Yn**2, (x, 0, L)) - 1
An = sympy.solve(exp, An)[1].simplify()
Yn = An*sin(n*pi*x/L)

# Finding Modal Force
F0 = 1.
Omega = 0.99*1*pi*sqrt(T/(rho*L**2))
Fx = F0
Fn = sympy.integrate(Fx*Yn, (x, 0, L))

# Finding modal initial conditions
eta0 = sympy.integrate(rho*Yn*y0, (x, 0, L)).simplify()
omegan = n*pi*sqrt(T/(rho*L**2))
k1 = eta0 - Fn/(omegan**2 - Omega**2)
etan = Fn/(omegan**2 - Omega**2)*cos(Omega*t) + k1*cos(omegan*t)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.cla()
nmaxs = [1, 2, 3, 4, 5, 10, 40]
for nmax in nmaxs:
    yxt = 0
    for ni in range(1, nmax+1):
        yxt += (Yn*etan).subs(n, ni)
    subs = dict(
            rho=7e3,
            T=100e2,
            A=0.01,
            )
    yxt = yxt.subs(subs)
    fyxt = sympy.lambdify('x, t', yxt, modules='numpy')
    ts = np.linspace(0, 10, 100)
    plt.plot(ts, fyxt(L/2, ts), label='$n_{max}=%d$' % nmax)
plt.legend()
plt.savefig('string_harmonic_excitation_convergence.png', bbox_inches='tight')

plt.cla()
nmax = 40
yxt = 0
for ni in range(1, nmax+1):
    yxt += (Yn*etan).subs(n, ni)
subs = dict(
        rho=7e3,
        T=100e2,
        A=0.01,
        )
yxt = yxt.subs(subs)
fyxt = sympy.lambdify('x, t', yxt, modules='numpy')
ts = np.linspace(0, 1200, 10000)
plt.plot(ts, fyxt(L/2, ts), label='$n_{max}=%d$' % nmax)
plt.legend()
plt.savefig('string_harmonic_excitation.png', bbox_inches='tight')

if True:
    fig = plt.gcf()
    ax = plt.gca()
    plt.cla()
    x = np.linspace(0, L, 500)
    ax.plot(x, x*0, '-k')
    ax.plot(x, x*0, '--k')
    lines = ax.get_lines()
    ts = np.linspace(0, 1200, 2000)
    fps = 25

    def animate(i):
        global lines
        y = fyxt(x, ts[i])
        lines[1].set_data(*[x, y])
        return lines

    ani = FuncAnimation(fig, animate, range(len(ts)))
    ani.save('string_harmonic_excitation.gif', fps=fps)



