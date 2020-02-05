import numpy as np
import matplotlib.pyplot as plt

def u_t(fn, zeta, m, k, t1, t2, t):
    H1 = np.heaviside(t1 - tn, 1.)
    H2 = np.heaviside(t2 - tn, 1.)
    omegan = np.sqrt(k/m)
    omegad = omegan*np.sqrt(1 - zeta**2)
    h = 1/(m*omegad)*np.sin(omegad*(t-tn))*np.exp(-zeta*omegan*(t-tn))*(H1 - H2)
    return fn*(t2 - t1)*h

def forcefunc(t):
    t = np.atleast_1d(t)
    f = np.zeros_like(t)
    check = t<1
    f[check] = 30*t[check]**2
    f[~check] = -30*(2-t[~check])**2
    return f

t = np.linspace(0, 2., 10000)

plt.plot(t, forcefunc(t), 'c')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.xlim(0, t.max())
plt.ylim(forcefunc(t).min()*1.1, forcefunc(t).max()*1.1)
if True:
    intervals = 20
    tapprox = np.linspace(t.min(), t.max(), intervals+1)
    plt.hlines(0, xmin=t.min(), xmax=t.max(), colors='k', linestyles='--')
    for t1, t2 in zip(tapprox[:-1], tapprox[1:]):
        tn = (t1 + t2)/2
        fn = forcefunc(tn)
        plt.plot((t1, t1), (0, fn), 'k--')
        plt.plot((t1, t2), (fn, fn), 'k--')
        plt.plot((t2, t2), (0, fn), 'k--')
plt.show()

plt.clf()
ax = plt.gca()

plt.figure()
for intervals in [4, 6, 8, 10, 50]:
    tapprox = np.linspace(t.min(), t.max(), intervals+1)
    for t1, t2 in zip(tapprox[:-1], tapprox[1:]):
        tn = (t1 + t2)/2
        fn = forcefunc(tn)
        plt.plot((t1, t1), (0, fn), 'k--')
        plt.plot((t1, t2), (fn, fn), 'k--')
        plt.plot((t2, t2), (0, fn), 'k--')
        plt.plot(t, forcefunc(t), 'c', zorder=0)
        plt.xlabel('$t$')
        plt.ylabel('$f(t)$')
        plt.xlim(0, t.max())
        plt.ylim(forcefunc(t).min()*1.1, forcefunc(t).max()*1.1)
    plt.hlines(0, xmin=t.min(), xmax=t.max(), colors='k', linestyles='--')
    plt.show()

    k = 30
    m = 2
    zeta = 0.1
    ans = 0
    t1s = []
    t2s = []
    for t1, t2 in zip(tapprox[:-1], tapprox[1:]):
        tn = (t1 + t2)/2
        ans += u_t(forcefunc(tn), zeta, m, k, t1, t2, t)
    ax.plot(t, ans, label='%d intervals' % intervals)

ax.set_xlabel('$t$')
ax.set_ylabel('$u(t)$')
ax.legend()
ax.get_figure()
