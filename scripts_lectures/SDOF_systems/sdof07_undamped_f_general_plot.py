import numpy as np
import matplotlib.pyplot as plt

def u_t(fn, m, k, t1, t2, t):
    tn = (t1 + t2)/2
    dt = t2 - t1
    H = np.heaviside(t - tn, 1.)
    omegan = np.sqrt(k/m)
    h = 1/(m*omegan)*np.sin(omegan*(t-tn))*H
    return fn*dt*h

def forcefunc(t):
    t = np.atleast_1d(t)
    f = np.zeros_like(t)
    check = t<1
    f[check] = 30*t[check]**2
    f[~check] = -30*(2-t[~check])**2
    f[t>2] = 0
    return f

tload = np.linspace(0, 2., 1000)
t = np.linspace(0, 10., 10000)

plt.plot(tload, forcefunc(tload), 'c')
plt.xlabel('$t$')
plt.ylabel('$f(t)$')
plt.xlim(0, tload.max())
plt.ylim(forcefunc(tload).min()*1.1, forcefunc(tload).max()*1.1)
if True:
    intervals = 20
    tapprox = np.linspace(tload.min(), tload.max(), intervals+1)
    plt.hlines(0, xmin=tload.min(), xmax=tload.max(), colors='k', linestyles='--')
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
for intervals in [2, 6, 10, 14]:
    tapprox = np.linspace(tload.min(), tload.max(), intervals+1)
    for t1, t2 in zip(tapprox[:-1], tapprox[1:]):
        tn = (t1 + t2)/2
        fn = forcefunc(tn)
        plt.plot((t1, t1), (0, fn), 'k--')
        plt.plot((t1, t2), (fn, fn), 'k--')
        plt.plot((t2, t2), (0, fn), 'k--')
        plt.plot(tload, forcefunc(tload), 'c', zorder=0)
        plt.xlabel('$t$')
        plt.ylabel('$f(t)$')
        plt.xlim(0, tload.max())
        plt.ylim(forcefunc(tload).min()*1.1, forcefunc(tload).max()*1.1)
    plt.hlines(0, xmin=tload.min(), xmax=tload.max(), colors='k', linestyles='--')
    plt.show()

    k = 30
    m = 2
    ans = 0
    t1s = []
    t2s = []
    for t1, t2 in zip(tapprox[:-1], tapprox[1:]):
        tn = (t1 + t2)/2
        ans += u_t(forcefunc(tn), m, k, t1, t2, t)
    ax.plot(t, ans, label='%d intervals' % intervals)

ax.set_xlabel('$t$')
ax.set_ylabel('$u(t)$')
ax.legend(loc='upper center')
ax.get_figure()
