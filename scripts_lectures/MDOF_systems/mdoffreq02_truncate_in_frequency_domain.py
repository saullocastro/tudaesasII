import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib

def f_t_func(t):
    t1 = 0.3
    t2 = 0.7
    tfmax = 0.5
    fmax = 10
    f_t = np.zeros_like(t)
    cond1 = (t >= t1) & (t <= tfmax)
    f_t[cond1] = fmax*(t[cond1] - t1)/(tfmax - t1)
    cond2 = (t > tfmax) & (t <= t2)
    f_t[cond2] = fmax*(1 - (t[cond2] - tfmax)/(t2 - tfmax))
    return f_t

def H(zeta, omegan, omegaf):
    """NOTE: all frequencies must be in rad/s
    """
    return ((omegan**2 - omegaf**2) - 1j*(2*zeta*omegan*omegaf))/((omegan**2 - omegaf**2)**2 + (2*zeta*omegan*omegaf)**2)

tmax = 1
Nsamples = 2**12
truncate = False
truncation_freq_Hz = 20

print(Nsamples)
t = np.linspace(0, tmax, Nsamples)
t_plot = np.linspace(0, tmax, 100000)
dt = t[1] - t[0]

f_t = f_t_func(t) # [N/kg]
zeta = 0.02
omegan = 10 * (2*pi) # [rad/s]

size_figs = (10, 6)
plt.figure(figsize=size_figs)
plt.plot(t, f_t, 'o')
plt.plot(t_plot, f_t_func(t_plot), 'b-', lw=1)
plt.xlabel('Time, [$s$]')
plt.ylabel('normalized force $f(t)$ [$N/kg$]')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

f_f = np.fft.rfft(f_t)[:Nsamples//2]/(Nsamples//2)
freqs_Hz = np.fft.fftfreq(Nsamples, dt)[:Nsamples//2]
freqs_rad = freqs_Hz*(2*pi)
if truncate:
    f_f[freqs_Hz > truncation_freq_Hz] = 0

plt.figure(figsize=size_figs)
plt.stem(freqs_Hz, f_f.real)
plt.xlim(0, 20)
plt.xlabel('Frequency, [Hz]')
plt.ylabel('normalized force $f(\\omega)$ [$N/kg$]')
plt.tight_layout()
plt.show()

u_f = H(zeta, omegan, freqs_rad)*f_f
plt.figure(figsize=size_figs)
plt.stem(freqs_Hz, u_f.real)
plt.xlim(0, 20)
plt.xlabel('Frequency, [Hz]')
plt.ylabel('displacement $u(\\omega)$ [$m$]')
plt.tight_layout()
plt.show()

u_t = np.fft.irfft(u_f*(Nsamples//2), n=Nsamples)
plt.figure(figsize=size_figs)
plt.plot(t, u_t, 'o', lw=0.5, mfc='None', mew=0.03)
plt.xlim(0, 1)
plt.ylim(-0.0001, 0.003)
plt.xlabel('Time, [$s$]')
plt.ylabel('displacement $u(t)$ [$m$]')
plt.tight_layout()
plt.show()

