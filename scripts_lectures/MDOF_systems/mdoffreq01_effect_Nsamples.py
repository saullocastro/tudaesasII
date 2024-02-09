import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [12, 6]

def f_t_func(t):
    return 13*cos(1*(2*pi)*t) + 5*cos(7*(2*pi)*t)

def H(zeta, omegan, omegaf):
    """NOTE: all frequencies must be in rad/s
    """
    return ((omegan**2 - omegaf**2) - 1j*(2*zeta*omegan*omegaf))/((omegan**2 - omegaf**2)**2 + (2*zeta*omegan*omegaf)**2)

tmax = 8
Nsamples = 100
print(Nsamples)
t = np.linspace(0, tmax, Nsamples)
t_plot = np.linspace(0, tmax, 100000)
dt = t[1] - t[0]

f_t = f_t_func(t) # [N/kg]
zeta = 0.02
omegan = 3 * (2*pi) # [rad/s]

size_fig = (10, 6)
plt.figure(figsize=size_fig)
plt.plot(t_plot, f_t_func(t_plot), 'b-', lw=1)
plt.xlabel('Time, [$s$]')
plt.ylabel('normalized force $f(t)$ [$N/kg$]')
plt.xlim(0, 2)
plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
plt.show()

plt.figure(figsize=size_fig)
plt.plot(t, f_t, 'o')
plt.plot(t_plot, f_t_func(t_plot), 'b-', lw=1)
plt.xlabel('Time, [$s$]')
plt.ylabel('normalized force $f(t)$ [$N/kg$]')
plt.xlim(0, 2)
plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
plt.show()

f_f = np.fft.rfft(f_t)[:Nsamples//2]/(Nsamples//2)
freqs_Hz = np.fft.fftfreq(Nsamples, dt)[:Nsamples//2]
freqs_rad = freqs_Hz*(2*pi)

plt.figure(figsize=size_fig)
plt.stem(freqs_Hz, np.abs(f_f))
plt.xlim(0, 10)
plt.xlabel('Frequency, [Hz]')
plt.ylabel('normalized force $f(\\omega)$ [$N/kg$]')
plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
plt.show()

u_f = H(zeta, omegan, freqs_rad)*f_f
u_t = np.fft.irfft(u_f*(Nsamples//2), n=Nsamples)
plt.figure(figsize=size_fig)
plt.plot(t, u_t, 'o--', lw=0.5, mfc='None')#, mew=0.04)
plt.xlim(0, 2)
plt.ylim(-0.065, 0.065)
plt.xlabel('Time, [$s$]')
plt.ylabel('displacement [$m$]')
plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
plt.show()
