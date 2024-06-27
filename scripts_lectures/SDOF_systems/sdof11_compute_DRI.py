import numpy as np


def compute_DRI(time, acc_z, zeta=0.224, omegan=52.9):
    def z_t(a, t1, t2, t):
        tn = (t1 + t2)/2
        dt = t2 - t1
        H = np.heaviside(t - tn, 1.)
        omegad = omegan*np.sqrt(1 - zeta**2)
        h = 1/(omegad)*np.sin(omegad*(t-tn))*np.exp(-zeta*omegan*(t-tn))*H
        return dt*a*h

    t = np.linspace(0, time.max(), 10000)
    z = np.zeros_like(t)

    for i, (t1, t2) in enumerate(zip(time[:-1], time[1:])):
        a = (acc_z[i] + acc_z[i+1])/2
        z += z_t(a, t1, t2, t)

    z_max = np.max(np.abs(z))

    DRI = omegan ** 2 * z_max / 9.81

    return DRI, t, z


time, acc_z = np.loadtxt('sdof11_compute_DRI_data.csv', delimiter=',',
                         skiprows=1, usecols=(1, 4), unpack=True)
DRI, t, z = compute_DRI(time, acc_z)
print(DRI)
