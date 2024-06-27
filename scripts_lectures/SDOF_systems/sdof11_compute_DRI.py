import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_DRI(time, acc_z, zeta=0.224, omegan=52.9):
    def u_t(a, t1, t2, t):
        tn = (t1 + t2)/2
        dt = t2 - t1
        H = np.heaviside(t - tn, 1.)
        omegad = omegan*np.sqrt(1 - zeta**2)
        h = 1/(omegad)*np.sin(omegad*(t-tn))*np.exp(-zeta*omegan*(t-tn))*H
        return dt*a*h

    t = np.linspace(0, time.max(), 10000)
    y = np.zeros_like(t)

    for i, (t1, t2) in enumerate(zip(time[:-1], time[1:])):
        a = (acc_z[i] + acc_z[i+1])/2
        y += u_t(a, t1, t2, t)
    y_max = np.max(np.abs(y))
    DRI = omegan ** 2 * y_max / 9.81

    return DRI, t, y


time, acc_z = np.loadtxt('compute_DRI_data.csv', delimiter=',',
                         skiprows=1, usecols=(1, 4), unpack=True)
DRI, t, y = compute_DRI(time, acc_z)
print(DRI)
