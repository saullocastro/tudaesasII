import numpy as np
import sympy
from sympy import zeros, symbols, Matrix, simplify, integrate

sympy.var('E')
sympy.var('ux, v0x, v0xx, vx, vxx, z, h, width')
sympy.var('Nux, Nvx, Nvxx')
#u = u0 - z*vx
#u = u0 + z*beta
#v = v0
exx = (ux - z*vxx + z*v0xx) + (ux - z*vxx + z*v0xx)**2/2 + (vx)**2/2 - v0x*vx + v0x**2/2
#exx = (ux - z*vxx) + (ux**2 - 2*z*ux*vxx + z**2*vxx**2)/2 + (vx)**2/2 - v0x**2/2
Dexx = (Nux - z*Nvxx) + (ux - z*vxx + z*v0xx)*(Nux - z*Nvxx) + vx*Nvx - v0x*Nvx
D2exx = (Nux - z*Nvxx)*(Nux - z*Nvxx) + Nvx*Nvx

print('exx', exx.expand())

KNL_terms = sympy.integrate(width*E*Dexx*Dexx, (z, -h/2, h/2)).expand()
print('KNL terms', KNL_terms)

fint_terms = sympy.integrate(width*E*Dexx*exx, (z, -h/2, h/2)).expand()
print('fint terms', fint_terms)

Pxx_Mxx = sympy.integrate(width*E*exx, (z, -h/2, h/2)).expand()
print('Pxx_Mxx terms', Pxx_Mxx)

KG_terms = sympy.integrate(width*E*exx*D2exx, (z, -h/2, h/2)).expand()
print('KG terms', KG_terms)



