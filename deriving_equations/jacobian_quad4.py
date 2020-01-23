import sympy
from sympy.vector import CoordSys3D

sympy.var('xi, eta')
sympy.var('x1, y1, x2, y2, x3, y3, x4, y4', real=True, positive=True)

R = CoordSys3D('R')
r1 = x1*R.i + y1*R.j
r2 = x2*R.i + y2*R.j
r3 = x3*R.i + y3*R.j
r4 = x4*R.i + y4*R.j

rbottom = r1 + (r2 - r1)*(xi + 1)/2
rtop = r4 + (r3 - r4)*(xi + 1)/2
r = rbottom + (rtop - rbottom)*(eta + 1)/2

# d/dx = d/dxi*dxi/dx + d/deta*deta/dx = [dxi/dx   deta/dx] d/dxi
# d/dy   d/dxi*dxi/dy + d/deta*deta/dy   [dxi/dy   deta/dy] d/deta




