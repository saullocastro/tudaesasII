from numpy import tan, cos, sin, linspace
from scipy.optimize import root

def fun(x):
    return tan(x) + 2*x

def jac(x):
    return 1/cos(x)**2 + 2

sols = set()
for x0 in linspace(0, 1000, 1e6):
    ans = root(fun, [x0], jac=jac, method='hybr')
    sols.add(ans.x[0])
print(sorted(list(sols)))

