from sympy import *
init_printing(use_unicode=False)
x = Symbol('x')
integrate(exp(cos(x)) * sin(x)**4, (x, 0, pi))