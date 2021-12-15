import numpy as np
import sympy as sp
from sympy import *
from sympy.core.function import Derivative

x,y = sp.symbols('x y')

def gradient(f):
    return (f.diff(x), f.diff(y))

f = x**2+y**2
g = gradient(f)
print(g)
print("Diferentation",f.diff(x)+f.diff(y))
print("Ojo",g[1](2))
#print(g(10,10))



def Gradd(f, x0 ,y0 , stepp):
    #Metodo de nweton de una sola iteracion
    graddd = dF(x0 ,y0)
    print(graddd)
    x1 = x0 - stepp * graddd
    y1 = y0 - stepp * graddd
    return x1 ,y1

def F(x,y):
    #return 418.9829*2 - x*np.sin(np.sqrt(abs(x))) - y*np.sin(np.sqrt(abs(y)))
    return x**2 +y**2

def dF(x,y):
    return 2*x+2*y



x0 = -10
y0 = -10

xf , yf = Gradd(F, x0 ,y0, .1)

print('x0: ', x0,y0)
print("f(x0) = ", F(x0 , y0))
print('xf: ', xf,yf)
print("f(xf) = ", F(xf ,yf))