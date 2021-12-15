import numpy as np
from numpy.core.defchararray import split
from sympy import *    


x, y ,z = symbols('x y z')
f = x**2 + y**2   # this is x SymPy expression, not x function
grad_f = lambdify((x, y ), derive_by_array(f, (x, y )))   # lambda from an expression for graduent    
print("Derivada",derive_by_array(f, (x, y)))

epsilon = 0.02
alpha = 0.1      # the value of 0.1 is too large for convergence here
#vector = Matrix([10, 10])  
nvars = 2
vector = np.array ([20, 20] )
print ("Vector",vector)

for i in range(10):    #  to avoid infinite loop
    #grad_value = Matrix(grad_f(vector[0], vector[1]))
    grad_value =-alpha* np.array(grad_f( vector[0],vector[1] ))
    print("Gradiente",grad_value)
    #vector += -alpha*grad_value
    vector = np.sum([vector,grad_value], axis=0)
    print("Vector",vector)
       
print("Resultado",vector)

