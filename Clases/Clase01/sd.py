#descenso del gradiente
from problema01 import *

#entradas alpha en r+ tolerancia r+
#mas cosas
def sd(f,j,x0 , alpha , tol ,maxiter):
    # pk = negativo del gradiente  = -j(x)
    # x_{k+1} = x_k +alpha *pk
    # medir perdida 
    #verificar optimalidad norm( j(x)  ) <= 1e-4 (tolerancia) ~ j(x)=0


    #f(x) =x^2
    #j(x) 2+x

    #f2(x)= x_1 ^2 + x_2 ^2
    # j(x) = [2_x1 , 2x_2 ]^T

    i=0

    xk = x0

    while maxiter > i and  np.linalg.norm(xk) > tol:

        pk = -(xk)

        xk = xk+alpha * pk

        print (i, xk, f(xk) , np.linalg.norm(j(xk)))
         
        i+=1

    return xk,f(xk)

f= spheres_f
j = spheres_f
x0 = np.array([5.0])
alpha = 0.1
tol = 1e-4
maxiter = 10

print ('resultado ',sd(f,j,x0 , alpha , tol , maxiter) )


