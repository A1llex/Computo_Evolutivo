#descenso del gradiente
from problema01 import *

def N(f,xk, ns,sigma):
    #N(xk,sigma,ns)
    n = xk +np.random.normal(sigma,(ns,len(xk)))
    fn=[]
    for i in range(ns):
        fn.append(f(n[i]))
    return n,fn

def hilclimber(f,x0,N,ns,maxiter,sigma):
    xk =x0
    fxk = f(xk)

    for i in range(maxiter):
        n, fn = N(f,xk,ns)
        bn_idx = np.argmin(fn)     #argumento minimo
        if fn[bn_idx] <= fxk:
            xk = n[bn_idx]
            fxk = fn[bn_idx]
        #else:
            #break
        sigma *= 3/5
        print(i,xk , fxk ,sigma)
    return xk ,fxk



#f= spheres_f
f = schwefel_f()
#x0 = np.array([5.0])
x0 = np.array([np.random.uniform((-500,500))])
ns = 10
maxiter = 100
#para reajustarse
sigma = 5

print ('resultado ',hilclimber(f,x0,N,ns,maxiter,sigma) )