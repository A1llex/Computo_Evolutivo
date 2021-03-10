import numpy as np
import time

#no solo aleatorio si no tambien con semilla del tiempo
r = np.random.RandomState((int)(time.time()))

#solo numeros aleatorios
print(r.randint(19))

#arreglo de dimension 0 , entre 2 digitos(0,1) y que sean 10 numeros
print(r.randint(0,2,10))

#permutaciones de 0-19 en un arreglo
print(r.permutation(19))

#sera necesario tener la semilla