#Algoritmo de Busqueda Voraz
#Alex GerardoFernandez Aguilar 
import random

inpt =  int(input("elige del 1-3 para elegir entre tsp_5_1,tsp_51_1,tsp_85900_1 \n"))
if inpt == 1:
    file1 = open('tsp_5_1', 'r')
elif inpt == 2:
    file1 = open('tsp_51_1', 'r')
else :
    file1 = open('tsp_85900_1', 'r')

#leer la primer linea 
num_ciudades = int(file1.readline())
ciudades = []
count = 0
#leer y guardar infomacion del archivo
while True:
    line = file1.readline()
    if not line:
        break
    x , y = map(int, line.split())
    calidad = x/y
    ciudades.append((calidad,x,y))
    count += 1
file1.close()

"""
problema de la mochila Busqueda Voraz
"""

#gredy lo que haremos sera buscar para cada nodo cual es el mas cercano a el , depues se intentara trazar un ciclo
#de ser que no se pueda seguir una linea se intentara conectar quitando el que pierda menos cantidad


    
