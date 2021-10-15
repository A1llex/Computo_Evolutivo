#Algoritmo de Busqueda Voraz
#Alex GerardoFernandez Aguilar 
import random

inpt =  int(input("elige del 1-3 para elegir entre ks_4_0,ks_50_1,ks_10000_0 \n"))
if inpt == 1:
    file1 = open('ks_4_0', 'r')
elif inpt == 2:
    file1 = open('ks_50_1', 'r')
else :
    file1 = open('ks_10000_0', 'r')

#leer la primer linea y dividirla
elementos, capacidad = map(int, file1.readline().split() ) 
articulos = []
#leer y guardar infomacion del archivo
while True:
    line = file1.readline()
    if not line:
        break
    valor , peso = map(int, line.split())
    calidad = valor/peso
    articulos.append((calidad,valor,peso))
file1.close()

"""
problema de la mochila Busqueda Voraz
"""
articulos.sort( key=lambda tup: (-tup[0],tup[2]) )

carga =0
precio = 0
solucion_voraz = []
indx = 0
for articulo in articulos:
    if carga+articulo[2] > capacidad:
        continue
    solucion_voraz.append([articulo[1],articulo[2]])
    carga+= articulo[2]
    precio+= articulo[1]

print(f"la solucion es {solucion_voraz} con un total de carga de {carga} y un valor de {precio} de un maximo de {capacidad}")



    
