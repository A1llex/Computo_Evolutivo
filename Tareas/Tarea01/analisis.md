# Practica01
* Alex Gerardo Fernandez Aguilar 

## Analisis Problema de la mochila
Seleccione un Algoritmo voraz con un aheuristica basado en su valor por peso y ordenado en este rubro.

No es completo 

Su complejidad en espacio es lineal ya que no usa mas que el propio arreglo.
Su complejidad en tiempo , dado el ordenamiento es nlogn y dado que se busca por todo el arreglo si caben mas sube a lineal ya que solo se recorre una vez.

## Resultados 
1. la solucion es [[8, 4], [10, 5]] con un total de carga de 9 y un valor de 18 de un maximo de 11
   
2. la solucion es [[838, 767], [649, 595], [180, 167], [958, 889], [196, 182], [259, 242], [126, 119], [258, 244], [757, 718], [995, 945], [23, 22], [61, 62], [26, 28]] con un total de carga de 4980 y un valor de 5326 de un maximo 
de 5000

3. la solucion es [[92264, 83877], [76140, 69221], [109027, 99123], [177323, 161215], [69271, 62979], [179749, 163423], [61650, 56051], [108657, 98789], [181206, 164754], [41755, 37964], [327, 298], [437, 399], [706, 649], [244, 225], [945, 873], [87, 81], [74, 70], [8, 9]] con un total de carga de 1000000 y un valor de 1099870 de un maximo de 1000000

## Codigo
Solo es necesario correr el archivo Problema_mochila con python3 y se requirara un input para seleccionar el archivo de entrada