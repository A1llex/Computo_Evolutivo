
## Tarea 02
# Implementación de un Evolutivo
* Alex Gerardo Fernandez Aguilar 
### Componentes Asignados:
- Representacion: Real Entera
- Selección de padres:  Universal Estocástica
- Escalamiento: Ninguno
- Cruza: Aritmética Total
- Mutación: Uniforme
- Selección: Más

# Epacio de Busqueda
El espacio de busqueda con 5 decimales viendolo con la perspectiva de los reales enteros podemos pensar que se trata de un rango de enteros desde [-50000000,50000000], la forma de saber cuanto espacio de busqueda recorrimos seria saber en cuantos puntos del plano nos llegamos a mover

# Ejecución con parámetros mínimos 6 individuos y 2 generaciones 
se utilizo lo unico que se pudo variar fue el porcentaje de cruza que se utilizo pc = 0.9 y un porcentaje de mutacion pm = 0.2

    Se utilizo la semilla aleatoria  48
    ------------------------------------------------------------
    Generación: 0
    -PADRES
    Genotipo [[ 25120128  36903347]
    [ 21181649  20743492]
    [-34490149 -23041232]
    [ 26134208  12112710]
    [ -3681978   1413340]
    [ 41613392  40024172]]
    Fenotipo [[ 251.20128  369.03347]
    [ 211.81649  207.43492]
    [-344.90149 -230.41232]
    [ 261.34208  121.1271 ]
    [ -36.81978   14.1334 ]
    [ 416.13392  400.24172]]
    Padres Seleccionados [0, 1, 2, 3, 3, 4]
    -HIJOS
    Genotipo [[ 22415007.  22415007.]
    [ 35231831.  35231831.]
    [-20669904.  23807120.]
    [  9741382.   9741382.]
    [-24140801. -24140801.]
    [ 37666851.  37666851.]]
    Fenotipo [[ 224.15007  224.15007]
    [ 352.31831  352.31831]
    [-206.69904  238.0712 ]
    [  97.41382   97.41382]
    [-241.40801 -241.40801]
    [ 376.66851  376.66851]]
    Valores de cruza 3
    Valores de Mutacion 1
    ------------------------------------------------------------
    Generación: 1
    -PADRES
    Genotipo [[41613392. 40024172.]
    [37666851. 37666851.]
    [21181649. 20743492.]
    [22415007. 22415007.]
    [25120128. 36903347.]
    [-3681978.  1413340.]]
    Fenotipo [[416.13392 400.24172]
    [376.66851 376.66851]
    [211.81649 207.43492]
    [224.15007 224.15007]
    [251.20128 369.03347]
    [-36.81978  14.1334 ]]
    Padres Seleccionados [1, 2, 3, 4, 5, 5]
    -HIJOS
    Genotipo [[ 17194734.  17194734.]
    [ 41215608.  41215608.]
    [ 15504723.  15504723.]
    [ 29146036. -27072063.]
    [  1413340. -17111814.]
    [  1413340. -34848188.]]
    Fenotipo [[ 171.94734  171.94734]
    [ 412.15608  412.15608]
    [ 155.04723  155.04723]
    [ 291.46036 -270.72063]
    [  14.1334  -171.11814]
    [  14.1334  -348.48188]]
    Valores de cruza 3
    Valores de Mutacion 4
    Tabla de mejores aptitudes :
    [[55.54459503]
    [55.54459503]]
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    El Resultado del mejor individuo en base a su aptitud es el que tiene     
    Genotipo es [41215608. 41215608.]
    Fenotipo es [412.15608 412.15608]
    Y una Aptitud de 19.456390229874046
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

![a](figure_1.png "plot")

#   Resultados promediados de 20 ejecuciones del algoritmo
Utilice los siguientes parametros
> Numero de variables para el cromosoma
> * nvars = 2 

> Numero de poblacion
> * npop = 20

> limite inferior del espacio de busqueda
> * lb = -500

> Limite superior del espacio de busqueda
> * ub = 500

> Usaremos una Precision de 5 decimales
> * precision = 5

> porcentaje de Cruza
> * pc = 0.9

> Porcentaje de Mutacion
> * pm = 0.2

> Numero de Generaciones
> * ngen = 40

Despues de Generar las 20 ejecuciones me sorprendio el valor mas bajo que logro $2.546923133195378e-05$  es decir $.00002546923133195378$ esto lo encontro con el Genotipo  $[42096851. 42096851.]$ y Fenotipo es $[420.96851 420.96851]$ con la seed 69421 .lo que se pude observar en las graficas generadas era que quiza se necesitaban menos generacion quiza 35 aunque a veces cercana a esta generacion era cuando se acercaba mucho mas.

![a](figure_2.png "mejor aproximamiento")

Sobre la poblacion experimentado con esta no encontre grandes cambios , solo que entre mas poblacion era mas facil llegar a un resultado mejor , ya que usabamos la seleccion mas,

Sobre los porcentajes de cruza y de mutacion , el porcentaje de cruza entre mas alto era mas facil que pudiera salir de optimos locales si se apareaba con un individuo lejado o diferente.
sobre la mutacion tampoco pude observar grandes cambios al jugar con esta.

Como conlusion durante las varias ejcucuciones observe que siempre se acercaba a los valores de 420 en "x" y "y"
 