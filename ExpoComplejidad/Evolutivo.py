import numpy as np
import matplotlib.pyplot as plt

"""
Algoritmo Evoulutivo para resolver 
Max Sat
### Componentes
"""


def f(x):
    """
    valuacion de desempeño de cierto individuo
    x[0-7]
    12 clausulas

    19*( ! x[2] && x[8] && x[4] ) ||
    11*( x[3] && x[6] ) ||
    22*( ! x[0] && ! x[4] ) ||
     5*( ! x[2] && x[5] && ! x[0] && x[7] && x[1] && x[6] ) ||
    30*( x[4] && x[1] && x[6] && x[0]) ||
    11*( ! x[1] && x[3] ) ||
     1*( x[0] && x[5] && x[6] )  ||
     2*( x[7] && ! x[0] ) ||
    17*( x[6] && ! x[7] ) ||
     7*( x[2] && x[3] && x[5] ) ||
    16*( x[0] && !x[5] && ! x[3] )
    """

    resul = np.zeros(len(x))
    for i  in range(len(x)):
        resul[i] = resul[i] = 19*((not x[i][2]) and (x[i][7]) and (x[i][4])) + \
        11*((x[i][3]) and (x[i][6])) + \
        22*((not x[i][0]) and (not x[i][4])) + \
        5*((not x[i][2]) and (x[i][5]) and (not x[i][0]) and (x[i][7]) and (x[i][1]) and (x[i][6])) + \
        30*((x[i][4]) and (x[i][1]) and (x[i][6]) and (x[i][0])) + \
        11*((not x[i][1]) and (x[i][3])) + \
        1*((x[i][0]) and (x[i][5]) and (x[i][6])) + \
        2*((x[i][7]) and (not x[i][0])) + \
        17*((x[i][6]) and (not x[i][7])) +  \
        7*((x[i][2]) and (x[i][3]) and (x[i][5])) +  \
        16*((x[i][0]) and (not x[i][5]) and (not x[i][3]))
    return resul


def inicializar(f, npop, nvars):
    """Inicia con una poblacion aleatoria """
    # Generamos el Genotipo desde el limite inferior hasta el superior multiplicados por la preciion deseada,
    #  randint regresa una distribucion uniforme
    genotipos = np.random.randint(2, size=[npop, nvars])
    # transformamos el genoripo a un Fenotipo dentro de los limites
    fenotipos = genotipos
    # Valuamos las aptitudes
    aptitudes = f(fenotipos)
    return genotipos, fenotipos, aptitudes


def seleccion_ruleta(aptitudes, npop):
    """Funcion Estocastica para determinar ``npop`` padres de una fomra Universal Etocastica"""
    # Probabilidad
    # Es necesaria que como minimizamos la aptitud entre menor sera sera mejor por esto hay que
    p = aptitudes/sum(aptitudes)
    # Valor Acumulado
    cp= np.cumsum(p)
    padres = np.zeros(npop)
    #genearar aleatorio
    for i in range(npop):
        X = np.random.uniform()
        #seleccionando padre
        padres[i] = np.argwhere(cp > X)[0]
    return padres.astype(int)


def cruza_de_un_punto(genotipos, padres, pc):
    """Cruza de un punto"""
    hijos_genotipos = np.zeros(np.shape(genotipos))
    k = 0
    cruzas = 0
    for i, j in zip(padres[::2], padres[1::2]):
        if np.random.uniform() <= pc:
            cruzas += 1
            punto_cruza = np.random.randint(0, len(genotipos[0]))
            hijos_genotipos[k] = np.concatenate(
                (genotipos[i, 0:punto_cruza], genotipos[j, punto_cruza:]))
            hijos_genotipos[k+1] = np.concatenate(
                (genotipos[j, 0:punto_cruza], genotipos[i, punto_cruza:]))
        else:
            hijos_genotipos[k] = np.copy(genotipos[i])
            hijos_genotipos[k + 1] = np.copy(genotipos[j])
        k += 2
    # si habia un numero impar de padres , el ultimo se incluira para que sea el mismo numero de padres que de hijos
    if(k < len(padres)):
        hijos_genotipos[k] = padres[-1]
    return hijos_genotipos, cruzas


def mutacion_inversion_de_un_bit(genotipos,  pm):
    """Mutacion inversion de un bit de un cromosoma"""
    mutaciones = 0
    for i in range(len(genotipos)):
        for j in range(len(genotipos[i])):
            if np.random.uniform() <= pm:
                mutaciones += 1
                genotipos[i, j] = 0 if(genotipos[i, j] == 1) else 1
    return genotipos, mutaciones


def seleccion_mas(npop, genotipos, fenotipos, aptitudes, hijos_genotipos, hijos_fenotipos, hijos_aptitudes):
    """Seleccion mas para la poblacion , esto es unir ambas poblaciones y elegir la mejor mitad deacuerdo a la aptitud"""
    # Lo que haremos sera juntar todos en una lista de 3-tuplas y ordenarlas por aptitud
    total = list(zip(genotipos, fenotipos, aptitudes)) + \
        list(zip(hijos_genotipos, hijos_fenotipos, hijos_aptitudes))
    total.sort(key=lambda x: -x[2])
    total = total[:npop]
    gen, fen, apt = zip(*total)
    return np.array(gen), np.array(fen), np.array(apt)


def estadistica(generacion, genotipos, fenotipos, aptitudes, hijos_genotipos, hijos_fenotipos, hijos_aptitudes, padres, mutaciones, cruzas):
    """Estadisticas, regresa [aptMin , aptMed ,aptMax,desvEst]
    No hay que olvidar que lo que estamos haciendo es minimizar por eso buscamos el mas pequeño"""
    print('------------------------------------------------------------')
    print('Generación:', generacion)
    aptMax = np.argmax(aptitudes)
    aptMed = np.mean(aptitudes)
    aptMin = np.argmin(aptitudes)
    desvEst = np.std(aptitudes)
    mediana = np.median(aptitudes)
    print(f"Mejor individuo\
        \nIndice del mejor individuo: {aptMax} \
        \nGenotipo: {genotipos[aptMax]} \
        \nFenotipo: {fenotipos[aptMax]} \
        \nAptitud: {aptitudes[aptMax]}")
    print(f"Aptitud Maxima {aptitudes[aptMax]} \
        \nAptitud Media {aptMed} \
        \nAptitud Minima {aptitudes[aptMin]}\
        \nAptitud Mediana {mediana}")
    print("Padres Seleccionados", padres)
    print('Frecuencia de padres seleccionados:', np.bincount(padres))
    print(f"Cruzas Efectuadas {cruzas} \
        \nMutaciones Efectuadas {mutaciones}")
    return [aptMin, aptMed, aptMax, desvEst, mediana]


def grafica(estadisticas):
    """Reportar gráfica de convergencia. 
    Eje x número de generaciones, 
    eje y mediana de la mejor aptitud de cada generación"""
    plt.plot(range(len(estadisticas)), list(zip(*estadisticas))[4], marker="o")
    plt.xlabel("Generaciones")
    plt.ylabel("Mediana de  Aptitudes")
    plt.title("Grafica de Convergencia")
    plt.show()

def EA(f, pc, pm, nvars, npop, ngen):
    """Algoritmo Evolutivo para resolver """
    # Inicializar
    estadisticas = []
    ba = np.zeros((ngen, 1))
    genotipos, fenotipos, aptitudes = inicializar(
        f, npop, nvars)
    # Hasta condición de paro
    for i in range(ngen):
        mutaciones = 0
        cruzas = 0
        # Selección de padres
        padres = seleccion_ruleta(aptitudes, npop)
        # Cruza
        hijos_genotipos, cruzas = cruza_de_un_punto(
            genotipos, padres, pc)
        # Mutación
        hijos_genotipos, mutaciones = mutacion_inversion_de_un_bit(
            hijos_genotipos, pm)
        hijos_fenotipos = hijos_genotipos
        hijos_aptitudes = f(hijos_fenotipos)

        # Estadistica
        estadisticas.append(
            estadistica(i, genotipos, fenotipos, aptitudes, hijos_genotipos, hijos_fenotipos, hijos_aptitudes, padres, mutaciones, cruzas))

        # Mejor individuo
        ba[i] = np.copy(aptitudes[estadisticas[i][2]])

        # Selección de siguiente generación
        genotipos, fenotipos, aptitudes = seleccion_mas(npop,
                                                        genotipos, fenotipos, aptitudes, hijos_genotipos, hijos_fenotipos, hijos_aptitudes)

    print('Tabla de mejores aptitudes :\n', ba.reshape(-1, 1))
    grafica(estadisticas)
    # Regresar mejor solución
    padres = np.argmin(aptitudes)
    return genotipos[padres], fenotipos[padres], aptitudes[padres]


# Random seed
seed = np.random.randint(100000)
np.random.seed(seed=seed)
# Numero de variables para el cromosoma
nvars = 8
# Numero de poblacion
npop = 10
# porcentaje de Cruza
pc = 0.8
# Porcentaje de Mutacion
pm = 0.2
# Numero de Generaciones
ngen = 10

# modificaremos el formato para que  no aparezca en forma exponencial
np.set_printoptions(suppress=True)
bgen, bfen, bapt = EA(f,  pc, pm, nvars, npop, ngen)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Se utilizo la semilla aleatoria ", seed)
print(f"El Resultado del mejor individuo en base a su aptitud es el que tiene \
    \nGenotipo es {bgen}  \
    \nFenotipo es {bfen}  \
    \nY una Aptitud de {bapt}")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
