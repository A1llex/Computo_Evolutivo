import numpy as np
#from numpy.lib.financial import pmt
#from numpy.lib.nanfunctions import _nanvar_dispatcher

"""
@author: Alex Gerardo Fernandez Aguilar
Algoritmo Evoulutivo para resolver 
f(x1,x2) = 418.9829*2 - x1*sin(sqrt(abs(x1))) - x2*sin(sqrt(abs(x2))) 
en un rango de en [-500, 500]
Se realizara con la siguiente seleccion de componentes
### Repartición de componentes
**Alex Fernández** : 
- Representacion: Real Entera
- Selección de padres:  Universal Estocástica
- Escalamiento: Ninguno
- Cruza: Aritmética Total
- Mutación: Uniforme
- Selección: Más
"""


def f(x1, x2):
    """
    f(x1,x2)
    Definicion de valucaion en esta funcion , esta tiene que minimizar el problema
    f(x1,x2) = 418.9829*2 - x1*sin(sqrt(abs(x1))) - x2*sin(sqrt(abs(x2))) 
    """
    return 418.9829*2 - x1 * np.sin(np.sqrt(np.abs(x1))) - x2*np.sin(np.sqrt(np.abs(x2)))


def inicializar(f, npop, nvars, lb, ub, precision):
    """Inicia con una poblacion aleatoria con un Genotipo Real Entero, 
    y un fenotipo ``precision`` """
    # Generamos el Genotipo desde el limite inferior hasta el superior multiplicados por la preciion deseada,
    #  randint regresa una distribucion uniforme
    genotipos = np.random.randint(
        low=lb*10**precision, high=ub*10**precision, size=[npop, nvars])
    # transformamos el genoripo a un Fenotipo dentro de los limites
    fenotipos = np.multiply(genotipos, 10**-precision)
    # Valuamos las aptitudes
    aptitudes = f(fenotipos)
    return genotipos, fenotipos, aptitudes


def seleccion_universal_estocastica(aptitudes, npop):
    """Funcion Estocastica para determinar ``npop`` padres de una fomra Universal Etocastica"""
    # Probabilidad
    p = aptitudes/sum(aptitudes)
    # Valor esperado
    vE = np.multiply(p, npop)
    # Numero uniformemente aleatorio
    ptr = np.random.uniform(0, 1)
    # Suma actual
    suma = 0
    # lista de padres
    parents = []
    for i in range(npop):
        suma += vE[i]
        for ptr in np.arange(ptr, suma, 1):
            parents.append(parents, i)
            ptr += 1
    return parents


def cruza_aritmetica_total(genotipos, idx, pc):
    hijos_genotipos = np.zeros(np.shape(genotipos))
    k = 0
    for i, j in zip(idx[::2], idx[1::2]):
        flip = np.random.uniform() <= pc
        if flip:
            punto_cruza = np.random.randint(0, len(genotipos[0]))
            hijos_genotipos[k] = np.concatenate(
                (genotipos[i, 0:punto_cruza], genotipos[j, punto_cruza:]))
            hijos_genotipos[k+1] = np.concatenate(
                (genotipos[j, 0:punto_cruza], genotipos[i, punto_cruza:]))
        else:
            hijos_genotipos[k] = np.copy(genotipos[i])
            hijos_genotipos[k + 1] = np.copy(genotipos[j])
        k += 2
    return hijos_genotipos


def mutacion_uniforme(genotipos, lb, ub, pm):
    for i in range(len(genotipos)):
        for j in range(len(genotipos[i])):
            flip = np.random.uniform() <= pm
            if flip:
                genotipos[i, j] = np.random.uniform(lb[j], ub[j])
    return genotipos


def seleccion_mas(npop, genotipos, fenotipos, aptitudes, hijos_genotipos, hijos_fenotipo, hijos_aptitudes):
    """Seleccion mas para la poblacion , esto es unir ambas poblaciones y elegir la mejor mitad deacuerdo a la aptitud"""
    # Lo que haremos sera juntar todos en una lista de 3-tuplas y ordenarlas por aptitud
    total = list(zip(genotipos, fenotipos, aptitudes)) + \
        list(zip(hijos_genotipos, hijos_fenotipo, hijos_aptitudes))
    total.sort(key=lambda x: x[2])
    total = total[:npop]
    gen, fen, apt = zip(*total)
    return gen, fen, apt


def estadisticas(generacion, genotipos, fenotipos, aptitudes, hijos_genotipos, hijos_fenotipo, hijos_aptitudes, padres):
    print('---------------------------------------------------------')
    print('Generación:', generacion)
    print('Población:\n', np.concatenate((np.arange(len(aptitudes)).reshape(-1, 1), genotipos,
                                          fenotipos, aptitudes.reshape(-1, 1), aptitudes.reshape(-1, 1)/np.sum(aptitudes)), 1))
    print('Padres:', padres)
    print('frecuencia de padres:', np.bincount(padres))
    print('Hijos:\n', np.concatenate((np.arange(len(aptitudes)).reshape(-1, 1), hijos_genotipos, hijos_fenotipo,
                                      hijos_aptitudes.reshape(-1, 1), hijos_aptitudes.reshape(-1, 1)/np.sum(hijos_aptitudes)), 1))
    print('Desempeño en línea para t=1: ', np.mean(aptitudes))
    print('Desempeño fuera de línea para t=1: ', np.max(aptitudes))
    print('Mejor individuo en la generación: ', np.argmax(aptitudes))


def EA(f, lb, ub, pc, pm, nvars, npop, ngen):
    # Inicializar
    bg = np.zeros((ngen, nvars))
    bf = np.zeros((ngen, nvars))
    ba = np.zeros((ngen, 1))
    genotipos, fenotipos, aptitudes = inicializar(f, npop, nvars, lb, ub)
    # Hasta condición de paro
    for i in range(ngen):
        # Selección de padres
        idx = seleccion_universal_estocastica(aptitudes, npop)
        # Cruza
        hijos_genotipos = cruza_aritmetica_total(genotipos, idx, pc)
        # Mutación
        hijos_genotipos = mutacion_uniforme(hijos_genotipos, lb, ub, pm)
        hijos_fenotipo = hijos_genotipos
        hijos_aptitudes = f(hijos_genotipos)

        # Estadisticas
        estadisticas(i, genotipos, fenotipos, aptitudes,
                     hijos_genotipos, hijos_fenotipo, hijos_aptitudes, idx)

        # Mejor individuo
        idx_best = np.argmax(aptitudes)
        b_gen = np.copy(genotipos[idx_best])
        b_fen = np.copy(fenotipos[idx_best])
        b_apt = np.copy(aptitudes[idx_best])
        ba[i] = np.copy(aptitudes[idx_best])

        # Selección de siguiente generación
        genotipos, fenotipos, aptitudes = seleccion_mas(npop,
                                                        genotipos, fenotipos, aptitudes, hijos_genotipos, hijos_fenotipo, hijos_aptitudes)

        # Elitismo
        idx = np.random.randint(npop)
        genotipos[idx] = b_gen
        fenotipos[idx] = b_fen
        aptitudes[idx] = b_apt
    # Fin ciclo

    print('Tabla de mejores:\n', ba)
    # Regresar mejor solución
    idx = np.argmax(aptitudes)
    return genotipos[idx], fenotipos[idx], aptitudes[idx]


# Random seed
seed = 21
np.random.seed(seed=seed)
# Numero de variables para el cromosoma
nvars = 2
# Numero de poblacion
npop = 200
# limite inferior del espacio de busqueda
lb = -500
# Limite superior del espacio de busqueda
ub = 500
# porcentaje de Cruza
pc = 0.9
# Porcentaje de Mutacion
pm = 0.01
# Numero de Generaciones
ngen = 500

np.set_printoptions(formatter={'float': '{0: 0.3f}'.format})
print(EA(f, lb, ub, pc, pm, nvars, npop, ngen))
