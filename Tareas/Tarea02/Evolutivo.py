import numpy as np
import matplotlib.pyplot as plt

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


def f(x):
    """
    f(x1,x2)
    Definicion de valucaion en esta funcion , esta tiene que minimizar el problema
    f(x1,x2) = 418.9829*2 - x1*sin(sqrt(abs(x1))) - x2*sin(sqrt(abs(x2))) 
    """
    return 418.9829*2 + np.sum(- x*np.sin(np.sqrt(np.abs( x ))) , axis = 1)


def fenotipo(genotipos, precision):
    """Funcion que regresa el fenotipo de un Genotipo con la presicion deseada"""
    return np.multiply(genotipos, 10**-precision)


def inicializar(f, npop, nvars, lb, ub, precision):
    """Inicia con una poblacion aleatoria con un Genotipo Real Entero, 
    y un fenotipo con ``precision`` """
    # Generamos el Genotipo desde el limite inferior hasta el superior multiplicados por la preciion deseada,
    #  randint regresa una distribucion uniforme
    genotipos = np.random.randint(
        low=lb*10**precision, high=ub*10**precision, size=[npop, nvars])
    # transformamos el genoripo a un Fenotipo dentro de los limites
    fenotipos = fenotipo(genotipos, precision)
    # Valuamos las aptitudes
    aptitudes = f(fenotipos)
    return genotipos, fenotipos, aptitudes


def seleccion_universal_estocastica(aptitudes, npop):
    """Funcion Estocastica para determinar ``npop`` padres de una fomra Universal Etocastica"""
    # Probabilidad
    # Es necesaria que como minimizamos la aptitud entre menor sera sera mejor por esto hay que 
    g = 1/aptitudes
    p = g/sum(g)
    print("aptitud",aptitudes)
    print("1/apt",g)
    print("prob",p)
    # Valor Esperado
    vE = np.multiply(p, npop)
    # Numero uniformemente aleatorio
    ptr = np.random.uniform(0, 1)
    # Suma actual
    suma = 0
    # lista de padres
    padres = []
    for i in range(npop):
        suma += vE[i]
        for ptr in np.arange(ptr, suma, 1):
            padres.append(i)
            ptr += 1
    return padres


def a(lb, ub, vk, wk, precision):
    """
    [max(α, β), min(γ, δ)] si v_k > w_k
    \n|0,0| si v_k = w_k (esto es que ambos valores se quedaran igual)
    \n[max(γ, δ), min(α, β)] si v_k < w_k
    """
    if (vk == wk):
        return vk, wk
    llb = lb * 10**precision
    uub = ub * 10 ** precision
    alpha = (llb-wk) / (vk - wk)
    beta = (uub-vk) / (wk - vk)
    gamma = (llb-vk) / (wk - vk)
    delta = (uub-wk) / (vk - wk)
    if(vk > wk):
        l = max(alpha, beta)
        u = min(gamma, delta)
    else:
        l = max(gamma, delta)
        u = min(alpha, beta)
    a = np.random.uniform(low=l, high=u)
    # Se transforman a int para que sean parte del genotipo
    v = int(a*wk + (1-a)*vk)
    w = int(a*vk + (1-a)*wk)
    return v, w


def cruza_aritmetica_total(lb, ub, genotipos, padres, pc, precision):
    """Cruza Artimetica Total, si es acpetada su cruza , se aplicara para cada alelo la cruza de la funcion de a que se determina
    \n[max(α, β), min(γ, δ)] si v_k > w_k
    \n|0,0| si v_k = w_k (esto es que ambos valores se quedaran igual)
    \n[max(γ, δ), min(α, β)] si v_k < w_k"""
    hijos_genotipos = np.zeros(np.shape(genotipos))
    k = 0
    cruzas = 0
    for i, j in zip(padres[::2], padres[1::2]):
        if np.random.uniform() <= pc:
            cruzas += 1
            for vk, wk in zip(genotipos[i], genotipos[j]):
                v, w = a(lb, ub, vk, wk, precision)
                hijos_genotipos[k] = v
                hijos_genotipos[k+1] = w
        else:
            hijos_genotipos[k] = np.copy(genotipos[i])
            hijos_genotipos[k + 1] = np.copy(genotipos[j])
        k += 2
    # si habia un numero impar de padres , el ultimo se incluira para que sea el mismo numero de padres que de hijos
    if(k < len(padres)):
        hijos_genotipos[k] = padres[-1]
    return hijos_genotipos, cruzas


def mutacion_uniforme(genotipos, lb, ub, pm, precision):
    """Mutacion Uniforme,lo que haremos sera elegir un numero aleatorio entre nuestros limites y sustituirlo en una posicion seleccionada"""
    mutaciones = 0
    for i in range(len(genotipos)):
        for j in range(len(genotipos[i])):
            if np.random.uniform() <= pm:
                mutaciones += 1
                genotipos[i, j] = np.random.randint(
                    low=lb*10**precision, high=ub*10**precision)
    return genotipos, mutaciones


def seleccion_mas(npop, genotipos, fenotipos, aptitudes, hijos_genotipos, hijos_fenotipos, hijos_aptitudes):
    """Seleccion mas para la poblacion , esto es unir ambas poblaciones y elegir la mejor mitad deacuerdo a la aptitud"""
    # Lo que haremos sera juntar todos en una lista de 3-tuplas y ordenarlas por aptitud
    total = list(zip(genotipos, fenotipos, aptitudes)) + \
        list(zip(hijos_genotipos, hijos_fenotipos, hijos_aptitudes))
    total.sort(key=lambda x: x[2])
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
    print("La poblaicon y los hijos seran presentados como : \n[[indice, genotipo , fenotipo, aptitudes  , probabilidad],...]")
    #print('Población:\n', np.concatenate((np.arange(len(aptitudes)).reshape(-1,1), genotipos, fenotipos, aptitudes.reshape(-1, 1), aptitudes.reshape(-1, 1)/np.sum(aptitudes)), 1))
    print(f"Mejor individuo\
        \nIndice: {aptMin} \
        \nGenotipo: {genotipos[aptMin]} \
        \nFenotipo: {fenotipos[aptMin]} \
        \nAptitud: {aptitudes[aptMin]}")
    print(f"Aptitud Maxima {aptitudes[aptMax]} \
        \nAptitud Media {aptMed} \
        \nAptitud Minima {aptitudes[aptMin]}\
        \nAptitud Mediana {mediana}")
    print("Desviacion Estandar", desvEst)
    print("Padres Seleccionados", padres)
    print('Frecuencia de padres seleccionados:', np.bincount(padres))
    print(f"Cruzas Efectuadas {cruzas} \
        \nMutaciones Efectuadas {mutaciones}")
    #print('Hijos:\n', np.concatenate(( np.arange(len(aptitudes)).reshape(-1, 1) , hijos_genotipos, hijos_fenotipos, hijos_aptitudes.reshape(-1, 1), hijos_aptitudes.reshape(-1, 1)/np.sum(hijos_aptitudes)), 1))
    # Informacion Requerida para ejecucion de 6 individuos y 2 generaciones.
    # print(f"-PADRES \
    #     \nGenotipo {genotipos} \
    #     \nFenotipo {fenotipos} \
    #     \nPadres Seleccionados {padres} ")
    # print(f"-HIJOS\
    #     \nGenotipo {hijos_genotipos} \
    #     \nFenotipo {hijos_fenotipos} \
    #     \nValores de cruza {cruzas} \
    #     \nValores de Mutacion {mutaciones}")
    
    return [aptMin, aptMed, aptMax, desvEst, mediana]


def grafica(estadisticas):
    """Reportar gráfica de convergencia. 
    Eje x número de generaciones, 
    eje y mediana de la mejor aptitud de cada generación"""
    # Plot
    # plt.xlim(0,ngen)
    # plt.ylim(0.8,1.4)

    # Plot
    plt.plot(range(len(estadisticas)), list(zip(*estadisticas))[4], marker="o")

    plt.xlabel("Generaciones")
    plt.ylabel("Mediana de  Aptitudes")
    plt.title("Grafica de Convergencia")
    plt.show()


def EA(f, lb, ub, pc, pm, nvars, npop, ngen, precision):
    """Algoritmo Evolutivo para resolver f con 
        -Representacion: Real Entera
        - Selección de padres:  Universal Estocástica
        - Escalamiento: Ninguno
        - Cruza: Aritmética Total
        - Mutación: Uniforme
        - Selección: Más"""
    # Inicializar
    estadisticas = []
    ba = np.zeros((ngen, 1))
    genotipos, fenotipos, aptitudes = inicializar(
        f, npop, nvars, lb, ub, precision)
    # Hasta condición de paro
    for i in range(ngen):
        mutaciones = 0
        cruzas = 0
        # Selección de padres
        padres = seleccion_universal_estocastica(aptitudes, npop)
        # Cruza
        hijos_genotipos, cruzas = cruza_aritmetica_total(
            lb, ub, genotipos, padres, pc, precision)
        # Mutación
        hijos_genotipos, mutaciones = mutacion_uniforme(
            hijos_genotipos, lb, ub, pm, precision)
        hijos_fenotipos = fenotipo(hijos_genotipos, precision)
        hijos_aptitudes = f(hijos_fenotipos)

        # Estadistica
        estadisticas.append(
            estadistica(i, genotipos, fenotipos, aptitudes, hijos_genotipos, hijos_fenotipos, hijos_aptitudes, padres, mutaciones, cruzas))

        # Mejor individuo
        ba[i] = np.copy(aptitudes[estadisticas[i][0]])

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
nvars = 2
# Numero de poblacion
npop = 20
# limite inferior del espacio de busqueda
lb = -500
# Limite superior del espacio de busqueda
ub = 500
# Usaremos una Precision de 5 decimales
precision = 5
# porcentaje de Cruza
pc = 0.9
# Porcentaje de Mutacion
pm = 0.2
# Numero de Generaciones
ngen = 30

#modificaremos el formato para que  no aparezca en forma exponencial
np.set_printoptions(suppress=True)
bgen, bfen, bapt = EA(f, lb, ub, pc, pm, nvars, npop, ngen, precision)
print("Se utilizo la semilla aleatoria ", seed)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"El Resultado del mejor individuo en base a su aptitud es el que tiene \
    \nGenotipo es {bgen}  \
    \nFenotipo es {bfen}  \
    \nY una Aptitud de {bapt}")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
