import numpy as np
from sympy import *
import matplotlib.pyplot as plt

"""
@author: Alex Gerardo Fernandez Aguilar
Algoritmo Evoulutivo para resolver  ACKLEY  en 10 dimensiones
en un rango de en [-32.768, 32.768]
Se realizara con la siguiente seleccion de componentes
### Repartición de componentes
**Alex Fernández** : 
- Representacion: Real Entera
- Selección de padres:  Universal Estocástica
- Escalamiento: Ninguno
- Cruza: Aritmética Total
- Mutación: Uniforme
- Selección: Más
### Tecmica Avanzada
- Meméticos : Lamarkiano con decenso del gradiente
- Tecnica de diversidad : Procedimiento de clearing

"""

# Aqui sera necesario declarar la funcion para que sympy realice el calculo del gradiente
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = symbols(
    "x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 ", real=True
)
sympyf = (
    -20
    * exp(
        -0.2
        * sqrt(
            1 / 10
            + x1 ** 2
            + x2 ** 2
            + x3 ** 2
            + x4 ** 2
            + x5 ** 2
            + x6 ** 2
            + x7 ** 2
            + x8 ** 2
            + x9 ** 2
            + x10 ** 2
        )
    )
    - exp(
        1 / 10
        + cos(2 * pi * x1)
        + cos(2 * pi * x2)
        + cos(2 * pi * x3)
        + cos(2 * pi * x4)
        + cos(2 * pi * x5)
        + cos(2 * pi * x6)
        + cos(2 * pi * x7)
        + cos(2 * pi * x8)
        + cos(2 * pi * x9)
        + cos(2 * pi * x10)
    )
    + 20
    + exp(1)
)
grad_f = lambdify(
    (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10),
    derive_by_array(sympyf, (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)),
)


def f(x):
    sum1 = -20 * np.exp(-0.2 * np.sqrt(0.1 * np.sum(x ** 2 , axis=1 ) ) )
    sum2 = - np.exp(0.1 * np.sum( np.cos(2 * np.pi * x) , axis=1))
    return sum1 + sum2 + 20 + np.exp(1)


def inicializar(
    f, npop, nvars, lb, ub,
):
    """Inicia con una poblacion aleatoria con un Genotipo Real Entero, 
    y un fenotipo  """
    # Generamos el Genotipo desde el limite inferior hasta el superior multiplicados por la preciion deseada,
    #  randint regresa una distribucion uniforme
    genotipos = lb + (ub - lb) * np.random.uniform(
        low=0.0, high=1.0, size=[npop, nvars]
    )
    # transformamos el genoripo a un Fenotipo dentro de los limites
    fenotipos = genotipos
    # Valuamos las aptitudes
    aptitudes = f(fenotipos)
    return genotipos, fenotipos, aptitudes


def seleccion_universal_estocastica(aptitudes, npop):
    """Funcion Estocastica para determinar ``npop`` padres de una fomra Universal Etocastica"""
    # Probabilidad
    # Es necesaria que como minimizamos la aptitud entre menor sera sera mejor por esto hay que
    g = 1 / aptitudes
    p = g / sum(g)
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
            if len(padres) < npop:
                padres.append(i)
            ptr += 1
    return padres


def a(lb, ub, vk, wk):
    """
    [max(α, β), min(γ, δ)] si v_k > w_k
    \n|0,0| si v_k = w_k (esto es que ambos valores se quedaran igual)
    \n[max(γ, δ), min(α, β)] si v_k < w_k
    """
    if vk == wk:
        return vk, wk
    alpha = (lb - wk) / (vk - wk)
    beta = (ub - vk) / (wk - vk)
    gamma = (lb - vk) / (wk - vk)
    delta = (ub - wk) / (vk - wk)
    if vk > wk:
        l = max(alpha, beta)
        u = min(gamma, delta)
    else:
        l = max(gamma, delta)
        u = min(alpha, beta)
    a = np.random.uniform(low=l, high=u)
    # Se transforman a int para que sean parte del genotipo
    v = int(a * wk + (1 - a) * vk)
    w = int(a * vk + (1 - a) * wk)
    return v, w


def cruza_aritmetica_total(lb, ub, genotipos, padres, pc):
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
                v, w = a(lb, ub, vk, wk)
                hijos_genotipos[k] = v
                hijos_genotipos[k + 1] = w
        else:
            hijos_genotipos[k] = np.copy(genotipos[i])
            hijos_genotipos[k + 1] = np.copy(genotipos[j])
        k += 2
    # si habia un numero impar de padres , el ultimo se incluira para que sea el mismo numero de padres que de hijos
    if k < len(padres):
        hijos_genotipos[k] = padres[-1]
    return hijos_genotipos, cruzas


def mutacion_uniforme(genotipos, lb, ub, pm):
    """Mutacion Uniforme,lo que haremos sera elegir un numero aleatorio entre nuestros limites y sustituirlo en una posicion seleccionada"""
    mutaciones = 0
    for i in range(len(genotipos)):
        for j in range(len(genotipos[i])):
            if np.random.uniform() <= pm:
                mutaciones += 1
                genotipos[i, j] = lb + (ub - lb) * np.random.uniform(low=0.0, high=1.0)
    return genotipos, mutaciones


def seleccion_mas_clearing(
    npop,
    genotipos,
    fenotipos,
    aptitudes,
    hijos_genotipos,
    hijos_fenotipos,
    hijos_aptitudes,
    radio,
):
    """Seleccion mas para la poblacion , esto es unir ambas poblaciones y elegir la mejor mitad deacuerdo a la aptitud"""
    # Lo que haremos sera juntar todos en una lista de 3-tuplas+1 y ordenarlas por aptitud

    # meteremos otra que sera true o false o 1 y 0 si son centros o no
    pganador = np.zeros(len(aptitudes), dtype=float)
    hganador = np.zeros(len(hijos_aptitudes), dtype=float)

    # Representacion de tuplas de ([genotipo1,genotipo2], [fenotipo1,fenotipo2], [aptitud] , [0])
    total = list(zip(genotipos, fenotipos, aptitudes, pganador)) + list(
        zip(hijos_genotipos, hijos_fenotipos, hijos_aptitudes, hganador)
    )
    # ordenados segun su aptitud
    total.sort(key=lambda x: x[2])
    # PROCEDIMIENTO DE CLEARING
    # modificaremos sus ganadores
    ganador = []
    total[0] = list(total[0])
    total[0][3] = 1
    ganador.append(total[0][1])
    i = 1
    while i < len(total) and len(ganador) < npop:
        total[i] = list(total[i])
        for j in ganador:
            if np.linalg.norm(np.array(j) - np.array(total[i][1])) > radio:
                total[i][3] = 1
                ganador.append(total[i][1])
            else:
                continue
        i += 1
    # ordenados segun su aptitud y si son ganadores
    total.sort(key=lambda x: ( -x[3] ,x[2] ))
    # cortamos la poblaciones hasta la deseada
    total = total[:npop]
    gen, fen, apt, gndr = zip(*total)
    return np.array(gen), np.array(fen), np.array(apt)


def descenso_del_gradiente(x, alpha, iter):
    """Descenso del gradiente como busqueda local x sera un arreglo de n vars"""
    x_i = x
    for i in range(iter):
        # depende el numero de variables
        grad_value = -alpha * np.array(
            grad_f(
                x_i[0],
                x_i[1],
                x_i[2],
                x_i[3],
                x_i[4],
                x_i[5],
                x_i[6],
                x_i[7],
                x_i[8],
                x_i[9],
            )
        )
        x_i = np.sum([x_i, grad_value], axis=0)
    return x_i


def memetico(hijos_genotipos, step, iteraciones, npop, nvars):
    """Algortimso memeteico de buscqueda local laamrkiano con descenso del gradiente"""
    genotipos = np.zeros((npop, nvars), dtype=float)
    for i in range(len(hijos_genotipos)):
        genotipos[i] = descenso_del_gradiente(hijos_genotipos[i], step, iteraciones)
    return genotipos


def estadistica(
    generacion,
    genotipos,
    fenotipos,
    aptitudes,
    hijos_genotipos,
    hijos_fenotipos,
    hijos_aptitudes,
    padres,
    mutaciones,
    cruzas,
    mostrar
):
    """Estadisticas, regresa [aptMin , aptMed ,aptMax,desvEst]
    No hay que olvidar que lo que estamos haciendo es minimizar por eso buscamos el mas pequeño"""
    aptMax = np.argmax(aptitudes)
    aptMed = np.mean(aptitudes)
    aptMin = np.argmin(aptitudes)
    desvEst = np.std(aptitudes)
    mediana = np.median(aptitudes)
    if mostrar:
        print("------------------------------------------------------------")
        print("Generación:", generacion)
        print(
            "La poblaicon y los hijos seran presentados como : \n[[indice, genotipo , fenotipo, aptitudes  , probabilidad],...]"
        )
        print(
            "Población :\n",
            np.concatenate(
                (
                    np.arange(len(aptitudes)).reshape(-1, 1),
                    genotipos,
                    aptitudes.reshape(-1, 1),
                    aptitudes.reshape(-1, 1) / np.sum(aptitudes),
                ),
                1,
            ),
        )
        print(
            f"Mejor individuo\
            \nIndice: {aptMin} \
            \nGenotipo: {genotipos[aptMin]} \
            \nFenotipo: {fenotipos[aptMin]} \
            \nAptitud: {aptitudes[aptMin]}"
        )
        print(
            f"Aptitud Maxima {aptitudes[aptMax]} \
            \nAptitud Media {aptMed} \
            \nAptitud Minima {aptitudes[aptMin]}\
            \nAptitud Mediana {mediana}"
        )
        print("Desviacion Estandar", desvEst)
        print("Padres Seleccionados", padres)
        print("Frecuencia de padres seleccionados:", np.bincount(padres))
        print(
            f"Cruzas Efectuadas {cruzas} \
            \nMutaciones Efectuadas {mutaciones}"
        )
    return [aptMin, aptMed, aptMax, desvEst, mediana]


def grafica(estadisticas):
    """Reportar gráfica de convergencia. 
    Eje x número de generaciones, 
    eje y mediana de la mejor aptitud de cada generación"""
    # Plot
    plt.plot(range(len(estadisticas)), list(zip(*estadisticas))[4], marker="o")

    plt.xlabel("Generaciones")
    plt.ylabel("Mediana de  Aptitudes")
    plt.title("Grafica de Convergencia")
    plt.show()


def EA(f, lb, ub, pc, pm, nvars, npop, ngen, step, bl_iteraciones, radio):
    """Algoritmo Evolutivo para resolver f con 
        -Representacion: Real Entera
        - Selección de padres:  Universal Estocástica
        - Escalamiento: Ninguno
        - Cruza: Aritmética Total
        - Mutación: Uniforme
        - Selección: Más"""
    # Inicializar
    estadisticas = []
    ba = np.zeros((ngen, 1), dtype=float)
    genotipos, fenotipos, aptitudes = inicializar(f, npop, nvars, lb, ub)
    # Hasta condición de paro
    for i in range(ngen):
        mutaciones = 0
        cruzas = 0

        # Selección de padres
        padres = seleccion_universal_estocastica(aptitudes, npop)

        # Cruza
        hijos_genotipos, cruzas = cruza_aritmetica_total(lb, ub, genotipos, padres, pc)

        # Mutación
        hijos_genotipos, mutaciones = mutacion_uniforme(hijos_genotipos, lb, ub, pm)

        # Memetico
        # Improve memetico lamarkiano
        hijos_genotipos = memetico(hijos_genotipos, step, bl_iteraciones, npop, nvars)

        # Fenotipos y aptitud
        hijos_fenotipos = hijos_genotipos

        hijos_aptitudes = f(hijos_fenotipos)

        # Selección de siguiente generación
        genotipos, fenotipos, aptitudes = seleccion_mas_clearing(
            npop,
            genotipos,
            fenotipos,
            aptitudes,
            hijos_genotipos,
            hijos_fenotipos,
            hijos_aptitudes,
            radio,
        )

        #Estadistica
        #Valor si imprimir estadisticas por gneracion
        mostrar = False
        estadisticas.append(
            estadistica(
                i,
                genotipos,
                fenotipos,
                aptitudes,
                hijos_genotipos,
                hijos_fenotipos,
                hijos_aptitudes,
                padres,
                mutaciones,
                cruzas,
                mostrar
            )
        )

    # Grafica
    grafica(estadisticas)
    return genotipos, fenotipos, aptitudes


# Random seed
seed = np.random.randint(100000)
np.random.seed(seed=seed)
# Numero de variables para el cromosoma
nvars = 10
# limite inferior del espacio de busqueda
lb = -32.768
# Limite superior del espacio de busqueda
ub = 32.768

# Numero de poblacion
npop = 20
# Numero de Generaciones
ngen = 30
# porcentaje de Cruza
pc = 0.9
# Porcentaje de Mutacion
pm = 0.2

# Numero de pasos de buscqueda local
bl_iteraciones = 30
# Tamaño de paso para la busqueda local
beta = np.linalg.norm(np.array([lb,lb])- np.array([ub,ub])) / 20
step = beta/500

# Radio para la los nichos
radio = beta

# modificaremos el formato para que  no aparezca en forma exponencial
np.set_printoptions(suppress=True)
#EA
bgen, bfen, bapt = EA(f, lb, ub, pc, pm, nvars, npop, ngen, step, bl_iteraciones, radio)
#Resultados
print("Se utilizo la semilla aleatoria ", seed)
best = np.argmin(bapt)
print("minima", np.min(bapt))
print("media", np.mean(bapt))
print("mediana", np.median(bapt))
print("desviacion estandar",np.std(bapt))
print("Aptitudes", bapt)