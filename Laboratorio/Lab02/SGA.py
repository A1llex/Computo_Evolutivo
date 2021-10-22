import numpy as np
from numpy.lib.financial import pmt
from numpy.lib.nanfunctions import _nanvar_dispatcher


def f(X):
    return -np.sum((X-5)**2, axis=1)

def inicializar (f,npop,nvar,lb,ub):
    genotipos = lb+ (ub-lb)*np.random.uniform(0, 1, size= [npop,nvar])
    
    fenotipos = genotipos

    aptitudes  = f(fenotipos)
    
    return genotipos,fenotipos,aptitudes

def seleccion_ruleta(aptitudes,n):
    #suma de aptitudes
    p= aptitudes/sum(aptitudes)
    #acumulada
    cp=np.cumsum(p)
    padres = np.zeros(n)
    #genearar aleatorio
    for i in range(n):
        X =np.reandom.uniform()
        #seleccionando padre
        padres[i] = np.argwhere(cp>X)[0]

    return padres

def cruza_un_punto(genotipos,indx,pc):
    hijos_genotipo=np.zeros(np.shape(genotipos))
    k=0
    #i y j
    for i,j in zip(indx[::2],indx[1::2]):
        flip =np.random.uniform() <= pc
        if flip:
            punto_cruza= np.random.randint(0,len(genotipos[0]))
            hijos_genotipo[k] = np.concatenate(genotipos[i,0:punto_cruza],genotipos[j,punto_cruza:])
            hijos_genotipo[k+1] = np.concatenate(genotipos[i,0:punto_cruza],genotipos[j,punto_cruza:])
        else:
            hijos_genotipo[k] = np.copy(genotipos[i])
            hijos_genotipo[k+1] = np.copy(genotipos[j])
        k+2
    return hijos_genotipo


def mutacion_uniforme(genotipos,lb,ub,pm):
    for i in range(len(genotipos)):
        for j in range(len(genotipos)):
            flip = np.random.uniform() <= pm
            if flip:
                genotipos[i,j] = np.random.uniform(lb[j],ub[j])
            
    return genotipos

def seleccion_coma(genotipos , fenotipos , aptitudes, hijos_genotipo , hijos_fenotipos , hijo_aptitudes ):

    return  hijos_genotipo , hijos_fenotipos , hijo_aptitudes


def EA(f,ngen,npop,nvar,lb,ub,pc,pm):
    #parametros
    #inicializacion
    genotipos,fenotipos,aptitudes = inicializar(f,npop,nvar,lb,ub)

    for i in range(ngen):
        #seleccion de ruleta
        indx= seleccion_ruleta(aptitudes,int(npop/2))
        #cruza
        hijos_genotipo = cruza_un_punto(genotipos,indx,pc)

        hijos_genotipo = mutacion_uniforme(genotipos,lb,ub,pm)

        hijos_fenotipos = hijos_genotipo

        hijo_aptitudes = f(hijos_fenotipos)

        genotipos , fenotipos , aptitudes = seleccion_coma(genotipos , fenotipos , aptitudes, hijos_genotipo , hijos_fenotipos , hijo_aptitudes )

    #return

    indx = np.argmax(aptitudes)
    #print(indx)
    return genotipos[indx],fenotipos[indx],aptitudes[indx]


lb = np.array([0,0])
ub =np,array([10,10])
pc = 0.9
pm = 0.001
nvars = 2
npop =20
ngen = 1000

