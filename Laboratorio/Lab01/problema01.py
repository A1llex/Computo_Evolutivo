import numpy as np
import torch

#solo para funciones ecalares , si se quiere para vectorres sera necesario modificarlo

#sphere
def spheres_f(x):
    return sum(x**2)

#algo de un jacobiano
def spheres_j(x):
    return 2*x

#esta y la deabajo solo euq esta esta hecha con numpy
def schwefel_f(x):
    alpha = 418.982887
    return sum(-x*np.sin(np.sqrt(np.abs(x)))) +alpha*len(x)

#hecho con torch y ahi esta el gradiente con el 
def schwefel_j(x):
    x_t= torch.from_numpy(x)
    x_t.requires_grad()
    alpha = 418.982887
    fx_t = sum(-x_t*torch.sin(torch.sqrt(torch.abs(x_t))))+alpha*len(x)
    #j_t = sum
    return x_t.grad.numpy()

