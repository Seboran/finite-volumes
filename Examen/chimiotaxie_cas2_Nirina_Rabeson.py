# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 08:22:34 2017

@author: nirin
"""

# Chimiotaxie


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from collections import defaultdict
from pylab import *
import csv
import os

cos = np.cos
pi = np.pi
exp = np.exp

plt.close("all")
#===================================================
# Définition d'une norme
#===================================================

def normeLinf(u):
    return np.max(u)

def normeL2(u, p):
    return np.linalg.norm(np.sqrt(p) * u)

#===================================================
# Définition des volumes finis
#===================================================

def creer_pas(number):
    return (1. / number + np.zeros([number, 1]))

def construireVolumes(pas):
    N = len(pas) + 1
    sommets = np.zeros([N, 1])
    
    distances = np.zeros([N, 1])
    
    for i in range(N - 1):
        sommets[i + 1] = sommets[i] + pas[i]
        
    centres = 0.5 * (sommets[0: N - 1] + sommets[1 : N])
    
    distances[0] = centres[0] - sommets[0]
    
    distances[-1] = sommets[-1] - centres[-1]
    
    distances[1: -1] = centres[1:] - centres[0: -1]
    
    return N, sommets, distances, centres

"""
    Le problème de chimiotaxie est une équation différentielle couplée de deux
    inconnues u et v. On peut réécrire ces équations pour faire apparaître Navier-Stokes
    en intégrant v dans nu et eta
    En l'état, le code ne traite que le cas 1
"""

def nu(x, fonction):
    
    if fonction == 'u':
        return -0.1
    return -1.

def eta(X, fonction):
    if(fonction == 'u'):
        return -X
    
def q(u, i, r, fonction):
    if fonction == 'u':
        return r * (1 - u[i])
    return 0. * u

"""
    On se place dans le cas 1 pour l'instant
    Le schéma proposé est alors un schéma semi-implicite en temps et upwind en espace
"""

def construireMatriceUpwind(pas, N, sommets, distances, centres, fonction = 'u'):
    
    # Vecteur l de taille N
    l = np.zeros([N, 1])
    for i in range(N):
        l[i] = nu(sommets[i], fonction) * 1. / (distances[i])
        
    # Calcul de la matrice de diffusion
    Adiff = np.zeros([N, N])
    
    for i in range(1, N):
        Adiff[i, i] = Adiff[i, i] + l[i]
        Adiff[i, i - 1] = Adiff[i, i - 1] - l[i]
        Adiff[i - 1, i - 1] = Adiff[i - 1, i - 1] + l[i]
        Adiff[i - 1, i] = Adiff[i - 1, i] - l[i]
    
    # Flux nul aux bords

    
    # Acen est nulle car eta = 0

    # Avol est nulle car q = 0
    
    A = Adiff
    
    return A

#A = construireMatriceUpwind(pas, N, sommets, distances, centres, fonction = 'u')

def construireMatriceG_v(N, distances, sommets, Xi, v):
    G = np.zeros([N, N])
    for i in range(1, N - 1):
        v_diff_p = (v[i + 1] - v[i]) / distances[i]
        v_diff_m = (v[i] - v[i - 1]) / distances[i - 1]
        G[i, i] = (v_diff_p + np.abs(v_diff_p)) / 2. + (np.abs(v_diff_m) - v_diff_m) / 2.
        G[i, i + 1] = - (np.abs(v_diff_p) - v_diff_p) / 2.
        G[i, i - 1] = - (np.abs(v_diff_m) + v_diff_m) / 2.
    
    return Xi * G

#G = construireMatriceG_v(N, distances, sommets, Xi, v0(sommets))

def construireTermeVolumique(r, Idm, u, v, fonction = 'u'):
    if fonction == 'u':
        return r * np.dot(Idm, u * (1 - u))
    return np.dot(Idm, (u - v))

#q = construireTermeVolumique(1, distances, u0(sommets), v0(sommets), fonction = 'u')

def construireMatriceIdm(pas):
    # pas est une liste de listes de singletons, qui n'est pas compatible avec la fonction np.diag
    diag_vecteur = list(map(lambda l: l[0], pas.tolist()))
    return np.diag(diag_vecteur)

def construireMatriceGsi(dt, pas, N, sommets, distances, centres, fonction = 'u'):
    return (-dt * construireMatriceUpwind(pas, N, sommets, distances, centres, fonction) +
            construireMatriceIdm(distances))



def u0(x, cas = 2):
    if cas == 1:
        return cos(pi*x)
    return 1. - 0.1 * exp(-10 * x ** 2)
def v0(x, cas = 2):
    if cas == 1:
        return 0.* x
    return 1. + exp(-10 * (x - 1.) ** 2)


def schema(dt, T, Nt, pas, N, sommets, distances, centres, r = 0, Xi = 0, cas = 2):

    
    GsiU = construireMatriceGsi(dt, pas, N, sommets, distances, centres, 'u')
    
    GsiV = construireMatriceGsi(dt, pas, N, sommets, distances, centres, 'v')
    
    Idm = construireMatriceIdm(distances)
    
    u = u0(sommets, cas)
    v = v0(sommets, cas)
    
    
    
    for n in range(Nt):
        
        s = list(u) + list(v)

        yield s
                
        b_u = np.dot(Idm, u) + dt * construireTermeVolumique(r, Idm, u, v, 'u')
        b_v2 = construireTermeVolumique(r, Idm, u, v, 'v')
        
        G_v = construireMatriceG_v(N, distances, sommets, Xi, v)
        
        u = np.linalg.solve(GsiU + dt * G_v, b_u)
        
        b_v = np.dot(Idm, v) + dt * b_v2
        
        v = np.linalg.solve(GsiV, b_v)
        
def erreurs(dt, T, Nt, pas, N, sommets, distances, centres):
    
    GsiU = construireMatriceGsi(dt, pas, N, sommets, distances, centres, 'u')
    
    GsiV = construireMatriceGsi(dt, pas, N, sommets, distances, centres, 'v')
    
    Idm = construireMatriceIdm(distances)
    
    u = u0(sommets, 1)
    v = v0(sommets, 1)
    
    r = 0.
    Xi = 0.
    
    for n in range(Nt):
        
        
        ue = uexacte(n * dt, sommets)
        yield normeL2(ue - u, distances)
                
        b_u = np.dot(Idm, u) + dt * construireTermeVolumique(r, Idm, u, v, 'u')
        b_v2 = construireTermeVolumique(r, Idm, u, v, 'v')
        
        G_v = construireMatriceG_v(N, distances, sommets, Xi, v)
        
        u = np.linalg.solve(GsiU + dt * G_v, b_u)
        
        b_v = np.dot(Idm, v) + dt * b_v2
        
        v = np.linalg.solve(GsiV, b_v)
        
        
        
D = 0.1

def uexacte(t, x):
    return exp(-pi**2 * D * t) * cos(pi * x)
        
N = 300
T = 1
Nt = 200
t = np.linspace(0., T, Nt)
dt =  T / Nt


r = 1
Xi = 0.
cas = 2
#rs = np.linspace(0, 1, 5)
#Xis = np.linspace(0, 5, 5)
#for r in rs:
#    for Xi in Xis:
pas = creer_pas(N)
N, sommets, distances, centres = construireVolumes(pas)
u = list(schema(dt, T, Nt, pas, N, sommets, distances, centres, Xi = Xi, r = r, cas = cas))




ue = Nt * [[]]
for i in range(Nt):
    ue[i] = uexacte(t[i], sommets)




figure, axes = plt.subplots(1, 1)


lines = [[], []]
lines[0], = axes.plot([], [], label = 'Solution u')
lines[1], = axes.plot([], [], label = 'Solution v')


xdata, ydata = sommets, 0 * sommets
axes.grid()

def init():
    axes.set_xlim(0, 1)
    axes.set_ylim(0., 3.)
    
    
    
    axes.set_title("Evolution pour r " + str(r) + ' Xi ' + str(Xi))
    
    
    axes.set_xlabel('x')
    axes.set_ylabel('Concentration')
    
    

    
    axes.legend(loc = 'upper left')
    
    lines[0].set_data([], [])
    lines[1].set_data([], [])
    
    lines[0].set_color('b')
    lines[1].set_color('r')
    return lines,



def animate(i):
    # update the data
    lines[0].set_data(sommets, u[199][0:N])
    
        
    lines[1].set_data(sommets, u[199][N:])
  
    return lines,


end_file = 'r' + str(r) + 'Xi' + str(Xi) 
ani = animation.FuncAnimation(figure, animate, frames = Nt, interval = dt, repeat=True, init_func=init)


plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#ani.save('animation' + end_file + '.mp4', dpi=80, writer= writer)
#os.system("ffmpeg -i animation" + end_file + ".mp4 animation" + end_file+ ".gif")
plt.plot(sommets, u0(sommets), label = 'Solution u initiale')





