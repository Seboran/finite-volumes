#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:46:20 2017

@author: provis-03
"""

# Importation des librairies
from collections import defaultdict
import numpy as np
import scipy
import matplotlib.pylab as plt
import functools


def normeL2(u, p):
    s = 0
    for i in range(len(u)):
        s = s + p[i] * u[i]**2
    return np.sqrt(s)

def normeH1(u, p):
    s = 0
    for i in range(0, len(u) - 1):
        s = s + p[i] * ((u[i + 1] - u[i]) / p[i]) ** 2
    return np.sqrt(s)


def u_exacte(x, i):
    if(i == 1):
        return 0.5 * x * (1 - x)
    else:
        return np.sin(np.pi * x)

def nu(x):
    return 1. + 0 * x
def f(x, i):
    if(i == 1):
        return 1. + 0 * x
    elif(i == 2):
        return np.pi ** 2 * np.sin(np.pi * x) + 0*x
    elif(i == 3):
        return (1 + np.pi ** 2) * np.sin(np.pi * x) + 0*x
    else:
        return np.pi * x * np.cos(np.pi * x) + (2 + np.pi ** 2) * np.sin(np.pi * x) + 0*x
def eta(x, i):
    if(i == 4):
        return x
    else:
        return 0. + 0 * x
def q(x, i):
    if(i <= 2):
        return 0. + 0 * x
    else:
        return 1. + 0 * x

k, axaar = plt.subplots(2, 2)

def afficher(centres, u_approchee, u_analytique, errL2, errH1, cas):
    
    maxu = np.max(u_analytique)
    row = int((cas - 1) / 2)
    col = (cas - 1) % 2
    plt_centres, = axaar[row, col].plot(centres, 0*centres, 'x')
    # plt_sommets, = axaar[row, col].plot(sommets, 0*sommets, 'o', label = 'sommets')
        
    plt_solution_exacte, = axaar[row, col].plot(centres, u_analytique, label = 'Solution exacte')
    plt_solution_approchee, = axaar[row, col].plot(centres, u_approchee, label = 'Solution numérique')
    
    axaar[row, col].text(0.1, 8 * maxu / 10, "Erreur L2: " + str(errL2))
    
    
    

    axaar[row, col].text(0.1, 9 * maxu / 10, "Erreur H1: " + str(errH1))
    
    
    axaar[row, col].legend(handles=[plt_solution_exacte, plt_solution_approchee])
    
    axaar[row, col].set_title('Schéma volumes finis cas' + str(cas))
    


def simulation(N, cas):
    # Les constantes

    h = 1. / N
    
    # Le maillage
    
    pas = h + np.zeros([N, 1])
        
    sommets = np.zeros([N + 1, 1])
    
    distances = np.zeros([N + 1,1])
    
    for i in range(N):
        sommets[i + 1] = sommets[i] + pas[i]
        
    centres = 0.5 * (sommets[0: N] + sommets[1 : N + 1])
    
    distances[0] = centres[0] - sommets[0]
    
    distances[-1] = sommets[-1] - centres[-1]
    
    distances[1: -1] = centres[1:] - centres[0: -1]
        
    #print("sommets, centres, pas")
    #print(sommets,centres,pas)
    # Assemblage de la diffusion
    
    
    
    lambd = np.zeros([N + 1, 1])
    lambd = distances * eta(sommets, cas) / (2 * nu(sommets))
    
    # Vecteur l de taille N+1
    l = np.zeros([N + 1, 1])
    for i in range(N + 1):
        l[i] = nu(sommets[i]) * (1 + lambd[i]) / (distances[i])
        
    # Calcul de la matrice de diffusion
    Adiff = np.zeros([N, N])
    for i in range(1, N):
        Adiff[i, i] = Adiff[i, i] + l[i]
        Adiff[i, i - 1] = Adiff[i, i - 1] - l[i]
        Adiff[i - 1, i - 1] = Adiff[i - 1, i - 1] + l[i]
        Adiff[i - 1, i] = Adiff[i - 1, i] - l[i]
    
    Adiff[0, 0] = Adiff[0, 0] + l[0]
    Adiff[-1, -1] = Adiff [-1, -1] + l[-1]
    print(Adiff)
    #Acen
    
    #etai = eta(sommets, cas) / 2
    etai = np.zeros(N+1)

    for i in range(N+1):
        etai[i] = 0.5 * eta(sommets[i], cas)
    
    Acen = np.zeros([N, N])
    
    
    for i in range(1, N):
        Acen[i, i] = Acen[i, i] - etai[i]
        Acen[i, i - 1] = Acen[i, i - 1] - etai[i]
        Acen[i - 1, i - 1] = Acen[i - 1, i - 1] + etai[i]
        Acen[i - 1, i] = Acen[i - 1, i] + etai[i]
        
    Acen[0, 0] = Acen[0, 0] + etai[0]
    Acen[-1, -1] = Acen[-1, -1] - etai[-1]

    # Avol
    
    # pas * q renvoie une liste de listes de singletons, qui n'est pas compatible avec un vecteur diag
    diag_vecteur = list(map(lambda l: l[0], (pas * q(centres, cas)).tolist()))
    Avol = np.diag(diag_vecteur)
    
    A = Adiff + Acen + Avol

    # Définition second membre b
    
    b = pas * f(centres, cas)
    
    # Calcul de la solution
    
    u_approchee = np.linalg.solve(A, b)
    
    u_analytique = u_exacte(centres, cas)
    
    erreur = u_approchee - u_analytique
    

        
    erL2 = normeL2(erreur, distances)
    erH1 = normeH1(erreur, distances)
    # Visualisation
    
    
    
    return centres, u_approchee, u_analytique, erL2, erH1, cas

# Quelques exemples, affichage

for cas in range(4):
    c, uap, uan, er1, er2, ca = simulation(40, cas + 1)
    afficher(c, uap, uan, er1, er2, ca)

N = [10, 100, 200, 500, 1000, 2000, 3000, 5000]

errH1 = np.zeros([4, len(N)])
errL2 = np.zeros([4, len(N)])

for i in range(len(N)):
    n = N[i]
    for cas in range(1, 5):
        
        _, _, _, e1, e2, c = simulation(n, cas)
        
        errH1[c - 1][i] = e2.tolist()[0]
        errL2[c - 1][i] = e1.tolist()[0]
        
# Calcul de l'erreur en fonction du nombre de volumes finis

f, ax = plt.subplots(1,1)

plterrors = []
for cas in range(4):
    pente1 = (np.log(errL2[cas][-1]) - np.log(errL2[cas][0])) / (np.log(N[-1]) - np.log(N[0]))
    pente2 = (np.log(errH1[cas][-1]) - np.log(errH1[cas][0])) / (np.log(N[-1]) - np.log(N[0]))
    q, = ax.loglog(N, errL2[cas], label = 'erreur L2 cas ' + str(cas + 1) + ' pente ' + str(pente1))
    p, = ax.loglog(N, errH1[cas], label = 'erreur H1 cas ' + str(cas + 1) + ' pente ' + str(pente2))
    plterrors.append(p)
    plterrors.append(q)
    
ax.legend(handles = plterrors)
ax.set_title("Evolution de l'erreur en fonction du nombre de volumes finis")
plt.show()
        












