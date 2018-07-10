from __future__ import division

from edgestruct import *
# Le nouveau fichier edgestruct.py ne marche pas chez moi

import numpy as np
import numpy.linalg as la
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import scipy.stats

pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp

def H1(u, mesh):
    """Norme H1 spatiale"""
    s = 0
    
    #inner edges
    s += np.sum(mesh.mes_i * mesh.dKL * ((u[mesh.Kin] - u[mesh.Lin]) / mesh.dKL) ** 2)
    
    #boundary edges
    s += np.sum(mesh.mes_b * mesh.dKs_b * (u[mesh.Kbnd] / mesh.dKs_b) ** 2)
    
    return np.sqrt(s)

def L2_x(u, mesh):
    """Norme L2 spatiale"""
    s = np.sum(mesh.mes_i * u[mesh.Kin] ** 2)
    s += np.sum(mesh.mes_b * u[mesh.Kbnd] ** 2)
    return np.sqrt(s)

def L2_t(u, dt):
    """Norme L2 temporelle"""
    s = dt * np.sum(u ** 2)

    return np.sqrt(s)

def generer_carre(number):
    Nx = number
    Ny = number
    plt.figure(0)
    
    mesh = rectangle(Nx, Ny)
    return mesh

class Solution:
    """ Permet de générer un couple de solutions """
    def __init__(self, P, C):
        self.P = P
        self.C = C

#==============================================================================
# Nous définissons plusieurs matrices qui seront utiles durant l'exercice
#==============================================================================

def creer_matrice_A(mesh):
    """Matrice de diffusion"""
    datadiff = np.concatenate([mesh.mes_i / mesh.dKL, mesh.mes_i / mesh.dKL, 
                           -mesh.mes_i / mesh.dKL, -mesh.mes_i / mesh.dKL,
                           np.zeros(mesh.nbnd)]) # Flux nul au bord
        
    I = np.concatenate([mesh.Kin, mesh.Lin, mesh.Kin, mesh.Lin, mesh.Kbnd])
    J = np.concatenate([mesh.Kin, mesh.Lin, mesh.Lin, mesh.Kin, mesh.Kbnd])
    
    Adiff = spsp.csr_matrix((datadiff, (I, J)), (mesh.nvol, mesh.nvol))
    return Adiff

def creer_matrice_Bcn(C, mesh, test3 = False):
    """Matrice de convection"""
    # Partie positif, partie négative
    pp = lambda x: (np.abs(x) + x) / 2
    pn = lambda x: (np.abs(x) - x) / 2
    
    diff_C = C[mesh.Lin] - C[mesh.Kin]
    
    data_convection = np.concatenate([mesh.mes_i[mesh.Kin] / mesh.dKL[mesh.Kin] * pp(diff_C), 
                                      mesh.mes_i[mesh.Lin] / mesh.dKL[mesh.Lin] * pn(diff_C),
                                      - mesh.mes_i[mesh.Kin] / mesh.dKL[mesh.Kin] * pn(diff_C),
                                      - mesh.mes_i[mesh.Lin] / mesh.dKL[mesh.Lin] * pp(diff_C),
                                      np.zeros(mesh.nbnd)]) # Flux nul au bord
    
    I = np.concatenate([mesh.Kin, mesh.Lin, mesh.Kin, mesh.Lin, mesh.Kbnd])
    J = np.concatenate([mesh.Kin, mesh.Lin, mesh.Lin, mesh.Kin, mesh.Kbnd])
    
    Bcn = spsp.csr_matrix((data_convection, (I, J)), (mesh.nvol, mesh.nvol))
    
    return Bcn


def creer_matrice_M(mesh):
    """Matrice des mesures des volumes finis"""
    M = spsp.diags(mesh.compute_vol(), 0, (mesh.nvol, mesh.nvol))
    
    return M


#==============================================================================
# Nous définissions une fonction qui calcule la concentration initiale C0
#==============================================================================

def calculer_concentration_initiale(rho_0, mesh, display = False, c_exacte = 0):
    """Calcule la concentration initiale en attractant en utilisant la matrice Adiff"""
    # display = true affiche la solution c0 obtenue et la compare avec l'erreur
    def c_exacte_test(x):
        return c_exacte(x[:, 0], x[:, 1])
 
    centres = mesh.centres

    # P vecteur de solution rho sur tout l'espace
    P = rho_0(centres[:, 0], centres[:, 1])
    
    # 2. Initilisation de C

    C = np.zeros(mesh.nvol)
    
    # 3. Construction de la matrice du Laplacien
    # Matrice diffusive et matrice des mesures des mailles
    Adiff = creer_matrice_A(mesh)    
    M = creer_matrice_M(mesh)    
    
    # 4. Résolution de c

    # On veut les mesures de chaque maille, pas    
    C = spla.spsolve(Adiff + M, M * P)
    
    """ Affichage de la solution """
    if(display):
        # Affichage des solutions
        fig_solution_exacte, axes_solution_exacte = plt.subplots(1,1)
        plotdiscrete(mesh, c_exacte_test)
        axes_solution_exacte.set_title("Solution c0 exacte")
        
        fig_solution_approchee, axes_solution_approchee = plt.subplots(1, 1)
        plotdiscrete(mesh, C)
        axes_solution_approchee.set_title("Solution c0 approchee")
        
        fig_erreurs, axes_erreurs = plt.subplots(1, 1)
        plotdiscrete(mesh, C - c_exacte_test(centres))
        axes_erreurs.set_title("Erreurs")
    
    return C
    

#==============================================================================
# Résolution numérique de l'équation parabolique par le schéma TPFA
#==============================================================================

def resolution_temporelle(mesh, rho_0, T, N, Xi, D, test3 = False):
    """Résout l'équation en Rho et P"""
    centres = mesh.centres
    P = rho_0(centres[:, 0], centres[:, 1])

    # Différenciation pour le test 3
    if not(test3):
        C = calculer_concentration_initiale(rho_0, mesh)
    else:
        C = mesh.centres[:, 0]
    
    
    dt = T / N
    Adiff = creer_matrice_A(mesh)
    M = creer_matrice_M(mesh)
    
    # Objet spécialement créé pour pouvoir mieux exporter la paire de solutions
    S = Solution(P, C)
    
    yield S
    
    for _ in range(N):
        # Matrice Bcn crée à partir de la concentration en attractant
        Bcn = creer_matrice_Bcn(C, mesh, test3)
        H = M + dt * D * Adiff + dt * Xi * Bcn
        
        # Mise à jour de la concentration en bactéries
        P = spla.spsolve(H, M * P)
        
        # Différenciation pour le test 3
        if not(test3):
            # Mise à jour de la concentration en attractant
            C = spla.spsolve(Adiff + M, M * P)
        
        S = Solution(P, C)
        yield S
        

def split_solutions(S, P, C):
    """P et C doivent être des listes vides"""
    for s in S:
        P.append(s.P)
        C.append(s.C)







