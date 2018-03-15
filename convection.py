# -*- coding: utf-8 -*-

from fonctions import *
from display import *
from tqdm import tqdm

#==============================================================================
# Test 1
# On s'intéresse à la solution initiale C0 et on calcule l'ordre de convergence
# du schéma en c
#==============================================================================

erreurs = []
numbers = [5, 11, 21, 51]


animations = []

def rho_0(x1, x2):
        return cos(pi * x1) * cos(pi * x2)
def rho_0_test(x):
    return rho_0(x[:, 0], x[:, 1])
    
def C0_exacte(x1, x2):
    return 1 / (1 + 2 * pi**2) * rho_0(x1, x2)
def C0_exacte_test(x):
    return C0_exacte(x[:, 0], x[:, 1])

for number in numbers:
    mesh = generer_carre(number)
    centres = mesh.centres
    C = calculer_concentration_initiale(rho_0, mesh)

    er = (L2_x(C - C0_exacte_test(centres), mesh))

    #er = -np.max(C) / np.max(C0_exacte_test(centres))

    erreurs.append(er)
    
# On trace l'erreur H1

plt.loglog(numbers, erreurs)
plt.title("Evolution de l'erreur L2 entre la solution exacte et la solution approchée")
plt.xlabel("Inverse du diamètre d'un volume fini")
plt.ylabel("Erreur commise")

number = 10
mesh = generer_carre(number)
C = calculer_concentration_initiale(rho_0, mesh, True, C0_exacte)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf(mesh, C, ax)
plt.title("Solution C à l'instant initial")
plt.xlabel("x")
plt.ylabel("y")


#==============================================================================
# Test 2
# On ne s'intéresse qu'au phénomène de diffusion et on veut déterminer l'ordre
# en temps
#==============================================================================
        
D = 1
Xi = 0
T = 0.1
N = 50
number = 10
mesh = generer_carre(number)
centres = mesh.centres

""" Nous affichons l'évolution d'une seule solution """
dt = T / N
S = resolution_temporelle(mesh, rho_0, T, N, Xi, D)
P = []
C = []
split_solutions(S, P, C)

# Solution exacte : équation de chaleur 2D
def rho_exacte(t, x):
    return exp(-2 * pi**2 * D * t) * rho_0_test(x)

# Solution exacte 
t = np.linspace(0, T, N)
P_exacte = []
for t_i in t:
    P_exacte.append(rho_exacte(t_i, mesh.centres))
P_exacte = np.array(P_exacte)
P = np.array(P)

# On calcule maintenant l'erreur L2 entre la solution exacte 
# et la solution approchée à chaque instant
erreur = []
for k in range(N):
    erreur.append(L2_x(P[k] - P_exacte[k], mesh))

erreur = np.array(erreur)
erreur = L2_t(erreur, dt)
# On ajoute l'erreur dans notre plot d'erreurs
erreurs.append(erreur)

ani_test_2 = animate("Test 2", mesh, t, [P, P_exacte], legends = ["Solution approchée", "Solution exacte"])
animations.append([ani_test_2, "test2"])


""" Nous allons générer des solutions pour un nombre croissant de volumes finis et ainsi calculer la convergence en espace"""
erreurs = []
Ns = np.arange(0, 100, 10)
for N in Ns:

    
    dt = T / N
    S = resolution_temporelle(mesh, rho_0, T, N, Xi, D)

    P = []
    C = []

    split_solutions(S, P, C)

    # Solution exacte : équation de chaleur 2D
    def rho_exacte(t, x):
        return exp(-2 * pi**2 * D * t) * rho_0_test(x)

    # Solution exacte 
    t = np.linspace(0, T, N)
    P_exacte = []
    for t_i in t:
        P_exacte.append(rho_exacte(t_i, mesh.centres))
    P_exacte = np.array(P_exacte)
    P = np.array(P)

    # On calcule maintenant l'erreur L2 entre la solution exacte 
    # et la solution approchée à chaque instant
    erreur = []
    for k in range(N):
        erreur.append(L2_x(P[k] - P_exacte[k], mesh))

    erreur = np.array(erreur)
    erreur = L2_t(erreur, dt)
    # On ajoute l'erreur dans notre plot d'erreurs
    erreurs.append(erreur)



plt.figure()
plt.loglog(Ns, erreurs)
plt.title("Evolution de l'erreur L2 temporelle entre la solution exacte et la solution approchée")
plt.xlabel("Nombre de pas")
plt.ylabel("Erreur commise")

#==============================================================================
# Test 3
# On ne s'intéresse qu'au phénomène de transport
# On pose C[k] = x[k, 0]
#==============================================================================

D = 0
Xi = 1
T = 0.5
N = 100
dt = T / N
number = 51
mesh = generer_carre(number)
centres = mesh.centres

def rho_0_3(x1, x2):
    sigma = 5 * 10e-3
    return 1. / (2 * pi * sigma) * exp(-((x1 - 0.5) ** 2 
            + (x2 - 0.5) ** 2) / (2 * sigma))
def sol_exacte(t, x):
    return rho_0_3(x[:, 0] - t, x[:, 1])

# On génère la solution exacte
t = np.linspace(0, T, N)
P_exacte = []

for t_i in t:
    P_exacte.append(sol_exacte(t_i, mesh.centres))
P_exacte = np.array(P_exacte)
P = np.array(P)

# On calcule la solution approchée
S = resolution_temporelle(mesh, rho_0_3, T, N, Xi, D, test3 = True)

P = []
C = []

split_solutions(S, P, C)
# On compare la solution approchée et la solution exacte
ani_test_3 = animate("test 3", mesh, t, [P, P_exacte], legends = ["Solution approchée", "Solution de référence"])
animations.append([ani_test_3, "test3"])
# Solution initiale


#==============================================================================
# Test 4 et 5
# On s'intéresse maintenant à l'équation de chimiotaxie en général et on veut 
# étudier le phénomène d'explosion en temps fini
#==============================================================================

#==============================================================================
# Test 4
#==============================================================================

# Paramètres d'initalisation
D = 1
Xis = [0.1, 0.5, 0.9, 1, 2.0, 4.]
number = 21
mesh = generer_carre(number)
centres = mesh.centres
for Xi in Xis:
        
    T = 20
    N = 100
    dt = T / N
    
    
    

    # Solution initiale
    def rho_0_4(x1, x2):
        sigma = 5 * 10e-3
        return 5. / sigma * exp(-((x1 - 0.5) ** 2 
                + (x2 - 0.5) ** 2) / (2 * sigma))

    # Résolution numérique et affichage
    t = np.linspace(0, T, N)
    S = resolution_temporelle(mesh, rho_0_4, T, N, Xi, D)
    P = []
    C = []

    split_solutions(S, P, C)
    ani_test_4 = animate("test 4 Xi " + str(Xi), mesh, t, [P, C])
    animations.append([ani_test_4, "test4Xi" + str(Xi)])

#==============================================================================
# Test 5
#==============================================================================
# Paramètres initiaux
D = 1

for Xi in Xis:
    T = 20
    N = 100
    dt = T / N
    number = 21

    t = np.linspace(0, T, N)

    # Solution initiale
    def rho_0_5(x1, x2):
        sigma = 5 * 10e-3
        return 3. / sigma * exp(-((x1 - 0.6) ** 2 
                + (x2 - 0.6) ** 2) / (2 * sigma))

    # Résolution numérique et affichage
    S = resolution_temporelle(mesh, rho_0_5, T, N, Xi, D)
    P = []
    C = []

    split_solutions(S, P, C)
    ani_test_5 = animate("test5 Xi " + str(Xi), mesh, t, [P, C], ["P", "C"])

    animations.append([ani_test_5, "test5Xi" + str(Xi)])

#==============================================================================
# Enregistrement de toutes les animations (ne fonctionne que sur windows)
#==============================================================================

for elt in animations:
    ani, title = elt
    export_animation(ani, title)

plt.show()































