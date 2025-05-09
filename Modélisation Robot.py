# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:08:22 2025

@author: Ngatam
"""
################################# Import ######################################

import sympy
import scipy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from numpy import linalg
from numpy.linalg import pinv
from mpl_toolkits.mplot3d import Axes3D
from sympy import sin, cos, sqrt, Matrix
from matplotlib.animation import FuncAnimation
from sympy.abc import alpha, beta, delta, theta, psi, omega



################################# Texte #######################################

"""
---- Paramètres du système ----
h_1 = 400 : hauteur poulie 1 (en mm)
h_2 = 2180 : hauteur poulie 2 (en mm)
L = 230 : longeur de la plaque de l'effecteur (en mm)
l = 230 : largeur de la plaque de l'effecteur (en mm)
l_1 = 2150 : distance entre 2 pieds de la structure (en mm)
K = 1.5 : rapport de transmission du ième enrouleur
e = 30 : rayon de l'enroleur i (en mm) 
rho = 0.005 : pas de l'enrouleur i (en mm)

---- Coefficients pour le calcul ----
X : position sur l'axe x_base des poulies 
Y : position sur l'axe y_base des poulies 
a : position sur l'axe x_effecteur des points d'accroche
b : position sur l'axe y_effecteur des points d'accroche
---
r = K * (e**2 + (rho**2)/2*np.pi) : coefficient d'enroulement des enrouleurs (supposé identiques car même enrouleurs)
lambda_1 = r*q_1 : lambda 1, longeur du câble 1
lambda_2 = r*q_2 : lambda 2, longeur du câble 2
lambda_3 = r*q_3 : lambda 3, longeur du câble 3
lambda_4 = r*q_4 : lambda 4, longeur du câble 4


---- Variables ----
lambda_1 : longueur du câble de la poulie 1
lambda_2 : longueur du câble de la poulie 2
lambda_3 : longueur du câble de la poulie 3
lambda_4 : longueur du câble de la poulie 4
---
q_1 : position angulaire du moteur 1
q_2 : position angulaire du moteur 2
q_3 : position angulaire du moteur 3
q_4 : position angulaire du moteur 4
---
p_1 : nombre du pas sur le moteur 1
p_2 : nombre du pas sur le moteur 2
p_3 : nombre du pas sur le moteur 3
p_4 : nombre du pas sur le moteur 4
---
Xe: position de l'effecteur sur l'axe X de la structure
Ye: position de l'effecteur sur l'axe Y de la structure
phi_1 : position angulaire de l'effecteur par rapport au repère de la base


---- Equations ----
lambda_1 = - sqrt((X_e - X[0] + a[0]*cos(phi_1) - b[0]*sin(phi_1))**2 + (Y_e - Y[0] + a[0]*sin(phi_1) + b[0]*cos(phi_1))**2)
lambda_2 =   sqrt((X_e - X[1] + a[1]*cos(phi_1) - b[1]*sin(phi_1))**2 + (Y_e - Y[1] + a[1]*sin(phi_1) + b[1]*cos(phi_1))**2)
lambda_3 =   sqrt((X_e - X[2] + a[2]*cos(phi_1) - b[2]*sin(phi_1))**2 + (Y_e - Y[2] + a[2]*sin(phi_1) + b[2]*cos(phi_1))**2)
lambda_4 = - sqrt((X_e - X[3] + a[3]*cos(phi_1) - b[3]*sin(phi_1))**2 + (Y_e - Y[3] + a[3]*sin(phi_1) + b[3]*cos(phi_1))**2)
---
q_1 = lambda_1 / r
q_2 = lambda_2 / r
q_3 = lambda_3 / r
q_4 = lambda_4 / r
---
pas_1 = (360*q_1) / 2*np.pi
pas_2 = (360*q_2) / 2*np.pi
pas_3 = (360*q_3) / 2*np.pi
pas_4 = (360*q_4) / 2*np.pi
---
Poulie 1 et 3 : {'-' : Dérouler, '+': Enrouler} 
Poulie 2 et 4 : {'-': Enrouler, '+': Dérouler}

Par convention, le sens trigonométrique est noté '+', 
On va donc considérer qu'il s'agit de l'action de dérouler du câble
On va donc avoir sur toutes les poulies:
    '-' -> Enrouler du câble sur la poulie
    '-' -> sens trigonométrique des poulies 1 et 4, sens horaire des poulies 2 et 3
    
    '+' -> Dérouler du câbles sur la poulie
    '+' -> sens trigonométrique des poulies 2 et 3, sens horaire des poulies 1 et 4
    
    Pour avoir cette logique sur tout les poulies, on doit mettre des '-' devant les calculs de lambda_1 et lambda_4

"""



################################# Définition ##################################

## Paramètres du système
h_1 = 400 # (en mm) hauteur poulie 1
h_2 = 2180 # (en mm) hauteur poulie 2
l = 230 # (en mm) largeur de la plaque de l'effecteur
L = 230 # (en mm) longeur de la plaque de l'effecteur
l_1 = 2150 # (en mm) distance entre 2 pieds de la structure
K = 0.5 # rapport de transmission de l'enrouleur
e = 30 # (en mm) rayon de l'enroleur 
rho = 5 # (en mm) pas de l'enrouleur
pas_mot = 1.8 # (en °) pas du moteur => 200 pas pour 1 tour


## Coefficients pour le calcul
r = K * (e**2 + (rho**2)/2*np.pi) # coefficient d'enroulement des enrouleurs
X = [l_1, l_1, 0, 0] # position sur l'axe x_base des poulies
Y = [h_1, h_2, h_1, h_2] # position sur l'axe y_base des poulies 
a = [l/2, l/2, -l/2, -l/2] # position sur l'axe x_effecteur des points d'accroche
b = [-L/2, L/2, -L/2, L/2] # position sur l'axe y_effecteur des points d'accroche


## Modèle inverse - variables - position de l'effecteur
X_e = sympy.symbols("X_e")
Y_e = sympy.symbols("Y_e")
phi_1 = sympy.symbols("phi_1")

# Position intiale de l'effecteur - au centre du repère
X_0 = 1075  # Position initiale du centre de la plaque en X
Y_0 = 1075  # Position initiale du centre de la plaque en Y 


## Modèle inverse - équations modèle analytique
lambda_1 =   sqrt((X_0 + X_e - X[0] + a[0]*cos(phi_1) - b[0]*sin(phi_1))**2 + (Y_0 + Y_e - Y[0] + a[0]*sin(phi_1) + b[0]*cos(phi_1))**2) # longueur du câble de la poulie 1 (en mm)
lambda_2 =   sqrt((X_0 + X_e - X[1] + a[1]*cos(phi_1) - b[1]*sin(phi_1))**2 + (Y_0 + Y_e - Y[1] + a[1]*sin(phi_1) + b[1]*cos(phi_1))**2) # longueur du câble de la poulie 2 (en mm)
lambda_3 =   sqrt((X_0 + X_e - X[2] + a[2]*cos(phi_1) - b[2]*sin(phi_1))**2 + (Y_0 + Y_e - Y[2] + a[2]*sin(phi_1) + b[2]*cos(phi_1))**2) # longueur du câble de la poulie 3 (en mm)
lambda_4 =   sqrt((X_0 + X_e - X[3] + a[3]*cos(phi_1) - b[3]*sin(phi_1))**2 + (Y_0 + Y_e - Y[3] + a[3]*sin(phi_1) + b[3]*cos(phi_1))**2) # longueur du câble de la poulie 4 (en mm)
# ---
q_1 = lambda_1 / r # angle de rotation du moteur 1 (en rad)
q_2 = lambda_2 / r # angle de rotation du moteur 2 (en rad)
q_3 = lambda_3 / r # angle de rotation du moteur 3 (en rad)
q_4 = lambda_4 / r # angle de rotation du moteur 4 (en rad)
# ---
p_1 = (200*q_1) / 2*np.pi # nombre de pas sur le moteur 1
p_2 = (200*q_2) / 2*np.pi # nombre de pas sur le moteur 2
p_3 = (200*q_3) / 2*np.pi # nombre de pas sur le moteur 3
p_4 = (200*q_4) / 2*np.pi # nombre de pas sur le moteur 4



################################# Méthode 1 ###################################

## Modèle inverse - Calcul de la Jacobienne du modèle cinématique inverse
def inverse():
    X = Matrix([X_e, Y_e, phi_1]) # Entrées
    Y = Matrix([p_1, p_2, p_3, p_4]) # Sorties
    J = Y.jacobian(X) # Jacobienne du modèle cinématique inverse 
    print("---- Modèle cinématique inverse ---- ")
    print("Variables d'entrée : \n", X)
    print("\nVariables de sortie : \n", Y)
    print("\nJacobienne : \n", J)
    return J


## Modèle inverse - Calcul de la Jacobienne du modèle cinématique inverse et test
def inverse_test(X_e_exp, Y_e_exp, phi_1_exp, temps):
    X_entrées = Matrix([X_e, Y_e, phi_1]) # Entrées
    Y_sorties = Matrix([lambda_1, lambda_2, lambda_3, lambda_4]) # Sorties
    J = Y_sorties.jacobian(X_entrées) # Jacobienne du modèle cinématique inverse 
    print("\n---- Modèle cinématique inverse pour test  ---- ")
    #-  print("Variables d'entrée : \n", X)
    #- print("\nVariables de sortie : \n", Y)
    #- print("\nJacobienne : \n", J)

    J = J.subs({X_e: X_e_exp, Y_e: Y_e_exp, phi_1: phi_1_exp})  # Substitution des valeurs d'entrées sur la Jacobienne
    
    # Création du vecteur des vitesses des l'effecteur, sur l'axe x, sur l'axe y et de rotation autour de l'axe z
    V_X_entrées = X_entrées.subs({X_e: X_e_exp/temps, Y_e: Y_e_exp/temps, phi_1: phi_1_exp/temps})  

    #- print("\nJacobienne  avec les valeurs expérimentales: \n", J) 
    V_lambda_exp = np.dot(J, V_X_entrées) # calcul le produit matriciel de la Jacobienne et de la matrice des variables d'entrées
    D_lambda_exp = V_lambda_exp*temps
    
    # Ajout de la longueur de base pour l'effecteur au centre
    longueur_initial = 930     
    for i in range(0, 4):     
        D_lambda_exp[i][0] = D_lambda_exp[i][0] + longueur_initial
        
    
    print("Pour un déplacement de: (", X_e_exp, "," , Y_e_exp, "), par rapport au point central, et un angle de :", phi_1_exp)
    print("\nVitesse de déroulage/enroulage de câble par les moteurs (en mm)")
    print("{'+' : dérouler du câble, '-': enrouler du câble} :")
    i, j = np.shape(V_lambda_exp)
    for k in  range(0, i):
        print("\nCâble ", k+1, " :")
        print("Vitesse : ", round(float(V_lambda_exp[k][0]), 3), " mm/s") 
        print("Longeur du câble : ", round(float(V_lambda_exp[k][0]*temps), 3), "mm") 

    return V_lambda_exp, D_lambda_exp


def animation_test(longueur_1, longueur_2, longueur_3, longueur_4):
    print("\nlongueur 1 = ", longueur_1)
    print("\nlongueur 2 = ", longueur_2)
    print("\nlongueur 3 = ", longueur_3)
    print("\nlongueur 4 = ", longueur_4)

    cable_lengths = {
        "Câble 1": longueur_1,
        "Câble 2": longueur_2,
        "Câble 3": longueur_3,
        "Câble 4": longueur_4,
    }

    num_frames = len(longueur_1)
    plaque_size = 230

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 2150)
    ax.set_ylim(0, 2150)
    ax.set_zlim(0, 500)
    ax.set_title("Déplacement de la plaque dans le plan XY")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Repère 3D
    ax.quiver(0, 0, 0, 100, 0, 0, color='r', arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 100, 0, color='g', arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, 100, color='b', arrow_length_ratio=0.05)

    anchor_points = {
        "Câble 1": [2150, 0, 0],
        "Câble 2": [2150, 2150, 0],
        "Câble 3": [0, 0, 0],
        "Câble 4": [0, 2150, 0],
    }

    lines = []
    texts = []
    plaque = None

    def update(frame):
        
        nonlocal lines, texts, plaque

        for line in lines:
            line.remove()
        for text in texts:
            text.remove()
        if plaque:
            plaque.remove()

        lines = []
        texts = []

        # Longueurs des câbles à cet instant
        l1 = cable_lengths["Câble 1"][frame]
        l2 = cable_lengths["Câble 2"][frame]
        l3 = cable_lengths["Câble 3"][frame]
        l4 = cable_lengths["Câble 4"][frame]
        
        print("\nl1 = ", l1)
        print("l2 = ", l2)
        print("l3 = ", l3)
        print("l4 = ", l4)
        

        # Calcul et affichage du centre de la plaque
        dx = (l3 + l4 - l1 - l2) * 0.25
        dy = (l1 + l3 - l2 - l4) * 0.25
        cx = 1075 + dx
        cy = 1075 + dy
        cz = 0
        print("Centre de la plaque en :", round(cx, 3), round(cy, 3))
        
    

        half = plaque_size / 2
        p1 = [cx + half, cy - half, cz]
        p2 = [cx + half, cy + half, cz]
        p3 = [cx - half, cy - half, cz]
        p4 = [cx - half, cy + half, cz]

        for name, anchor, corner in zip(
            ["Câble 1", "Câble 2", "Câble 3", "Câble 4"],
            [anchor_points["Câble 1"], anchor_points["Câble 2"], anchor_points["Câble 3"], anchor_points["Câble 4"]],
            [p1, p2, p3, p4],
        ):
            line = ax.plot([anchor[0], corner[0]], [anchor[1], corner[1]], [anchor[2], corner[2]], 'gray')[0]
            lines.append(line)

            mid = [(anchor[i] + corner[i]) / 2 for i in range(3)]
            text = ax.text(mid[0], mid[1], mid[2] + 10, name, color='black', fontsize=9, ha='center')
            texts.append(text)

        # Matrices pour le plot_surface
        X = np.array([[p1[0], p2[0]], [p3[0], p4[0]]], dtype=float)
        Y = np.array([[p1[1], p2[1]], [p3[1], p4[1]]], dtype=float)
        Z = np.array([[cz, cz], [cz, cz]], dtype=float)

        plaque = ax.plot_surface(X, Y, Z, color='skyblue', alpha=0.8)

        return lines + [plaque] + texts
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=600, blit=False)
    return ani


def animation_1(X_e_final, Y_e_final, phi_1_final, nombre_iteration, temps):
    """
    
    Parameters
    ----------
    X_e_final : {float} Coordonnées sur l'axe x finale du centre de l'effecteur
    Y_e_final : {float} Coordonnées sur l'axe y finale du centre de l'effecteur 
    phi_1_final : {float} Angle de rotation finale autour de l'axe z du centre l'effecteur
    nombre_iteration : {int} Nombre ditération pour la simulation
    temps : {int} Temps necéssaire pour effectuer la simulation

    Returns
    -------
    None.

    """
    X1 = np.linspace(0, X_e_final, nombre_iteration)
    X2 = np.linspace(0, Y_e_final, nombre_iteration)
    PHI = np.linspace(0, phi_1_final, nombre_iteration)
    Etape = np.arange(nombre_iteration)

    D1, D2, D3, D4 = [], [], [], []
    V1, V2, V3, V4 = [], [], [], []

    for i, j, k in zip(X1, X2, PHI):
        vitesse_cable_actuel, deplacement_cable_actuel = inverse_test(i, j, k, temps)
        print("\ndeplacement_cable_actuel: ", deplacement_cable_actuel)

        D1.append(deplacement_cable_actuel[0][0])
        D2.append(deplacement_cable_actuel[1][0])
        D3.append(deplacement_cable_actuel[2][0])
        D4.append(deplacement_cable_actuel[3][0])

        V1.append(vitesse_cable_actuel[0][0])
        V2.append(vitesse_cable_actuel[1][0])
        V3.append(vitesse_cable_actuel[2][0])
        V4.append(vitesse_cable_actuel[3][0])
        
        # Détection des distances de câbles négatives
        if D1[-1] < 0 or D2[-1] < 0 or D3[-1] < 0 or D4[-1] < 0:
            D1[-1] = D1[-2]
            D2[-1] = D2[-2]
            D3[-1] = D3[-2]
            D4[-1] = D4[-2]
            
            V1[-1] = 0
            V2[-1] = 0
            V3[-1] = 0
            V4[-1] = 0
            
       
    # Plot des déplacements
    fig_1, axs_1 = plt.subplots(nrows=2, ncols=2)
    fig_1.suptitle("Tailles des câbles")
    for ax, D, title in zip(axs_1.flat, [D1, D2, D3, D4], ["Câble 1", "Câble 2", "Câble 3", "Câble 4"]):
        ax.plot(Etape, D, marker='o')
        ax.set_title(title)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Déplacement [mm]")
        ax.grid()

    # Plot des vitesses
    fig_2, axs_2 = plt.subplots(nrows=2, ncols=2)
    fig_2.suptitle("Vitesses linéaires des câbles")
    for ax, V, title in zip(axs_2.flat, [V1, V2, V3, V4], ["Câble 1", "Câble 2", "Câble 3", "Câble 4"]):
        ax.plot(Etape, V, marker='o')
        ax.set_title(title)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Vitesse [mm/s]")
        ax.grid()

    # Animation
    global ani
    ani = animation_test(D1, D2, D3, D4)

    # Tracée de la trajectoire estimé
    CX = []
    CY = []
    
    for l1, l2, l3, l4 in zip(D1, D2, D3, D4):
        dx = (l3 + l4 - l1 - l2) * 0.25
        dy = (l1 + l3 - l2 - l4) * 0.25
        cx = 1075 + dx
        cy = 1075 + dy
        CX.append(cx)
        CY.append(cy)
    
    # Plot position estimée
    fig_pos, ax_pos = plt.subplots()
    ax_pos.plot(CX, CY, label="Trajectoire estimée", marker='X')
    ax_pos.set_title("Position estimée du centre de la plaque")
    ax_pos.set_xlabel("X [mm]")
    ax_pos.set_ylabel("Y [mm]")
    ax_pos.grid()
    ax_pos.legend()

    # Affichage
    plt.show()

        

    

## Modèle direct - Calcul de la pseudo inverse de la Jacobienne
def direct():
    J = inverse()
    print("\n---- Modèle cinématique direct ---- ")
    print("Variables d'entrée : \n", Y)
    print("\nVariables de sortie : \n", X)
    
    # J = J.subs({X_e: 1, Y_e: 2, phi_1: 3})  # Substitution temporaire des valeurs variables
    
    J_T = np.transpose(J) 
    print("\nJ_T : \n", J_T)
    print("\nnp.shape(J_T) : \n", np.shape(J_T))
    
    J_prod = sympy.Matrix(np.dot(J_T, J))
    print("\nJ_prod : \n", J_prod)
    print("\nnp.shape(J_prod) : \n", np.shape(J_prod))
    print("\nJ_prod.det() : \n", J_prod.det())
    
    J_prod_inv = J_prod.inv()
    print("\nJ_prod_inv : \n", J_prod_inv)
    print("\nnp.shape(J_prod_inv) : \n", np.shape(J_prod_inv))
    
    J_pseudo_inv = np.dot(J_prod_inv, J_T)
    print("\nJ_pseudo_inv : \n", J_pseudo_inv)
    print("\nnp.shape(J_pseudo_inv) : \n", np.shape(J_pseudo_inv))
    

################################# Méthode 2 ###################################


# Positions des poulies (points fixes) dans le plan
poulies = np.array([
    [l_1, h_1],  # P1 (en bas à droite)
    [l_1, h_2],  # P2 (en haut à droite)
    [0, h_1],    # P3 (en bas à gauche)
    [0, h_2]     # P4 (en haut à gauche)
])


# Coordonnées des points d'attache de la plaque (dans son repère local)
v_attache = np.array([
    [ l/2, -L/2],  # Coin bas droite
    [ l/2,  L/2],  # Coin haut droite
    [-l/2, -L/2],  # Coin bas gauche
    [-l/2,  L/2]   # Coin haut gauche
])


# Matrice de rotation 2D selon un angle phi
def rotation_matrix(phi):
    return np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)]
    ])


# Calcule les coordonnées globales des points d'attache de la plaque
def compute_attachment_points(X, Y, phi):
    R = rotation_matrix(phi)
    return np.array([[X, Y]]) + (v_attache @ R.T)


# Calcule les longueurs de câble entre poulies et coins de la plaque
def cable_lengths(X, Y, phi):
    A = compute_attachment_points(X, Y, phi)
    return np.linalg.norm(A - poulies, axis=1)


# Calcule la matrice Jacobienne ∂l/∂q (variation des longueurs par rapport à X, Y, phi)
def jacobian(X, Y, phi):
    A = compute_attachment_points(X, Y, phi)
    R = rotation_matrix(phi)
    J = np.zeros((4, 3))
    for i in range(4):
        diff = A[i] - poulies[i]  # vecteur du point poulie vers attache
        d = np.linalg.norm(diff)  # longueur du câble
        if d == 0:
            continue  # évite division par zéro
        dX = diff[0] / d  # dérivée partielle par rapport à X
        dY = diff[1] / d  # dérivée partielle par rapport à Y
        dphi_vec = R @ np.array([-v_attache[i][1], v_attache[i][0]])  # rotation du vecteur d’attache
        dphi = np.dot(diff, dphi_vec) / d  # dérivée partielle par rapport à phi
        J[i, :] = [dX, dY, dphi]
    return J


# Fonction principale de simulation
def animation_2(X_inital, Y_initial, phi_1_initial, X_final, Y_final, phi_1_final, step, nb_points, epsilon):
    """
    Parameters
    ----------
    X_intial : {float} Coordonnées sur l'axe x finale du centre de l'effecteur initialement
    Y_initial : {float} Coordonnées sur l'axe y finale du centre de l'effecteur initialement
    phi_1_initial : {float} Angle de rotation finale autour de l'axe z du centre l'effecteur initialement
    
    X_final : {float} Coordonnées sur l'axe x finale du centre de l'effecteur
    Y_final : {float} Coordonnées sur l'axe y finale du centre de l'effecteur 
    phi_1_final : {float} Angle de rotation finale autour de l'axe z du centre l'effecteur
    
    step : {float} Taille des pas de la simulation
    nb_points : {int} Nombre de points pour effectuer la simulation
    epsilon : {int} Valeur en % de la bande d'arrêt

    Returns
    -------
    Cette fonction fait la simulation du déplcement de l'effecteur du robot parallèle à câble de sa postion inital (le centre du repère)
    à une position final de coordonée X_final, Y_final et avec un angle phi_1_final.
    Elle affiche différent plot:
        - le 1er est l'animation liée à cette simulation
        - Le 2nd est constitué de 4 subplot qui chacun affichent la longueurs des 4 câbles en fonction des itérations
        - Le 3ème  affiche la position du centre de l'effecteur en fonction de l'itération choisi
        - le 4ème affiche la rotation du centre de l'effecteur en fonction de l'itération choisi
        - Le 5ème affiche les vitesses linéaires des câbles en fonction de l'itération choisi
        - Le 6ème affiche les vitesse de rotation des moteur en fonction de l'itération choisi 
        """

    # Initialisation à la position centrale
    X0, Y0 , phi_1_0= X_inital, Y_initial, phi_1_initial
    print("\nPosition initiale: X0 = ", X0, "Y0 = ", Y0, "phi_1_0 = ", phi_1_0)
    X_traj, Y_traj, phi_traj = [X0], [Y0], [0.0]
    
    # Initialisation des longueurs de câbles
    l_0 = cable_lengths(X0, Y0, 0.0)
    l_traj = [l_0]
    print("Longueurs des câbles initiales: ",l_0)
    
    X, Y, phi_1 = X0, Y0, phi_1_0
   
    # Initialisation des vitesses
    v_traj = []      
    
    # Marge d'erreur sur les valeurs finales
    epsilon_plus = 1 + (epsilon/100)
    epsilon_moins = 1 - (epsilon/100)

    # Boucle de simulation
    for i in range(nb_points):
        dx = X_final - X
        dy = Y_final - Y
        dphi = phi_1_final - phi_1
        error = np.array([dx, dy, dphi])
        if np.linalg.norm(error) < 1e-4:
            break  # Sortie de la boucle si l’erreur est négligeable

        J = jacobian(X, Y, phi_1)  # Jacobienne à l’instant courant
        dl = J @ error * step  # Variation attendue des longueurs
        lambda_reg = 1e-4  # Paramètre de régularisation
        J_pseudo_inv = pinv(J.T @ J + lambda_reg * np.eye(3)) @ J.T  # Pseudo-inverse de la Jacobienne
        delta_q = J_pseudo_inv @ (l_traj[-1] + dl - l_traj[-1])  # Variation de position

        # Mise à jour de la position
        X += delta_q[0]
        Y += delta_q[1]
        phi_1 += delta_q[2]
        print("\nPosition : X = ", X, "Y = ", Y, "phi_1 = ", phi_1)
        
        X_traj.append(X)
        Y_traj.append(Y)
        phi_traj.append(phi_1)

        # Calcul de la nouvelle longueur et vitesse des câbles
        l_curr = cable_lengths(X, Y, phi_1)
        l_prev = l_traj[-1]
        v = (l_curr - l_prev) / step  # Vitesse estimée
        v_traj.append(v)
        l_traj.append(l_curr)
        print("Longueurs des câbles : ",l_curr)

            
        # Arrêt si on a atteint la position finale à avec une marge de [Valeur finale * epsilon_moins; Valeur finale * epsilon_plus]
        if X_final*epsilon_moins <= X <=  X_final * epsilon_plus and Y_final*epsilon_moins <= Y <=  Y_final * epsilon_plus and phi_1_final*epsilon_moins <= phi_1 <=  phi_1_final * epsilon_plus:
            print("\nArrêt à l'itération:", i,"\n")
            break
        
        
    # Fonction d’animation graphique
    def anim():
        fig, ax = plt.subplots()
        ax.set_xlim(-200, l_1 + 200)
        ax.set_ylim(-200, h_2 + 200)
        ax.set_aspect('equal')
        plate, = ax.plot([], [], 'b-', lw=2)
        cables, = ax.plot([], [], 'k--', lw=1)
        center, = ax.plot([], [], 'ro')

        def update(frame):
            X, Y, phi_1 = X_traj[frame], Y_traj[frame], phi_traj[frame]
            A = compute_attachment_points(X, Y, phi_1)
            plate.set_data(A[:, 0].tolist() + [A[0, 0]], A[:, 1].tolist() + [A[0, 1]])
            cable_x, cable_y = [], []
            for i in [0, 1, 3, 2]:
                cable_x += [poulies[i, 0], A[i, 0], None]
                cable_y += [poulies[i, 1], A[i, 1], None]
            cables.set_data(cable_x, cable_y)
            center.set_data([X], [Y])
            return plate, cables, center

        ani = FuncAnimation(fig, update, frames=len(X_traj), interval=50, blit=True)
        return ani

    global ani
    ani = anim()
    plt.title("Simulation du Robot Parallèle à Câbles")

    # Tracé des longueurs de câble
    fig_var, axs_var = plt.subplots(nrows=2, ncols=2)
    fig_var.suptitle("Tailles des câbles")
    l_traj_vec = np.array(l_traj)
    D1, D2, D3, D4 = l_traj_vec[:,0], l_traj_vec[:,1], l_traj_vec[:,2], l_traj_vec[:,3]
    for ax, D, title, color in zip(axs_var.flat, [D4, D2, D3, D1], ["Câble 4", "Câble 2", "Câble 3", "Câble 1"], ["red", "green", "blue", "purple"]):
        Etape = np.linspace(0, nb_points, np.shape(D1)[0])
        ax.plot(Etape, D, marker='o', color=color)
        ax.set_title(title)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Longueur de câble [mm]")
        ax.grid()

    # Tracé de la position du centre de l’effecteur
    fig_pos, ax_pos = plt.subplots()
    fig_pos.suptitle("Position du centre de l'effecteur")
    ax_pos.plot(X_traj, Y_traj, marker='o', color='orangered')
    ax_pos.set_xlabel("X [mm]")
    ax_pos.set_ylabel("Y [mm]")
    ax_pos.grid()
    
    # Tracé de la rotation du centre de l’effecteur
    fig_rot, ax_rot = plt.subplots()
    fig_rot.suptitle("Rotation du centre de l'effecteur")
    ax_rot.plot(Etape, phi_traj, marker='o', color='grey')
    ax_rot.set_xlabel("Itération")
    ax_rot.set_ylabel("Angle [rad]")
    ax_rot.grid()

    # Tracé des vitesses linéaires des câbles
    fig_vit_lin, axs_vit_lin = plt.subplots(nrows=2, ncols=2)
    fig_vit_lin.suptitle("Vitesses linéaires des câbles")
    v_traj_vec = np.array(v_traj)
    V1, V2, V3, V4 = v_traj_vec[:,0], v_traj_vec[:,1], v_traj_vec[:,2], v_traj_vec[:,3]
    Etape_v = np.arange(len(V1))  # une étape de moins que les longueurs

    for ax, V, title, color in zip(axs_vit_lin.flat, [V4, V2, V3, V1], ["Câble 4", "Câble 2", "Câble 3", "Câble 1"], ["red", "green", "blue", "purple"]):
        ax.plot(Etape_v, V, marker='o', color=color)
        ax.set_title(title)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Vitesse [mm/itération]")
        ax.grid()
        
    # Tracé des vitesses angulaires des moteurs
    fig_vit_ang, axs_vit_ang = plt.subplots(nrows=2, ncols=2)
    fig_vit_ang.suptitle("Vitesses angulaires des moteurs")
    v_traj_vec = np.array(v_traj)
    Omega1, Omega2, Omega3, Omega4 = r*v_traj_vec[:,0], r*v_traj_vec[:,1], r*v_traj_vec[:,2], r*v_traj_vec[:,3]
    Etape_v = np.arange(len(V1))  # une étape de moins que les longueurs

    for ax, Omega, title, color in zip(axs_vit_ang.flat, [Omega4, Omega2, Omega3, Omega1], ["Moteur 4", "Moteur 2", "Moteur 3", "Moteur 1"], ["red", "green", "blue", "purple"]):
        ax.plot(Etape_v, Omega, marker='o', color=color)
        ax.set_title(title)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Vitesse [rad/itération]")
        ax.grid()

    # Affichage de tous les graphes
    plt.show()



