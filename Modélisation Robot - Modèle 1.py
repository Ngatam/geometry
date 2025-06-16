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

from numpy import linalg
from sympy import sin, cos, sqrt, Matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sympy.abc import alpha, beta, phi, delta, theta, psi, omega

############################# Paramètres du système ###########################

## Paramètres du système
h_1 = 400 # (en mm) hauteur poulie 1
h_2 = 2180 # (en mm) hauteur poulie 2
l = 230 # (en mm) largeur de la plaque de l'effecteur
L = 230 # (en mm) longueur de la plaque de l'effecteur
l_1 = 1900 # (en mm) distance entre 2 poulie de la structure
K = 0.5 # rapport de transmission de l'enrouleur
e = 30 # (en mm) rayon de l'enrouleur 
rho = 5 # (en mm) pas de l'enrouleur
pas_mot = 1.8 # (en °) pas du moteur => 200 pas pour 1 tour

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

# Position intiale de l'effecteur - au centre du repère
X_0 = 1075  # Position initiale du centre de la plaque en X
Y_0 = 1090  # Position initiale du centre de la plaque en Y 

## Modèle inverse - variables - position de l'effecteur
X_e = sympy.symbols("X_e") # position de l'effecteur sur l'axe X de la structure
Y_e = sympy.symbols("Y_e") # position de l'effecteur sur l'axe Y de la structure
phi_1 = sympy.symbols("phi_1") # position angulaire de l'effecteur par rapport au repère de la base



############################# Equations du système ############################

## Coefficients pour le calcul
r = K * (e**2 + (rho**2)/2*np.pi)**1/2 # coefficient d'enroulement des enrouleurs
X = [l_1, l_1, 0, 0] # position sur l'axe x_base des poulies
Y = [h_1, h_2, h_1, h_2] # position sur l'axe y_base des poulies 
a = [l/2, l/2, -l/2, -l/2] # position sur l'axe x_effecteur des points d'accroche
b = [-L/2, L/2, -L/2, L/2] # position sur l'axe y_effecteur des points d'accroche

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


################################# Modèle 1 ####################################


# Calcul les vitesses et longueurs de câble pour un déplacement de l’effecteur en X_e, Y_e, phi_1 sur une durée "temps"
def inverse_test(X_e_exp, Y_e_exp, phi_1_exp, temps):
    """
    
    Parameters
    ----------
    X_e_exp : {float} Coordonnée finale de l'effecteur sur l'axe x
    Y_e_exp : {float} Coordonnée finale de l'effecteur sur l'axe y
    phi_1_exp : {float} Coordonnée finale la rotation de l'effecteur autour de l'axe z
    temps : {int} temps de la simulation

    Returns
    -------
    V_lambda_exp : {array} Vecteur des vitesses linéaires des câbles
    D_lambda_exp : {array} Vecteur des longueurs de câbles

    """
    X_vars = Matrix([X_e, Y_e, phi_1])
    Y_out = Matrix([lambda_1, lambda_2, lambda_3, lambda_4])
    J = Y_out.jacobian(X_vars)
    J = J.subs({X_e: X_e_exp, Y_e: Y_e_exp, phi_1: phi_1_exp})

    V_X = X_vars.subs({X_e: X_e_exp/temps, Y_e: Y_e_exp/temps, phi_1: phi_1_exp/temps})
    V_lambda_exp = np.dot(J, V_X)
    D_lambda_exp = V_lambda_exp * temps

    longueur_initial = 1320
    for i in range(4):
        D_lambda_exp[i][0] += longueur_initial

    return V_lambda_exp, D_lambda_exp


# Crée une animation 3D du mouvement de la plaque à partir des longueurs de câbles
def animation_test(longueur_1, longueur_2, longueur_3, longueur_4):
    """
    
    Parameters
    ----------
    longueur_1 : {float} Longueur courante du câble 1
    longueur_2 : {float} Longueur courante du câble 2
    longueur_3 : {float} Longueur courante du câble 3
    longueur_4 : {float} Longueur courante du câble 4

    Returns
    -------
    ani : animation à l'instant courant

    """
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
        for l in lines: l.remove()
        for t in texts: t.remove()
        if plaque: plaque.remove()

        lines.clear()
        texts.clear()

        l1 = cable_lengths["Câble 1"][frame]
        l2 = cable_lengths["Câble 2"][frame]
        l3 = cable_lengths["Câble 3"][frame]
        l4 = cable_lengths["Câble 4"][frame]

        dx = (l2 + l4 - l1 - l3) * 0.25
        dy = (l1 + l2 - l3 - l4) * 0.25
        cx = X_0 + dx
        cy = Y_0 + dy
        cz = 0

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

        X = np.array([[p1[0], p2[0]], [p3[0], p4[0]]], dtype=float)
        Y = np.array([[p1[1], p2[1]], [p3[1], p4[1]]], dtype=float)
        Z = np.array([[cz, cz], [cz, cz]], dtype=float)

        plaque = ax.plot_surface(X, Y, Z, color='skyblue', alpha=0.8)
        return lines + [plaque] + texts

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=300, blit=False)
    print("\n Animation lancée !")
    plt.show()
    return ani


# Simule le déplacement de l’effecteur avec un mouvement linéaire + rotation
# Affiche les courbes des longueurs et vitesses de câbles, animation 3D
# et une estimation de trajectoire du centre de la plaque
def inverse_plot(X_e_final, Y_e_final, phi_1_final, nombre_etape, temps):
    """

    Parameters
    ----------
    X_e_final : {float} Coordonnée finale de l'effecteur sur l'axe x
    Y_e_final : {float} Coordonnée finale de l'effecteur sur l'axe y
    phi_1_final : {float} Coordonnée finale la rotation de l'effecteur autour de l'axe z
    nombre_etape : {int} Nombre d'étape de la simulation
    temps : {int} Temps de la simulation

    Returns
    -------
    None.

    """
    X1 = np.linspace(X_0, X_e_final, nombre_etape)
    X2 = np.linspace(Y_0, Y_e_final, nombre_etape)
    PHI = np.linspace(0, phi_1_final, nombre_etape)
    Etape = np.arange(nombre_etape)

    D1, D2, D3, D4 = [], [], [], []
    V1, V2, V3, V4 = [], [], [], []

    for i, j, k in zip(X1, X2, PHI):
        vitesse, deplacement = inverse_test(i, j, k, temps)
        D1.append(deplacement[0][0])
        D2.append(deplacement[1][0])
        D3.append(deplacement[2][0])
        D4.append(deplacement[3][0])

        V1.append(vitesse[0][0])
        V2.append(vitesse[1][0])
        V3.append(vitesse[2][0])
        V4.append(vitesse[3][0])

    # Plot des longueurs
    fig_1, axs_1 = plt.subplots(2, 2)
    fig_1.suptitle("Tailles des câbles")
    for ax, D, title in zip(axs_1.flat, [D1, D2, D3, D4], ["Câble 1", "Câble 2", "Câble 3", "Câble 4"]):
        ax.plot(Etape, D)
        ax.set_title(title)
        ax.grid()

    # Plot des vitesses
    fig_2, axs_2 = plt.subplots(2, 2)
    fig_2.suptitle("Vitesses linéaires des câbles")
    for ax, V, title in zip(axs_2.flat, [V1, V2, V3, V4], ["Câble 1", "Câble 2", "Câble 3", "Câble 4"]):
        ax.plot(Etape, V)
        ax.set_title(title)
        ax.grid()

    global ani
    ani = animation_test(D1, D2, D3, D4)

    # Estimation de position
    CX, CY = [], []
    for l1, l2, l3, l4 in zip(D1, D2, D3, D4):
        dx = (l2 + l4 - l1 - l3) * 0.25
        dy = (l1 + l2 - l3 - l4) * 0.25
        cx = 1075 + dx
        cy = 1075 + dy
        CX.append(cx)
        CY.append(cy)

    fig_pos, ax_pos = plt.subplots()
    ax_pos.plot(CX, CY, label="Trajectoire estimée", marker='o')
    ax_pos.set_title("Position estimée du centre de la plaque")
    ax_pos.set_xlabel("X [mm]")
    ax_pos.set_ylabel("Y [mm]")
    ax_pos.grid()
    ax_pos.legend()

    plt.show()


################################# Appel principal ######################################

#if __name__ == "__main__":
#    inverse_plot(X_e_final=300, Y_e_final=200, phi_1_final=0.3, nombre_etape=30, temps=5)
