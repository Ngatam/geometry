# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:08:22 2025

@author: Ngatam
"""
################################# Import ######################################

import math
import sympy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpy.linalg import pinv
from sympy import sin, cos, sqrt, Matrix
from matplotlib.animation import FuncAnimation
from sympy.abc import alpha, beta, delta, theta, psi, omega



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
r_p = 40 #(en mm) rayon interne des poulies
lambda_troisieme = 740 # (en mm) longueur de cable supposée constante entre l'enrouleur et la poulie (pas de centre a centre)

## Coefficients pour le calcul
r = K * (e**2 + (rho**2)/2*np.pi)**1/2 # coefficient d'enroulement des enrouleurs
X = [l_1, l_1, 0, 0] # position sur l'axe x_base des poulies
Y = [h_1, h_2, h_1, h_2] # position sur l'axe y_base des poulies 
a = [l/2, l/2, -l/2, -l/2] # position sur l'axe x_effecteur des points d'accroche
b = [-L/2, L/2, -L/2, L/2] # position sur l'axe y_effecteur des points d'accroche

## Positions des poulies (points fixes) dans le plan
poulies = np.array([
    [l_1, h_1],  # P1 (en bas à droite)
    [l_1, h_2],  # P2 (en haut à droite)
    [0, h_1],    # P3 (en bas à gauche)
    [0, h_2]     # P4 (en haut à gauche)
])

## Coordonnées des points d'attache de la plaque (dans son repère local)
v_attache = np.array([
    [ l/2, -L/2],  # Coin bas droite
    [ l/2,  L/2],  # Coin haut droite
    [-l/2, -L/2],  # Coin bas gauche
    [-l/2,  L/2]   # Coin haut gauche
])

## Position intiale de l'effecteur - au centre du repère
X_0 = 1010  # Position initiale du centre de la plaque en X
Y_0 = 1075  # Position initiale du centre de la plaque en Y 

    

################################# Modèle 2 ####################################

# Matrice de rotation 2D selon un angle phi
def rotation_matrix(phi):
    """

    Parameters
    ----------
    phi : {float} Angle de rotation autour de l'axe z en °

    Returns
    -------
    R : {array} Matrice de rotation en 2D autour de l'axe z

    """
    R = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)]
    ])

    return R


# Calcul des coordonnées globales des points d'attache de la plaque
def compute_attachment_points(X, Y, phi):
    """

    Parameters
    ----------
    X : {float} Coordonnée sur x du point d'attache sur l'effecteur
    Y : {float} Coordonnée sur y du point d'attache sur l'effecteur
    phi : {float} Rotation en ° autour de l'axe z

    Returns
    -------
    M : {array} Coordonées dans le repère global des points d'attache de l'effecteur

    """
    R = rotation_matrix(phi)
    M = np.array([[X, Y]]) + (v_attache @ R.T)
    return M


# Calcul des longueurs de câble entre poulies et coins de la plaque
def cable_lengths(X, Y, phi):
    """

    Parameters
    ----------
    X : {float} Coordonnée sur x du point d'attache sur l'effecteur
    Y : {float} Coordonnée sur y du point d'attache sur l'effecteur
    phi : {float} Rotation en ° autour de l'axe z

    Returns
    -------
    L : {float} Distance entre le point d'attache i sur l'effecteur et le centre de la poulie. 

    """
    A = compute_attachment_points(X, Y, phi) # Calcul les coordonées des points d'attache de la plaque
    L = np.linalg.norm(A - poulies, axis=1) # Renvoie la norme de la matrice qui est la différence des coordonées des points d'attache de la plaque actuellement et les coordonnées des poulies initialement
    return L


# Calcul de la Jacobienne d_rond lambda/ d_rond X (variation des longueurs des câbles par rapport à X, Y, phi)
def jacobian(X, Y, phi):
    """

    Parameters
    ----------
    X : {float} Coordonnée sur x du point d'attache sur l'effecteur
    Y : {float} Coordonnée sur y du point d'attache sur l'effecteur
    phi : {float} Rotation en ° autour de l'axe z

    Returns
    -------
    J : {array} Jacobienne d_rond lambda/ d_rond X (variation des longueurs des câbles par rapport à X, Y, phi)

    """
    A = compute_attachment_points(X, Y, phi)  # Calcul les coordonées des points d'attache de la plaque
    R = rotation_matrix(phi) # Matrice de rotation autour de l'axe z
    J = np.zeros((4, 3)) # Initialisation de la Jacobienne
    for i in range(4):
        diff = A[i] - poulies[i]  # Vecteur du point poulie vers point d'attache i
        d = np.linalg.norm(diff)  # Longueur du câble i
        if d == 0: 
            continue  # Evite division par zéro
        dX = diff[0] / d  # Dérivée partielle par rapport à X
        dY = diff[1] / d  # Dérivée partielle par rapport à Y
        dphi_vec = R @ np.array([-v_attache[i][1], v_attache[i][0]])  # Rotation du vecteur d’attache
        dphi = np.dot(diff, dphi_vec) / d  # Dérivée partielle par rapport à phi
        J[i, :] = [dX, dY, dphi] # Remplissage de la Jacobienne sur la ligne i
    return J


# Rectification de l'expression de lambda (en prenant en compte le rayon de la poulie et la longueur totale du cable depuis l'enrouleur jusqu'à l'effecteur)
def rectification_lambda(lambda_1, lambda_2, lambda_3, lambda_4, Y_e):
    """

    Parameters
    ----------
    lambda_1 : {float} longueur de câble 1 sans complexification du modèle. Distance entre le centre de la poulie 1 et le pooint d'attache 1 sur l'effecteur.
    lambda_2 : {float} longueur de câble 2 sans complexification du modèle. Distance entre le centre de la poulie 2 et le pooint d'attache 2 sur l'effecteur.
    lambda_3 : {float} longueur de câble 3 sans complexification du modèle. Distance entre le centre de la poulie 3 et le pooint d'attache 3 sur l'effecteur.
    lambda_4 : {float} longueur de câble 4 sans complexification du modèle. Distance entre le centre de la poulie 4 et le pooint d'attache 4 sur l'effecteur.

    Returns
    -------
    lambda_tot_1 : {float} longueur de câble total depuis le tambour 1 jusqu'au point d'attache 1 sur l'effecteur
    lambda_tot_2 : {float} longueur de câble total depuis le tambour 2 jusqu'au point d'attache 2 sur l'effecteur
    lambda_tot_3 : {float} longueur de câble total depuis le tambour 3 jusqu'au point d'attache 3 sur l'effecteur
    lambda_tot_4 : {float} longueur de câble total depuis le tambour 4 jusqu'au point d'attache 4 sur l'effecteur

    """    
    # lambda_prime_i : correction sur la longueur de câble entre la poulie i et le point d'accroche i sur l'effecteur
    lambda_prime_1 = sqrt(float(lambda_1)**2 + r_p**2)
    lambda_prime_2 = sqrt(float(lambda_2)**2 + r_p**2)
    lambda_prime_3 = sqrt(float(lambda_3)**2 + r_p**2)
    lambda_prime_4 = sqrt(float(lambda_4)**2 + r_p**2)

    # Clamp pour éviter les erreurs de domaine de asin
    def safe_asin(x):
        return math.asin(min(1.0, max(-1.0, x)))

    # beta_i : angle angle entre le sommet de la poulie i et le point tangent entre la poulie i et le câble i
    arg_beta_1 = (Y_e - h_1 - L/2 + r_p) / lambda_prime_1
    arg_beta_2 = (h_2 - Y_e - L/2 + r_p) / lambda_prime_2
    arg_beta_3 = (Y_e - h_1 - L/2 + r_p) / lambda_prime_3
    arg_beta_4 = (h_2 - Y_e - L/2 + r_p) / lambda_prime_4

    beta_1 = safe_asin(arg_beta_1)
    beta_2 = safe_asin(arg_beta_2)
    beta_3 = safe_asin(arg_beta_3)
    beta_4 = safe_asin(arg_beta_4)

    # lambda_sec_i : longueur de câble directement enroulé sur la poulie i
    lambda_sec_1 = (beta_1 + math.pi / 2) * r_p
    lambda_sec_2 = (beta_2 + math.pi / 2) * r_p
    lambda_sec_3 = (beta_3 + math.pi / 2) * r_p
    lambda_sec_4 = (beta_4 + math.pi / 2) * r_p

    # lambda_tot_i : longueur de câble total depuis le tambour jusqu'au point d'attache sur l'effecteur
    lambda_tot_1 = lambda_prime_1 + lambda_sec_1 + lambda_troisieme
    lambda_tot_2 = lambda_prime_2 + lambda_sec_2 + lambda_troisieme
    lambda_tot_3 = lambda_prime_3 + lambda_sec_3 + lambda_troisieme
    lambda_tot_4 = lambda_prime_4 + lambda_sec_4 + lambda_troisieme

    return lambda_tot_1, lambda_tot_2, lambda_tot_3, lambda_tot_4


# Fonction principale de simulation
def animation_2(X_inital, Y_initial, phi_1_initial, X_final, Y_final, phi_1_final, V_min, step, nb_points, epsilon):
    """
    Parameters
    ----------
    X_intial : {float} Coordonnées sur l'axe x finale du centre de l'effecteur initialement
    Y_initial : {float} Coordonnées sur l'axe y finale du centre de l'effecteur initialement
    phi_1_initial : {float} Angle de rotation finale autour de l'axe z du centre l'effecteur initialement
    
    X_final : {float} Coordonnées sur l'axe x finale du centre de l'effecteur
    Y_final : {float} Coordonnées sur l'axe y finale du centre de l'effecteur 
    phi_1_final : {float} Angle de rotation finale autour de l'axe z du centre l'effecteur
    
    V_min : {float} Vitesse minimale de l'effecteur
    step : {float} Taille des pas de la simulation
    nb_points : {int} Nombre de points pour effectuer la simulation (en seconde)
    epsilon : {int} Valeur en % de la bande d'arrêt

    Returns
    -------
    Cette fonction fait la simulation du déplacement de l'effecteur du robot parallèle à câble de sa postion initale (le centre du repère)
    à une position final de coordonée X_final, Y_final et avec un angle phi_1_final.
    Elle affiche différent plot:
        - le 1er est l'animation liée à cette simulation
        - Le 2nd est constitué de 4 subplot qui chacun affichent la longueurs des 4 câbles en fonction des itérations
        - Le 3ème affiche la position angulaire des moteurs
        - le 4ème affiche les pas des moteurs
        - le 5ème affiche la position du centre de l'effecteur 
        - le 6ème affiche la rotation du centre de l'effecteur
        - Le 7ème affiche les vitesses linéaires des câbles 
        - Le 8ème affiche les vitesse de rotation des moteur
        - Le 9ème affiche le longueurs totales des câbles
        
    Cette fonction créé aussi un document xls avec les données de la simulation
        """
    # --------------- Initialisation des paramètres --------------------------#
    
    # Initialisation à la position initiale
    X0, Y0 , phi_1_0 = X_inital, Y_initial, phi_1_initial
    print("\nPosition initiale: X0 = ", X0, "Y0 = ", Y0, "phi_1_0 = ", phi_1_0)
    X_traj, Y_traj, phi_traj = [X0], [Y0], [phi_1_0]
    
    # Initialisation des longueurs de câbles
    l_0 = cable_lengths(X0, Y0, 0.0)
    l_traj = [l_0]
    print("Longueurs des câbles initiales: ",phi_1_0)
    
    X, Y, phi_1 = X0, Y0, phi_1_0
   
    # Initialisation des vitesses
    v_traj = []      
    

    # ----------------- Boucle de simulation ---------------------------------#
    
    for i in range(nb_points):
        dx = X_final - X 
        dy = Y_final - Y
        dphi = phi_1_final - phi_1
        error = np.array([dx, dy, dphi])
        error_norm = np.linalg.norm(error)
    
        # Pour éviter de diviser par 0
        if error_norm < 1e-4:
            break
        
        # Direction normalisée de l'erreur
        direction = error / error_norm 
        
        # Vitesse constante (ou minimale)
        vitesse_constante = V_min  # mm/itération
        
        # Déplacement souhaité avec vitesse fixe
        target_move = direction * vitesse_constante
        
        # Calcul de la Jacobienne
        J = jacobian(X, Y, phi_1)  # Jacobienne (d_rond L/ d_rond X); X = [x, y, phi_1) à l’instant courant 
        
        # Calcul de la pseudo inverse de la Jacobienne pour estimer variation de la position de la plaque
        J_pseudo_inv = pinv(J.T @ J) @ J.T  # Pseudo-inverse de la Jacobienne
        dl = J @ target_move  # Variation attendue des longueurs des câbles
        delta_q = J_pseudo_inv @ dl  # Variation de position à partir de la longueur des câbles attendues
        

        # Mise à jour de la position
        X += delta_q[0]
        Y += delta_q[1]
        phi_1 += delta_q[2]
        print("\nPosition : X = ", X, "Y = ", Y, "phi_1 = ", phi_1)
        
        # Stockage des nouvelles valeures
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

        # Arrêt si on a atteint la position finale à avec une marge de [Valeur finale * epsilon/100; Valeur finale * epsilon/100]
        tol = epsilon / 100 
        abs_tol_x = tol * max(1.0, abs(X_final))
        abs_tol_y = tol * max(1.0, abs(Y_final))
        abs_tol_phi = tol * max(1.0, abs(phi_1_final))
        
        if abs(X - X_final) <= abs_tol_x and \
           abs(Y - Y_final) <= abs_tol_y and \
           abs(phi_1 - phi_1_final) <= abs_tol_phi:
            print("\nArrêt à l'itération:", i, "\n")
            break
        
        
    # ----------------- Animation graphique ----------------------------------#
    
    def anim():
        fig, ax = plt.subplots()
        ax.set_xlim(-200, l_1 + 200)
        ax.set_ylim(-200, h_2 + 200)
        ax.set_aspect('equal')
        plate, = ax.plot([], [], 'b-', lw=2)
        cables, = ax.plot([], [], 'k--', lw=1)
        center, = ax.plot([], [], 'ro')
        traj_line, = ax.plot([], [], 'g-', linewidth=3, label="Trajectoire")

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
            traj_line.set_data(X_traj[:frame+1], Y_traj[:frame+1])  

            return plate, cables, center, traj_line  
        
        ani = FuncAnimation(fig, update, frames=len(X_traj), interval=50, blit=True)
        return ani

    global ani
    ani = anim()
    plt.title("Simulation du Robot Parallèle à Câbles")

    

    # ------------------------- Tracés ---------------------------------------#

    # Tracé des longueurs de câble
    fig_var, axs_var = plt.subplots(nrows=2, ncols=2)
    fig_var.suptitle("Longueurs des câbles")
    l_traj_vec = np.array(l_traj)
    D1, D2, D3, D4 = l_traj_vec[:,0], l_traj_vec[:,1], l_traj_vec[:,2], l_traj_vec[:,3]
    for ax, D, title, color in zip(axs_var.flat, [D4, D2, D3, D1], ["Câble 4", "Câble 2", "Câble 3", "Câble 1"], ["red", "green", "blue", "purple"]):
        Etape = np.linspace(0, nb_points, np.shape(D1)[0])
        ax.plot(Etape, D, marker='o', color=color)
        ax.set_title(title)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Longueur de câble [mm]")
        ax.grid()
        
        
    # Tracé des positions des moteurs
    fig_var, axs_var = plt.subplots(nrows=2, ncols=2)
    fig_var.suptitle("Positions angulaires des moteurs")
    q_traj_vec = np.array(l_traj)
    Q1, Q2, Q3, Q4 = q_traj_vec[:,0]/r, q_traj_vec[:,1]/r, q_traj_vec[:,2]/r, q_traj_vec[:,3]/r
    for ax, Q, title, color in zip(axs_var.flat, [Q4, Q2, Q3, Q1], ["Moteur 4", "Moteur 2", "Moteur 3", "Moteur 1"], ["red", "green", "blue", "purple"]):
        Etape = np.linspace(0, nb_points, np.shape(D1)[0])
        ax.plot(Etape, Q, marker='o', color=color)
        ax.set_title(title)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Position du moteur [radians]")
        ax.grid()
    
    
    # Tracé des pas
    fig_var, axs_var = plt.subplots(nrows=2, ncols=2)
    fig_var.suptitle("Pas des moteurs")
    l_traj_vec = np.array(l_traj)
    P1, P2, P3, P4 = pas_mot * l_traj_vec[:,0], pas_mot * l_traj_vec[:,1], pas_mot * l_traj_vec[:,2], pas_mot * l_traj_vec[:,3]
    for ax, P, title, color in zip(axs_var.flat, [P4, P2, P3, P1], ["Moteur 4", "Moteur 2", "Moteur 3", "Moteur 1"], ["red", "green", "blue", "purple"]):
        Etape = np.linspace(0, nb_points, np.shape(D1)[0])
        ax.plot(Etape, P, marker='o', color=color)
        ax.set_title(title)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Pas du moteur")
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
    
    
    # Tracé des longueurs totales de câble
    fig_var, axs_var = plt.subplots(nrows=2, ncols=2)
    fig_var.suptitle("Longueurs totales des câbles")
    l_traj_vec = np.array(l_traj)
    D1, D2, D3, D4 = l_traj_vec[:,0], l_traj_vec[:,1], l_traj_vec[:,2], l_traj_vec[:,3]
    
    lambda_tot_1, lambda_tot_2, lambda_tot_3, lambda_tot_4  = [], [], [], []
    for l_tot_1, l_tot_2, l_tot_3, l_tot_4, Ye in zip(D1, D2, D3, D4, Y_traj):
        l1, l2, l3, l4 = rectification_lambda(l_tot_1, l_tot_2, l_tot_3, l_tot_4, Ye)
        lambda_tot_1.append(l1)
        lambda_tot_2.append(l2)
        lambda_tot_3.append(l3)
        lambda_tot_4.append(l4)

    for ax, D, title, color in zip(axs_var.flat, [lambda_tot_4, lambda_tot_2, lambda_tot_3, lambda_tot_1], ["Câble 4", "Câble 2", "Câble 3", "Câble 1"], ["red", "green", "blue", "purple"]):
        Etape = np.linspace(0, nb_points, np.shape(D1)[0])
        ax.plot(Etape, D, marker='o', color=color)
        ax.set_title(title)
        ax.set_xlabel("Itération")
        ax.set_ylabel("Longueur totale de câble [mm]")
        ax.grid()
    
    # Affichage de tous les graphes
    plt.show()
    
    
    
    #----------- Création du fichier xls avec les données des test -----------#
    
    # Listes avec les longueurs des câbles
    longueur_cable_1 = ["Longueurs câble 1"] + list(D1)
    longueur_cable_2 = ["Longueurs câble 2"] + list(D2)
    longueur_cable_3 = ["Longueurs câble 3"] + list(D3)
    longueur_cable_4 = ["Longueurs câble 4"] + list(D4)
    
    # Listes avec les longueurs totales des câbles
    longueur_totale_cable_1 = ["Longueurs totales câble 1"] + list(lambda_tot_1)
    longueur_totale_cable_2 = ["Longueurs totales câble 2"] + list(lambda_tot_1)
    longueur_totale_cable_3 = ["Longueurs totales câble 3"] + list(lambda_tot_1)
    longueur_totale_cable_4 = ["Longueurs totales câble 4"] + list(lambda_tot_1)

    # Listes avec les vitesses des câbles
    vitesse_cable_1 = ["Vitesses câble 1"] + list(V1)
    vitesse_cable_2 = ["Vitesses câble 2"] + list(V2)
    vitesse_cable_3 = ["Vitesses câble 3"] + list(V3)
    vitesse_cable_4 = ["Vitesses câble 4"] + list(V4)

    # Listes avec les coordonées du centre de l'effecteur
    trajectoire_x = ["Coordonées du centre de l'effecteur sur l'axe x"] + list(X_traj)
    trajectoire_y = ["Coordonées du centre de l'effecteur sur l'axe y"] + list(Y_traj)


    # Regroupe-les dans une liste (ordre d’écriture)
    arrays = [longueur_cable_1, longueur_cable_2, longueur_cable_3, longueur_cable_4,
              longueur_totale_cable_1, longueur_totale_cable_2, longueur_totale_cable_3, longueur_totale_cable_4,
              vitesse_cable_1, vitesse_cable_2, vitesse_cable_3, vitesse_cable_4,
              trajectoire_x, trajectoire_y]

    # Nom de la feuille et du fichier
    sheet_name = "Données de test numérique"  # Nom de la feuille
    filename = "Data_test.xlsx" # Nom du fichier xls

    # Paramètre : écriture verticale ou horizontale
    ecriture_verticale = False  # Mettre False pour les mettre côte à côte

    # Création du writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        start_row, start_col = 0, 0

        for array in arrays:
            df = pd.DataFrame(array)
            # Écrit le DataFrame dans la feuille, à la position voulue
            df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=start_col, index=False, header=False)

            # Mise à jour de la position de départ pour le prochain array
            if ecriture_verticale:
                start_row += df.shape[0] + 1  # Ajoute une ligne vide entre les blocs
            else:
                start_col += df.shape[1] + 1  # Ajoute une colonne vide entre les blocs

    print(f"\n\nLes tableaux ont été écrits dans la feuille '{sheet_name}' du fichier '{filename}'.")
    
    
