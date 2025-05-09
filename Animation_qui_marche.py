import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import pinv

# Paramètres du système
L = 1.0  # Largeur
l = 1.0  # Longueur
X0, Y0 = L / 2, l / 2  # Position initiale au centre
dt = 0.05
nb_points = 200
Xf, Yf, phi_f = 0.2, 0.3, 2  # Position finale et angle final

# Positions des poulies (points fixes)
poulies = np.array([
    [0, 0],    # P1
    [L, 0],    # P2
    [L, l],    # P3
    [0, l]     # P4
])

# Vecteurs fixes entre centre et points d'attache
v_attache = np.array([
    [-0.1, -0.1],
    [0.1, -0.1],
    [0.1, 0.1],
    [-0.1, 0.1]
])

def rotation_matrix(phi):
    return np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)]
    ])

def compute_attachment_points(X, Y, phi):
    R = rotation_matrix(phi)
    return np.array([[X, Y]]) + (v_attache @ R.T)

def cable_lengths(X, Y, phi):
    A = compute_attachment_points(X, Y, phi)
    return np.linalg.norm(A - poulies, axis=1)

def jacobian(X, Y, phi):
    A = compute_attachment_points(X, Y, phi)
    R = rotation_matrix(phi)
    J = np.zeros((4, 3))
    for i in range(4):
        diff = A[i] - poulies[i]
        d = np.linalg.norm(diff)
        if d == 0:
            continue
        dX = diff[0] / d
        dY = diff[1] / d
        dphi_vec = R @ np.array([-v_attache[i][1], v_attache[i][0]])
        dphi = np.dot(diff, dphi_vec) / d
        J[i, :] = [dX, dY, dphi]
    return J

# Génération de la trajectoire inverse
X_traj, Y_traj, phi_traj = [X0], [Y0], [0.0]
l_traj = [cable_lengths(X0, Y0, 0.0)]
X, Y, phi = X0, Y0, 0.0
step = 0.05

for _ in range(nb_points):
    dx = Xf - X
    dy = Yf - Y
    dphi = phi_f - phi
    error = np.array([dx, dy, dphi])
    
    # Critère de convergence ajusté
    if np.linalg.norm(error) < 1e-4:
        break

    J = jacobian(X, Y, phi)
    dl = J @ error * step
    
    # Estimation de position par pseudo-inverse
    try:
        J_pinv = pinv(J)
    except np.linalg.LinAlgError:
        print("Erreur : matrice Jacobienne non inversible.")
        break
    
    # Régularisation (ajout d'un petit terme de régularisation pour éviter la singularité)
    lambda_reg = 1e-4
    J_pinv_reg = pinv(J.T @ J + lambda_reg * np.eye(3)) @ J.T
    
    delta_q = J_pinv_reg @ (l_traj[-1] + dl - l_traj[-1])
    
    # Mise à jour de la position
    X += delta_q[0]
    Y += delta_q[1]
    phi += delta_q[2]
    
    X_traj.append(X)
    Y_traj.append(Y)
    phi_traj.append(phi)
    
    l_traj.append(cable_lengths(X, Y, phi))
    
    # Affichage des positions dans la console
    print(f"X = {X:.3f}, Y = {Y:.3f}, phi = {phi:.3f}")
    print(f"Longueurs des câbles : {cable_lengths(X, Y, phi)}")

# Animation
fig, ax = plt.subplots()
ax.set_xlim(-0.2, L + 0.2)
ax.set_ylim(-0.2, l + 0.2)
ax.set_aspect('equal')
plate, = ax.plot([], [], 'o-', lw=2)
cables, = ax.plot([], [], 'k--', lw=1)
center, = ax.plot([], [], 'ro')

def update(frame):
    X, Y, phi = X_traj[frame], Y_traj[frame], phi_traj[frame]
    A = compute_attachment_points(X, Y, phi)
    plate.set_data(A[:, 0].tolist() + [A[0, 0]], A[:, 1].tolist() + [A[0, 1]])
    cable_x = []
    cable_y = []
    for i in [0, 1, 3, 2]:
        cable_x += [poulies[i, 0], A[i, 0], None]
        cable_y += [poulies[i, 1], A[i, 1], None]
    cables.set_data(cable_x, cable_y)
    center.set_data(X, Y)
    return plate, cables, center

ani = FuncAnimation(fig, update, frames=len(X_traj), interval=50, blit=True)
plt.title("Simulation du Robot à Câbles")
plt.show()
