## importations

import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import numpy as np
from sympy import *
from mpl_toolkits.mplot3d import Axes3D

## déclaration des variables
mu = [0 ,0]
Sigma = [[0.25 ,0.3] ,[0.3 ,1.0]]
p = [0.1,0.5,0.8, 0.9]
x=generatePoints(Sigma,mu,p)

## utilitaires

def diagonaliser_matrice_numpy(matrice):
    # Calcul des valeurs propres et des vecteurs propres
    valeurs_propres, vecteurs_propres = np.linalg.eig(matrice)

    # Tri des valeurs propres et des vecteurs propres
    indices_tri = np.argsort(valeurs_propres)
    valeurs_propres_triees = valeurs_propres[indices_tri]
    vecteurs_propres_tries = vecteurs_propres[:, indices_tri]

    # Renvoie des valeurs propres, des vecteurs propres et de la matrice de passage

    return valeurs_propres_triees, vecteurs_propres_tries

def detMat(Matrice):
    return (Matrice[0][0]*Matrice[1][1]-Matrice[0][1]*Matrice[1][0])

def generatePoints(matriceCov,mu,p):
    x = stats.multivariate_normal.rvs(mu,Sigma ,1000)
    return x


## premier apercu

def calculF(matriceCov, Mu):

    x=Symbol("x")
    y=Symbol("y")

    Matrice1 = np.array([x - Mu[0], y - Mu[1]])

    Matrice2 = np.linalg.inv(matriceCov)

    Matrice3 = np.array([[x - Mu[0]], [y - Mu[1]]])

    Matrice12 = [Matrice1[0]*Matrice2[0][0]+Matrice1[1]*Matrice2[1][0], Matrice1[0]*Matrice2[0][1]+Matrice1[1]*Matrice2[1][1]]

    Matrice123 = Matrice12[0]*Matrice3[0][0]+Matrice12[1]*Matrice3[1][0]

    ContenuExp = -0.5 * Matrice123

    fonction = 1 / (sqrt((2 * np.pi)* (2 * np.pi) * detMat(matriceCov))) * exp(ContenuExp)

    return fonction

def drawfX(fonction, mu, taille):
    # Convertir l'expression en une fonction utilisable
    x, y = symbols('x y')
    f = lambdify((x, y), fonction, 'numpy')

    # Créer une grille de points (x, y)
    x_vals = np.linspace(mu[0] - taille, mu[0] + taille, 100)
    y_vals = np.linspace(mu[1] - taille, mu[1] + taille, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Calculer les valeurs de fX(x, y) pour chaque point de la grille
    Z = f(X, Y)

    # Créer une figure en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Tracer la surface en 3D
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Ajouter des étiquettes et un titre
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('fX(x, y)')
    ax.set_title('Courbe de densité de fX(x, y)')

    # Afficher le graphe en 3D
    plt.show()

## Pour les points et l'ellipse
def drawPoints(matriceCov, mu, p,x): # K une liste de K

    plt.scatter(x[:, 0], x[:, 1], s=5, edgecolors="black")
    # On s'attaque à l'ellipse
    ValPropres,VectPropres=diagonaliser_matrice_numpy(matriceCov)

    for element in p :

        k = (1-element)/(2*np.pi*math.sqrt(ValPropres[0] * ValPropres[1]))

        print(k)

        if(k>0):
            a= math.sqrt(ValPropres[0]*-2*math.log(k*2*np.pi*math.sqrt(ValPropres[0]*ValPropres[1])))
            b= math.sqrt(ValPropres[1]*-2*math.log(k*2*np.pi*math.sqrt(ValPropres[0]*ValPropres[1])))
            theta= math.atan(VectPropres[1][0]/VectPropres[1][1])
        else:
            print(k + 'n est pas valide')


        t = np.linspace(0, 2*np.pi, 100)
        ellipse_x = mu[0] + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
        ellipse_y = mu[1] + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

        plt.plot(ellipse_x, ellipse_y)

    plt.xlabel('Axe x')
    plt.ylabel('Axe y')
    plt.title('Ellipse')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calculatePointsInEllipses(Sigma, mu, p,x):
    matriceCov = np.array(Sigma)  # Convertir Sigma en matrice numpy

    points_in_ellipses = []

    for element in p:
        k = (1 - element) / (2 * np.pi * math.sqrt(matriceCov[0, 0] * matriceCov[1, 1]))

        if k > 0:
            a = math.sqrt(matriceCov[0, 0] * -2 * math.log(k * 2 * np.pi * math.sqrt(matriceCov[0, 0] * matriceCov[1, 1])))
            b = math.sqrt(matriceCov[1, 1] * -2 * math.log(k * 2 * np.pi * math.sqrt(matriceCov[0, 0] * matriceCov[1, 1])))
            theta = math.atan(matriceCov[1, 0] / matriceCov[1, 1])

            points_inside_ellipse = 0

            for point in x:
                x_diff = point[0] - mu[0]
                y_diff = point[1] - mu[1]
                rotated_x = x_diff * np.cos(theta) + y_diff * np.sin(theta)
                rotated_y = -x_diff * np.sin(theta) + y_diff * np.cos(theta)
                ellipse_term = (rotated_x / a) ** 2 + (rotated_y / b) ** 2

                if ellipse_term <= 1:
                    points_inside_ellipse += 1

            points_in_ellipses.append(points_inside_ellipse)

        else:
            print(str(k) + " n'est pas valide.")

    for i in range(len(p)):
        print("Nombre de points dans l'ellipse", i+1, ":", points_in_ellipses[i])

    return points_in_ellipses

## Pour l'estimateur de mu

def drawMeansSolo(matriceCov, mu, p,x):
    plt.scatter(x[:, 0], x[:, 1], s=5, edgecolors="black")

    # Diviser les points en paquets
    num_points = [10, 50, 100]
    colors = ['red', 'green', 'blue']
    markers = ['x', '+', 'o']
    labels = ['Paquet de 10 points', 'Paquet de 50 points', 'Paquet de 100 points']

    for i, num in enumerate(num_points):
        num_paquets = 1000 // num
        paquets = np.split(x[:num_paquets * num], num_paquets)

        moyennes_x = np.mean(np.array(paquets)[:, :, 0], axis=1)
        moyennes_y = np.mean(np.array(paquets)[:, :, 1], axis=1)

        plt.scatter(moyennes_x, moyennes_y, marker=markers[i], color=colors[i], label=labels[i])

        # Calcul des valeurs propres et vecteurs propres de la matrice de covariance
        val_propres, vect_propres = np.linalg.eig(matriceCov)
        order = val_propres.argsort()[::-1]
        val_propres = val_propres[order]
        vect_propres = vect_propres[:, order]

        # Calcul des demi-axes de l'ellipse
        demi_axe_x = np.sqrt(val_propres[0]) * np.sqrt(-2 * np.log(1 - p[i]))
        demi_axe_y = np.sqrt(val_propres[1]) * np.sqrt(-2 * np.log(1 - p[i]))

        # Calcul de l'angle d'inclinaison de l'ellipse
        theta = np.degrees(np.arctan2(vect_propres[1, 0], vect_propres[0, 0]))

        # Tracer l'ellipse d'isodensité pour le groupe de moyennes
        t = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = mu[0] + demi_axe_x * np.cos(np.radians(theta)) * np.cos(t) - demi_axe_y * np.sin(np.radians(theta)) * np.sin(t)
        ellipse_y = mu[1] + demi_axe_x * np.sin(np.radians(theta)) * np.cos(t) + demi_axe_y * np.cos(np.radians(theta)) * np.sin(t)

        plt.plot(ellipse_x, ellipse_y, color=colors[i])

    plt.scatter(mu[0], mu[1], color='pink', label='Coordonnées de mu')

    plt.xlabel('Axe x')
    plt.ylabel('Axe y')
    plt.title('Moyennes des paquets de points')
    plt.grid(True)
    plt.xlim(-5.5, 5.5)  # Définir la limite de l'axe x
    plt.ylim(-5.5, 5.5)
    plt.axis('equal')
    plt.legend(loc='best')
    plt.show()


def drawMeansByThree(matriceCov, mu, p,x):

    x=generatePoints(matriceCov,mu,p)

    num_points = [10, 50, 100]
    colors = ['red', 'green', 'blue']
    markers = ['x', '+', 'o']
    labels = ['Paquet de 10 points', 'Paquet de 50 points', 'Paquet de 100 points']

    fig, axs = plt.subplots(1, len(num_points), figsize=(12, 4))

    for i, num in enumerate(num_points):
        num_paquets = 1000 // num
        paquets = np.split(x[:num_paquets * num], num_paquets)

        moyennes_x = np.mean(np.array(paquets)[:, :, 0], axis=1)
        moyennes_y = np.mean(np.array(paquets)[:, :, 1], axis=1)

        axs[i].scatter(moyennes_x, moyennes_y, marker=markers[i], color=colors[i], label=labels[i])
        axs[i].set_xlim(-5.5, 5.5)
        axs[i].set_ylim(-5.5, 5.5)
        axs[i].set_title('Moyennes - ' + labels[i])

        val_propres, vect_propres = np.linalg.eig(matriceCov)
        order = val_propres.argsort()[::-1]
        val_propres = val_propres[order]
        vect_propres = vect_propres[:, order]

        demi_axe_x = np.sqrt(val_propres[0]) * np.sqrt(-2 * np.log(1 - p[i]))
        demi_axe_y = np.sqrt(val_propres[1]) * np.sqrt(-2 * np.log(1 - p[i]))

        theta = np.degrees(np.arctan2(vect_propres[1, 0], vect_propres[0, 0]))

    fig.suptitle('Moyennes des paquets de points')
    plt.tight_layout()
    plt.show()

## pour l'estimateur de sigma

import numpy as np
import matplotlib.pyplot as plt

def drawCovariance(matriceCov, mu, p, x):
    num_points = [10, 50, 100]
    colors = ['red', 'green', 'blue']
    markers = ['x', '+', 'o']
    labels = ['Paquet de 10 points', 'Paquet de 50 points', 'Paquet de 100 points']

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, num in enumerate(num_points):
        num_paquets = 1000 // num
        paquets = np.split(x[:num_paquets * num], num_paquets)

        covariances = np.zeros((num_paquets, 2, 2))
        for j, paquet in enumerate(paquets):
            covariances[j] = np.cov(paquet.T)

        moyennes_x = np.mean(np.array(paquets)[:, :, 0], axis=1)
        moyennes_y = np.mean(np.array(paquets)[:, :, 1], axis=1)

        val_propres, vect_propres = np.linalg.eig(matriceCov)
        order = val_propres.argsort()[::-1]
        val_propres = val_propres[order]
        vect_propres = vect_propres[:, order]

        demi_axe_x = np.sqrt(val_propres[0]) * np.sqrt(-2 * np.log(1 - p[i]))
        demi_axe_y = np.sqrt(val_propres[1]) * np.sqrt(-2 * np.log(1 - p[i]))

        theta = np.degrees(np.arctan2(vect_propres[1, 0], vect_propres[0, 0]))

        t = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = mu[0] + demi_axe_x * np.cos(np.radians(theta)) * np.cos(t) - demi_axe_y * np.sin(np.radians(theta)) * np.sin(t)
        ellipse_y = mu[1] + demi_axe_x * np.sin(np.radians(theta)) * np.cos(t) + demi_axe_y * np.cos(np.radians(theta)) * np.sin(t)

        ax.plot(ellipse_x, ellipse_y, color=colors[i], label=labels[i])

        for j in range(num_paquets):
            val_propres, vect_propres = np.linalg.eig(covariances[j])
            order = val_propres.argsort()[::-1]
            val_propres = val_propres[order]
            vect_propres = vect_propres[:, order]

            demi_axe_x = np.sqrt(val_propres[0]) * np.sqrt(-2 * np.log(1 - p[i]))
            demi_axe_y = np.sqrt(val_propres[1]) * np.sqrt(-2 * np.log(1 - p[i]))

            theta = np.degrees(np.arctan2(vect_propres[1, 0], vect_propres[0, 0]))

            ellipse_x = moyennes_x[j] + demi_axe_x * np.cos(np.radians(theta)) * np.cos(t) - demi_axe_y * np.sin(np.radians(theta)) * np.sin(t)
            ellipse_y = moyennes_y[j] + demi_axe_x * np.sin(np.radians(theta)) * np.cos(t) + demi_axe_y * np.cos(np.radians(theta)) * np.sin(t)

            ax.plot(ellipse_x, ellipse_y, color='black', alpha=0.2)

        ax.scatter(moyennes_x, moyennes_y, marker='.', color=colors[i], alpha=0.5)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_title('Covariances des paquets de points')
    ax.legend()
    plt.tight_layout()
    plt.show()


def drawCovarianceByThree(matriceCov, mu, p,x):
    num_points = [10, 50, 100]
    colors = ['red', 'green', 'blue']
    markers = ['x', '+', 'o']
    labels = ['Paquet de 10 points', 'Paquet de 50 points', 'Paquet de 100 points']

    fig, axs = plt.subplots(1, len(num_points), figsize=(12, 4))

    for i, num in enumerate(num_points):
        num_paquets = 1000 // num
        paquets = np.split(x[:num_paquets * num], num_paquets)

        covariances = np.zeros((num_paquets, 2, 2))
        for j, paquet in enumerate(paquets):
            covariances[j] = np.cov(paquet.T)

        moyennes_x = np.mean(np.array(paquets)[:, :, 0], axis=1)
        moyennes_y = np.mean(np.array(paquets)[:, :, 1], axis=1)

        val_propres, vect_propres = np.linalg.eig(matriceCov)
        order = val_propres.argsort()[::-1]
        val_propres = val_propres[order]
        vect_propres = vect_propres[:, order]

        demi_axe_x = np.sqrt(val_propres[0]) * np.sqrt(-2 * np.log(1 - p[i]))
        demi_axe_y = np.sqrt(val_propres[1]) * np.sqrt(-2 * np.log(1 - p[i]))

        theta = np.degrees(np.arctan2(vect_propres[1, 0], vect_propres[0, 0]))

        t = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = mu[0] + demi_axe_x * np.cos(np.radians(theta)) * np.cos(t) - demi_axe_y * np.sin(np.radians(theta)) * np.sin(t)
        ellipse_y = mu[1] + demi_axe_x * np.sin(np.radians(theta)) * np.cos(t) + demi_axe_y * np.cos(np.radians(theta)) * np.sin(t)

        axs[i].plot(ellipse_x, ellipse_y, color=colors[i])

        for j in range(num_paquets):
            val_propres, vect_propres = np.linalg.eig(covariances[j])
            order = val_propres.argsort()[::-1]
            val_propres = val_propres[order]
            vect_propres = vect_propres[:, order]

            demi_axe_x = np.sqrt(val_propres[0]) * np.sqrt(-2 * np.log(1 - p[i]))
            demi_axe_y = np.sqrt(val_propres[1]) * np.sqrt(-2 * np.log(1 - p[i]))

            theta = np.degrees(np.arctan2(vect_propres[1, 0], vect_propres[0, 0]))

            ellipse_x = moyennes_x[j] + demi_axe_x * np.cos(np.radians(theta)) * np.cos(t) - demi_axe_y * np.sin(np.radians(theta)) * np.sin(t)
            ellipse_y = moyennes_y[j] + demi_axe_x * np.sin(np.radians(theta)) * np.cos(t) + demi_axe_y * np.cos(np.radians(theta)) * np.sin(t)

            axs[i].plot(ellipse_x, ellipse_y, color='black', alpha=0.2)

        axs[i].scatter(moyennes_x, moyennes_y, marker='.', color=colors[i], alpha=0.5)
        axs[i].set_xlim(-5.5, 5.5)
        axs[i].set_ylim(-5.5, 5.5)
        axs[i].set_title('Covariances - ' + labels[i])

    fig.suptitle('Covariances des paquets de points')
    plt.tight_layout()
    plt.show()
