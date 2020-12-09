import pandas as pd
import numpy as np
import scipy as sc
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl

df = pd.read_csv('reg_simple.csv')


# affichage en nuage de points 

x = np.array((df['heure_rev']))
y = np.array((df['note']))
plt.scatter(x, y)

#redimentionnement des matrices
x = x.reshape((x.shape[0], 1))
y = y.reshape((y.shape[0], 1))

# ajout colonne de biais
X = np.hstack((x, np.ones(x.shape)))

# création de theta
theta = np.random.randn(2, 1)

# Création du modèle
def model(X, theta):
    return X.dot(theta)

# Fonction du coût 
def fonction_cout(X,y,theta):
    y1 = model(X, theta)
    m = len(y)
    return 1/(2*m) * np.sum((y1-y)**2)

# Gradient 
def gradient(X,y,theta):
    y1 = model(X, theta)
    m=len(y)
    return 1/m * X.T.dot(y1-y)

# Descente du gradient 
def descente_gradient(X, y, theta, alpha, n_iterations):
    histo_cout = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta -= alpha * gradient(X,y,theta)
        histo_cout[i] = fonction_cout(X,y,theta)
    return theta, histo_cout

#Entrainement

axes = plt.gca()# va permettre de limiter la taille des x et/ou y
alpha = 0.001
n_iterations = 30

final, histo_cout = descente_gradient(X,y,theta,alpha,n_iterations)

predictions = model(X, final)

# plt.subplot(121)
plt.scatter(x,y, c='r')
plt.plot(x, predictions, c='b')
# axes.set_ylim(0, 100)
plt.show()

# plt.subplot(122)
plt.plot(range(n_iterations), histo_cout)
plt.show()


# coefficient de détermination

def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

print(coef_determination(y, predictions))