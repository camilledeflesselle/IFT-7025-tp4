"""
Nous définissons une classe pour l'algorithme des k plus proches voisins, avec les méthodes suivantes :
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
import math

# Knn pour le modèle des k plus proches voisins

class Knn:

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		
	def euclidean_distance(row1, row2):
		"""
		calcule la distance euclidienne entre deux instances (ou vecteurs)
		"""
		distance = 0.0
		for i in range(len(row1)-1):
			distance += (row1[i] - row2[i])**2
		return math.sqrt(distance)

	def get_neighbors(train, test_row, num_neighbors):
		distances = list()
		for train_row in train:
			dist = euclidean_distance(test_row, train_row)
			distances.append((train_row, dist))
		distances.sort(key=lambda tup: tup[1])
		neighbors = list()
		for i in range(num_neighbors):
			neighbors.append(distances[i][0])
		return neighbors
		
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
        
	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
        
	
	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.