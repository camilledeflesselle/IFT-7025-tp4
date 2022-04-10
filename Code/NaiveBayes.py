"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
import metrics
import math

# BayesNaif pour le modèle bayesien naif

class BayesNaif:

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		
		
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		"""
		# on sépare le jeu de données train par classe
		self.separateByClass(train, train_labels)
		self.meanAndStd()
        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		self.calculateClassProbabilities(x)
		bestClasse, bestProb = None, -1
		for classe, proba in self.probabilities.items():
			# on sélectionne la classe avec la probailité la plus grande
			if bestClasse is None or proba > bestProb:
				bestProb = proba
				bestClasse = classe
		return bestClasse

	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		"""
		y_pred = np.array([self.predict(x) for x in X])
		metrics.show_metrics(y, y_pred)

	def separateByClass(self, train, train_labels):
		self.separated = {}
		for i in range(len(train)):
			vector = train[i]
			classe = train_labels[i]
			if (classe not in self.separated):
				self.separated[classe] = [] # liste vide
			self.separated[classe].append(vector)

	def meanAndStd(self):
		self.resume = {}
		for classe, vecteurs in self.separated.items():
			# moyenne et écart type de chaque attribut par classe
			self.resume[classe] = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*vecteurs)]

	def gaussProbability(self, x, mean, std):
		"""
		C'est la méthode qui calcule la densité de probabilité d'une distribution normale de x avec
		mean : la moyenne
		std : l'écart-type
		"""
		exponent = math.exp(-((x-mean)**2/(2*std**2)))
		return (1/(std*math.sqrt(2*math.pi))*exponent)

	def calculateClassProbabilities(self, inputVector):
		"""
		calcule la probabilité 
		"""
		self.probabilities = {}
		for classe, classeMeanStd in self.resume.items():
			# initialisation à 1
			self.probabilities[classe] = 1
			# pour chaque attribut
			for i in range(len(classeMeanStd)):
				mean, std = classeMeanStd[i]
				# attribut 
				x = inputVector[i]
				# on utilise la formule de Bayes
				self.probabilities[classe] *= self.gaussProbability(x, mean, std)





