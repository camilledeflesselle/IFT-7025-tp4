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
import metrics
import matplotlib.pyplot as plt
import operator 
# Knn pour le modèle des k plus proches voisins

class Knn:

	def __init__(self, repeat_kfold = 30, L = 5, k = 3, **kwargs):
		"""
		C'est un Initializer avec
		repeat_kfold : entier servant à la boucle de validation croisée
		L : entier qui permet de diviser les données en L échantillons lors de la validation croisée
		"""
		self.L = L
		self.repeat_kfold = repeat_kfold
		self.k = k

		

	def euclideanDistance(self, row1, row2):
		"""
		calcule la distance euclidienne entre deux instances (ou vecteurs)
		row1 et row2, matrices de taille 1xm, avec 
		m : le nombre d'attribus (le nombre de caractéristiques)
		"""
		distance = 0.0
		for i in range(len(row1)-1):
			distance += (row1[i] - row2[i])**2
		return math.sqrt(distance)

	def getNeighbors(self, train, train_labels, test_row):
		"""
		renvoie les k plus proches voisins de la ligne de test parmi les
		données d'entraînement

		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		test_row est une matrice de type Numpy et de taille 1xm, avec 
		m : le nombre d'attribus (le nombre de caractéristiques)

		k est un entier, le nombre de voisins retenus
		"""
		distances = list()

		for i in range(train.shape[0]): # on calcule la distance entre la donnée test et pour chaque donnée d'entraînement
			dist = self.euclideanDistance(test_row, train[i])
			distances.append((train_labels[i], dist))

		# on ordonne la liste de tuples par distance croissante
		distances.sort(key=lambda tuple: tuple[1])
		neighbors = list()

		# on conserve la liste des k voisins avec les distances minimales
		for i in range(self.k):
			neighbors.append(distances[i][0])
		return neighbors
		
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1

		test est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le nombre d'attribus (le nombre de caractéristiques)
		
		k est un entier correpondant à la valeur de k utilisée
		"""
		self.train_data = train
		self.train_labels = train_labels
        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		neighbors = self.getNeighbors(self.train_data, self.train_labels, x)
		numberClasses = {}
		for neighbor in neighbors :
			if neighbor in numberClasses :
				numberClasses[neighbor]+=1
			else:
				numberClasses[neighbor]=1
		
		sortedClasses = sorted(numberClasses.items(), key=operator.itemgetter(1), reverse=True)
		return sortedClasses[0][0] # classe majoritaire des voisins

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
		self.predictions = np.array([self.predict(x) for x in X])
		metrics.show_metrics(y, self.predictions)

	
	def getBestKppv(self, train, train_labels):
		"""
		C'est la méthode qui va calculer la meilleure valeur de K, en utilisant la validation croisée :
		1) On divise les données originales en L échantillons.
		2) Pour chacune des valeurs de K possibles :
			2-1) Pour chacun des L échantillons de validation :
				a) Nous entraînons le modèle sur L-1 échantillons d'entraînement.
				b) Nous effectuons une prédiction sur l'échantillon de validation restant.
				c) Nous calculons l'exactitude entre les vrais labels et la prédiction.
			2-2) Nous calculons ensuite la moyenne des exactitudes.
		3) Nous choisissons la valeur de K qui maximise la moyenne des exactitudes
		"""
		best_kppv = 0
		max_mean_accuracy = 0
		# 1)
		fold_indices = np.arange(train.shape[0])
		valid_indices = np.array_split(fold_indices, self.L)
		self.mean_accuracies = []
		# 2)
		for k in range(1, self.repeat_kfold+1):
			# initialisation d'un tableau qui contient les exactitudes pour chaque échantillon
			self.k = k
			all_accuracy = []
			# 2-1) boucle de validation croisée
			for i_fold in range(self.L):
				idx = valid_indices[i_fold]
				# séparation train/validation
				valid_data, valid_lab = train[idx], train_labels[idx]
				train_data, train_lab = np.delete(train, idx, axis =0), np.delete(train_labels, idx)
				# a) entraînement du modèle
				self.train(train_data, train_lab)
				# b) prédiction sur les données de validation
				valid_pred = np.array([self.predict(valid_row) for valid_row in valid_data])
				# c) calcul de l'exactitude
				all_accuracy.append(np.sum([valid_lab[i] == valid_pred[i] for i in range(valid_lab.shape[0])])/valid_lab.shape[0])
			# 2-2) moyenne des exactitudes pour un certain k
			mean_accuracy = np.mean(all_accuracy)
			self.mean_accuracies.append(mean_accuracy)
			if max_mean_accuracy < mean_accuracy :
				# nous retenons le maximum des exactitudes
				max_mean_accuracy = mean_accuracy 
				# et le k associé
				best_kppv = k
		print("best_kppv =", best_kppv)
		self.k = best_kppv

	def plotAccuracy(self):
		plt.plot(range(1, self.repeat_kfold +1), self.mean_accuracies)
		plt.axvline(x=self.k,color='red',linestyle='--')
		plt.xlabel('K (nombre de plus proches voisins)')
		plt.ylabel('Moyenne des exactitudes')
		plt.savefig('best_kppv_search.png', bbox_inches='tight')
		#plt.show()