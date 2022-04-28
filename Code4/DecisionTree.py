"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
from metrics import show_metrics
import matplotlib.pyplot as plt
import networkx as nx
import pylab


# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class DecisionTree: #nom de la class à changer

	def __init__(self, names = None, index_fact = None, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.index_fact = index_fact
		self.names = names

	def calculateEntropy(self, vector):
		"""
		calcule l'entropie d'un vecteur
		"""
		_, uniqueClassesNumber = np.unique(vector, return_counts = True)
		proba = uniqueClassesNumber / uniqueClassesNumber.sum()
		return sum(-proba * np.log2(proba))
		
	def calculateGlobalEntropie(self, vector, labels):
		"""
		calcul le gain d'information pour un attribut
		"""
		global_entropie = 0
		for i in np.unique(vector):
			index = np.where(vector==i)[0]
			pData = len(index)/len(vector)
			dataB = labels[index]  # classes des individus dont la valeur de l'attribut est i
			global_entropie += pData * self.calculateEntropy(dataB)
		return global_entropie

	def classifyData(self, vector):
		uniqueClasses, uniqueClassesCounts = np.unique(vector, return_counts = True)
		return uniqueClasses[uniqueClassesCounts.argmax()]

	def determineBestColumn(self, train, train_labels, potentialSplits):
		entropie_init = self.calculateEntropy(train_labels)
		best_column = None
		best_split_value = None
		max_gain = None
		for i in range(train.shape[1]) :
			for split_value in potentialSplits[i]:
				data = train[:, i]
				data = np.where(data<=split_value, 0, 1)
				gain = entropie_init - self.calculateGlobalEntropie(data, train_labels)
				#print(gain)
				if max_gain == None or max_gain < gain :
					max_gain = gain
					best_column = i
					best_split_value = split_value
		return best_column, best_split_value

	def getPotentialSplits(self, data):
		potentialSplits = {}
		_, columns = data.shape
		for column in range(columns):
			values = data[:, column]
			uniqueValues = np.unique(values)
			if len(uniqueValues) == 1:
				potentialSplits[column] = uniqueValues
			else:
				potentialSplits[column] = []
				for i in range(len(uniqueValues)):
					if i != 0:
						currentValue = uniqueValues[i]
						previousValue = uniqueValues[i - 1]
						potentialSplits[column].append((currentValue + previousValue) / 2)
		return potentialSplits


	def train(self, train, train_labels, current_depth = 0, max_depth = None): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		construction de l'abre de décision de manière récursive
		"""
		if self.names == None :
			self.names = ["Attribut " + str(i) for i in range(train.shape[1])]
		if max_depth == None :
			max_depth = train.shape[1] +1
		if current_depth == max_depth or len(np.unique(train_labels)) == 1: # si l'arbre est assez profond ou si une seule classe on classifie
			return self.classifyData(train_labels)
		else :
			current_depth += 1
			potentialSplits = self.getPotentialSplits(train)
			racine, best_split_value = self.determineBestColumn(train, train_labels, potentialSplits)
			decisionSubTree = {}
			if self.index_fact !=None and racine in self.index_fact : # si l'attribut est de type factoriel 
				uniquesValuesRacine = np.unique(train[:, racine])
				for value in uniquesValuesRacine :
					question = "Attribut {} == {}".format(self.names[racine], value)
					index = np.where(train[:, racine]==value)[0]
					dataB = train[index, :]  # classes des individus dont la valeur de l'attribut est i
					labelsB = train_labels[index]
					yesAnswer = self.train(dataB, labelsB, current_depth, max_depth)
					decisionSubTree[question] = yesAnswer
				return decisionSubTree
			else : # si l'attribut est numérique, l'arbre de décision est fait sur un intervalle
				#value = np.mean(train[:, racine])
				questionInf = "{} <= {}".format(self.names[racine], round(best_split_value, 3))
				questionSup = "{} > {}".format(self.names[racine], round(best_split_value, 3))
				indexInf = np.where(train[:, racine]<=best_split_value)[0]
				dataInf = train[indexInf, :]  # classes des individus dont la valeur de l'attribut est i
				labelsInf = train_labels[indexInf]
				decisionSubTree[questionInf] = self.train(dataInf, labelsInf, current_depth, max_depth)

				indexSup = np.where(train[:, racine]>best_split_value)[0]
				dataSup = train[indexSup, :]  # classes des individus dont la valeur de l'attribut est i
				labelsSup = train_labels[indexSup]
				decisionSubTree[questionSup] = self.train(dataSup, labelsSup, current_depth, max_depth)
				return decisionSubTree
    
	def separate_att(self, question, sep = " == "):
		attribute, value = question.split(sep)
		_, attributeIndex = attribute.split(" ")
		return [int(attributeIndex), float(value)]

	def predict(self, x, decision_tree):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		if not isinstance(decision_tree, dict):
			return decision_tree
		
		questions = list(decision_tree.keys())
		if self.index_fact !=None and list_map[0, 0] in self.index_fact : 
			list_map = np.array([self.separate_att(question) for question in questions])
			index = np.where(list_map[:, 1] == x[list_map[0, 0]])[0][0]
			answer = decision_tree[questions[index]]
		else :
			question = questions[0]
			attribute, value = question.split(" <= ")
			_, attribute = attribute.split(" ")
			
			if x[int(attribute)] <= float(value):
				answer = decision_tree[question]
			else:
				answer = decision_tree[questions[1]]
		return self.predict(x, answer)
	
	def build_learning_curve(self, train, train_labels, test, test_labels, dataset, step):
		size = []
		acc = []
		for nb_instances in range(1, train.shape[0]+1):
			decision_tree = self.train(train[0:nb_instances, :], train_labels[0:nb_instances])
			y_pred = np.array([self.predict(x, decision_tree) for x in test])
			correct_pred = y_pred == test_labels
			accuracy = np.mean(correct_pred)
			size.append(nb_instances)
			acc.append(accuracy)
		plt.figure()
		plt.plot(size, acc)
		plt.xlabel("Training set size")
		plt.ylabel("Proportion correct on test set")

		plt.savefig("learning_curve_{}.png".format(dataset))
		#plt.show()


	def evaluate(self, X, y, decision_tree):
		"""
		c'est la méthode qui va évaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		y : est une matrice numpy de taille nx1
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		y_pred = np.array([self.predict(x, decision_tree) for x in X])
		show_metrics(y, y_pred)


	def drawTree1(self, decision_tree):
		if not isinstance(decision_tree, dict):
			return decision_tree
		questions = list(decision_tree.keys())
		attribute, value = questions[0].split(" <= ")
		attribute = questions[0]
		return (attribute, self.drawTree(decision_tree[questions[0]])), (attribute, self.drawTree(decision_tree[questions[1]]))


	def drawTree2(self, decision_tree, edges = [], edge_labels = {}, oldAttribute = None, sep = " <= "):
		if isinstance(decision_tree, dict):
			questions = list(decision_tree.keys())
			currentAttribute, value = questions[0].split(" <= ")
			currentAttribute = questions[0]
			if oldAttribute != None : 
				edge = (oldAttribute, currentAttribute)
				edges.append(edge)
				edge_labels[edge] = sep + value
			oldAttribute = currentAttribute
			edges, edge_labels = self.drawTree2(decision_tree[questions[0]], edges, edge_labels, oldAttribute, " <= ")
			edges, edge_labels = self.drawTree2(decision_tree[questions[1]], edges, edge_labels, oldAttribute, " > ")
		else :
			edge = (oldAttribute, decision_tree)
			currentAttribute, value = oldAttribute.split(" <= ")
			edges.append(edge)
			edge_labels[edge] = sep + value
		return edges, edge_labels

	def drawTree(self, decision_tree):
		print(decision_tree)
		edges, edges_labels = self.drawTree2(decision_tree)
		print(edges)
		G = nx.DiGraph()
		G.add_edges_from(edges)
		pos = nx.spring_layout(G)
		plt.figure()
		nx.draw(
			G, pos, edge_color='black', width=1, linewidths=1,
			node_size=1000, node_color='pink', alpha=0.9,
			labels={node: node for node in G.nodes()}
		)
		nx.draw_networkx_edge_labels(
			G, pos,
			edge_labels=edges_labels,
			font_color='red'
		)
		plt.axis('off')
		plt.show()

