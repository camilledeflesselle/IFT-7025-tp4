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
from scipy.stats import chi2
from drawTree import hierarchy_pos

# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class DecisionTree: #nom de la class à changer

	def __init__(self, names = None, index_fact = None, conversion_labels = None, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.index_fact = index_fact
		self.names = names
		self.conversion_labels = conversion_labels

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
			uniqueValues = np.unique(values) # uniques valeurs
			if len(uniqueValues) == 1:
				potentialSplits[column] = uniqueValues
			else:
				potentialSplits[column] = []
				for i in range(len(uniqueValues)):
					if i != 0:
						currentValue = uniqueValues[i]
						previousValue = uniqueValues[i - 1]
						potentialSplits[column].append((currentValue + previousValue) / 2) # on prend la médiane comme potentielle coupe
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
			max_depth = 1000
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
					question = "{} == {}".format(self.names[racine], value)
					index = np.where(train[:, racine]==value)[0]
					dataB = train[index, :]  # classes des individus dont la valeur de l'attribut est i
					labelsB = train_labels[index]
					yesAnswer = self.train(dataB, labelsB, current_depth, max_depth)
					decisionSubTree[question] = yesAnswer
				return decisionSubTree

			else : # si l'attribut est numérique, l'arbre de décision est fait sur un intervalle
				#value = np.mean(train[:, racine])
				questionInf = "{} <= {}".format(self.names[racine], best_split_value)
				questionSup = "{} > {}".format(self.names[racine], best_split_value)
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
		if len(question.split(sep)) > 1: 
			attribute, value = question.split(sep)
		else : 
			return None
		index = [i==attribute for i in self.names]
		attribute = np.where(index)[0]
		return [int(attribute), float(value)]

	def predict(self, x, decision_tree):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		if not isinstance(decision_tree, dict):
			return decision_tree
		
		questions = list(decision_tree.keys())
		list_map = np.array([self.separate_att(question) for question in questions])
		if self.index_fact !=None and list_map.all() !=None : 
			index = np.where(list_map[:, 1]==float(x[int(list_map[0, 0])]))
			#print(x)
			#print(index)
			index = index[0][0]
			answer = decision_tree[questions[index]]
		else :
			question = questions[0]
			attribute, value = question.split(" <= ")
			index = [i==attribute for i in self.names]
			attribute = np.where(index)[0]
			if x[int(attribute)] <= float(value):
				answer = decision_tree[questions[0]]
			else:
				answer = decision_tree[questions[1]]
		return self.predict(x, answer)
	
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

	def pruningLeaves(self, decision_tree, data, labels, pruned = False, alpha = 0.05):
		"""
		fait l'élagage des noeuds, prend en argument :
		- decision_tree : un arbre de décision
		- data : les données 
		- labels : les classes des données
		- alpha : le seuil pour le test chi2
		
		retourne :
		- un arbre élagué 1 fois en se basant sur chi square
		"""
		isLeaf = True
		do_change = False
		for key in list(decision_tree.keys()):
			if isinstance(decision_tree[key], dict):
				isLeaf = False
				parent = key
		
				sep = ' <= '
				question = parent.split(sep)
				if len(question)<2 : 
					sep = " > "
					question = parent.split(sep)
				if len(question)<2 : 
					sep = " == "
					question = parent.split(' == ')
				attribute, value = question[0], question[1]
				attribute = np.where([i == attribute for i in self.names])[0][0]
				indexCorrect = eval("np.where(data[:, attribute]" + sep + value +")")[0]
				labels = labels[indexCorrect]
				data = data[indexCorrect,:]
				_, do_change, pruned = self.pruningLeaves(decision_tree[parent], data, labels, pruned)
				if do_change :
					print("remplacement")
					decision_tree[parent] = self.classifyData(labels) # on élague et on rempace le noeud par un noeud feuille

		
		if isLeaf: # si deux feuilles
			# calcul de delta à comparer à chi2 pour alpha
			attribute, value = list(decision_tree.keys())[0].split(' <= ')
			attributeIndex = np.where([i == attribute for i in self.names])[0][0]
			value = float(value)

			index_child_left = np.where(data[:, attributeIndex] <= value)[0]
			index_child_right = np.where(data[:, attributeIndex] > value)[0]
			labels_child_left = labels[index_child_left]
			labels_child_right= labels[index_child_right]

			uniques_root, nb_uniques_root = np.unique(labels, return_counts = True)
			nb_uniques_child_left = np.array([np.sum(labels_child_left==i) for i in uniques_root])
			nb_uniques_child_right = np.array([np.sum(labels_child_right==i) for i in uniques_root])

			p_left = np.sum(nb_uniques_child_left)/np.sum(nb_uniques_root)
			p_right= np.sum(nb_uniques_child_right)/np.sum(nb_uniques_root)

			nb_expected_left = p_left * nb_uniques_root
			nb_expected_right = p_right * nb_uniques_root 

			# calcul de delta
			delta = np.sum(np.divide((nb_expected_left-nb_uniques_child_left)**2 , nb_expected_left) + np.divide((nb_expected_right-nb_uniques_child_right)**2 , nb_expected_right) )
			print(attribute + " <= " + str(value))
			print("delta", delta)
			dof = len(labels) - 1
			print("chi2", chi2.isf(q=alpha, df=dof))
			if delta > chi2.isf(q=alpha, df=dof): # on fait l'élagage, on rejette l'hypothèse
				do_change = True
				pruned = True
				decision_tree = "pruned"
				print("élagage")

		return decision_tree, do_change, pruned

	def pruningTree(self, decision_tree, data, labels, alpha = 0.05):
		"""takes decision tree as parameter and returns a pruned tree based on chi square
			params:
				obj (dict):
				obj is a decision tree encoded in the form of decision tree
			return:
				obj (dict):
				obj is decision tree with pruned leaves
		"""
		if isinstance(decision_tree, dict):
			pruned = True
			new_decision_tree = decision_tree
			while pruned and new_decision_tree != "pruned":
				print("itération")
				#continue l'élagage tant qu'il est possible ou jusqu'à ce que l'abre soit entièrement élagué 
				pruned = False
				new_decision_tree, _, pruned = self.pruningLeaves(decision_tree, data, labels, pruned)
				if new_decision_tree != "pruned" :
					decision_tree = new_decision_tree
		return decision_tree

	def build_learning_curve(self, train, train_labels, seed, do_pruning = False):
		size = []
		acc = []
		np.random.seed(seed) 
		indices = np.arange(100)
		np.random.shuffle(indices)
		train = train[indices]
		train_labels = train_labels[indices]
		for nb_instances in range(1, 100):
			train_used = train[0:nb_instances,:]
			labels_train = train_labels[0:nb_instances]
			test_used = train[nb_instances:100,:]
			labels_test = train_labels[nb_instances:100]
			decision_tree = self.train(train_used, labels_train)
			if do_pruning :
				decision_tree = self.pruningTree(decision_tree, train_used, labels_train)
			y_pred = np.array([self.predict(x, decision_tree) for x in test_used])
			correct_pred = y_pred == labels_test
			accuracy = np.mean(correct_pred)
			size.append(nb_instances)
			acc.append(accuracy)
		return acc, size

	def show_learning_curve(self, list_acc, list_size, dataset)	:
		size = np.mean(list_size, axis = 0)
		acc = np.mean(list_acc, axis = 0)
		print("dernière exactitude", acc[-1])
		plt.figure()
		plt.plot(size, acc)
		plt.xlabel("Taille du jeu d'entraînement")
		plt.ylabel("Exactitude sur les données test")

		plt.savefig("learning_curve_pruned_{}.png".format(dataset))
		#plt.show()

	def extractEdgesFromTree(self, decision_tree, edges = [], edge_labels = {}, oldAttribute = None, olDvalue = None, sep = " <= ", index=0):
		index +=1
		if isinstance(decision_tree, dict):
			questions = list(decision_tree.keys())
			currentAttribute, value = questions[0].split(" <= ")
			currentAttribute = questions[0] + str(index)
			if oldAttribute != None : 
				edge = (oldAttribute, currentAttribute)
				edges.append(edge)
				edge_labels[edge] = sep + str(round(float(olDvalue), 2))
			oldAttribute = currentAttribute
			edges, edge_labels, value, index = self.extractEdgesFromTree(decision_tree[questions[0]], edges, edge_labels, oldAttribute, value, " <= ", index)
			edges, edge_labels, value, index = self.extractEdgesFromTree(decision_tree[questions[1]], edges, edge_labels, oldAttribute, value, " > ", index)
		else :
			edge = (oldAttribute, str(decision_tree) + " + " + str(index))
			currentAttribute, value = oldAttribute.split(" <= ")
			edges.append(edge)
			edge_labels[edge] = sep + str(round(float(value), 2))
		return edges, edge_labels, value, index
	   

	def drawTree(self, decision_tree, dataset, name="big"):

		plt.figure(figsize = (12, 12))
		edges, edges_labels, _, _ = self.extractEdgesFromTree(decision_tree)
		G = nx.DiGraph()
		G.add_edges_from(edges)

		pos = hierarchy_pos(G, edges[0][0])
		#pos = nx.spring_layout(G)
		if dataset == "iris":
			nx.draw(
				G, pos, edge_color='black', width=2, linewidths=2, font_size = 15,
				node_size=7000, node_color=['green' if node.split(" + ")[0] in ['0', '1', '2'] else 'pink' for node in G.nodes()],
				labels={node: self.conversion_labels[node.split(" + ")[0]] if node.split(" + ")[0] in ['0', '1', '2'] else node.split(" <= ")[0] for node in G.nodes()}
			)
		else :
			nx.draw(
				G, pos, edge_color='black', width=2, linewidths=2,
				node_size=700, node_color='pink',
				labels={node: node.split(" <= ")[0] for node in G.nodes()}
			)
		nx.draw_networkx_edge_labels(
			G, pos,
			edge_labels=edges_labels,
			font_color='red',
			font_size = 15
		)
		plt.axis('off')
		plt.savefig("tree_{}_{}.png".format(name, dataset))
		#plt.show()
