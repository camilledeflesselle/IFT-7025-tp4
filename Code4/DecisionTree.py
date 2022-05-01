import numpy as np
from metrics import show_metrics
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import chi2  # utilisé pour l'élagage
from drawTree import hierarchy_pos

# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class DecisionTree: #nom de la class à changer

	def __init__(self, names = None, index_fact = None, conversion_labels = None, **kwargs):
		"""
		C'est un Initializer. 
		Paramètres: (facultatifs)
			names: noms de chaque attribut utilisés lors de la construction de l'arbre (meilleure compréhension)
			index_fact: liste des index des attributs qui sont des attributs catégoriels
			 -> par exemple [0] pour abalones dataset
			conversion_labels: dictionnaire de conversion pour l'affichage de l'arbre avec les étiquettes initiales
			 -> par exemple: {'0': 'Iris-setosa', '1' : 'Iris-versicolor', '2' : 'Iris-virginica'}
		"""
		self.index_fact = index_fact
		self.names = names
		self.conversion_labels = conversion_labels

	def calculateEntropy(self, vector):
		"""
		Calcule l'entropie d'un vecteur vector.
		"""
		_, uniqueClassesNumber = np.unique(vector, return_counts = True)
		proba = uniqueClassesNumber / uniqueClassesNumber.sum()
		return sum(-proba * np.log2(proba))
		
	def calculateGlobalEntropie(self, vector, labels):
		"""
		Méthode qui calcule l'entropie associée à un attribut.
		Paramètres: 
			vector: vecteur contenant les données d'un attribut
			labels: vecteur de labels
		Retourne:
			l'entropie associée à un attribut 
		"""
		global_entropie = 0
		for i in np.unique(vector):
			index = np.where(vector==i)[0]
			pData = len(index)/len(vector)
			dataB = labels[index]  # classes des individus dont la valeur de l'attribut est i
			global_entropie += pData * self.calculateEntropy(dataB)
		return global_entropie

	def classifyData(self, vector):
		"""
		Méthode qui renvoie la classe majoritaire d'un vecteur.
		Paramètres: 
			vector: vecteur de labels
		Retourne:
			classe majoritaire 
		"""
		uniqueClasses, uniqueClassesCounts = np.unique(vector, return_counts = True)
		return uniqueClasses[uniqueClassesCounts.argmax()]

	def determineBestColumn(self, train, train_labels, potentialSplits):
		"""
		Méthode qui calcule le gain d'information sur chaque attribut pour différentes valeurs de coupes.
		Renvoie l'attribut qui maximise le gain et la valeur de coupe associée.
		Paramètres: 
			train est une matrice de type Numpy et de taille nxm, avec 
				n: le nombre d'exemple d'entrainement dans le dataset
				m: le nombre d'attributs (le nombre de caractéristiques)
			train_labels: est une matrice numpy de taille nx1
			potentialSplits: valeurs de coupes potentielles pour tous les attributs
		Retourne:
			best_column: attribut avec le meilleur gain d'information (int)
			best_split_value: valeur de coupe correspondante (float)
		"""
		entropie_init = self.calculateEntropy(train_labels)
		best_column = None
		best_split_value = None
		max_gain = None
		for i in range(train.shape[1]) :
			for split_value in potentialSplits[i]:
				data = train[:, i]
				data = np.where(data<=split_value, 0, 1)
				gain = entropie_init - self.calculateGlobalEntropie(data, train_labels)
				
				if max_gain == None or max_gain < gain :
					max_gain = gain
					best_column = i
					best_split_value = split_value
		return best_column, best_split_value

	def getPotentialSplits(self, data):
		"""
		Méthode qui retourne les différentes valeurs de coupes par attribut.
		Paramètres: 
			data est une matrice de type Numpy et de taille nxm, avec 
				n: le nombre d'exemple d'entrainement dans le dataset
				m: le nombre d'attributs (le nombre de caractéristiques)
		Retourne:
			potentialSplits: dictionnaire à m clés (attributs) dont les valeurs sont les listes de 
			valeurs de coupes possible d'un attribut
		"""
		potentialSplits = {}
		_, columns = data.shape
		for column in range(columns):
			values = data[:, column]
			uniqueValues = np.unique(values) # uniques valeurs
			# si l'attribut est un attribut catégoriel ou qu'il n'a qu'une unique valeur possible,
			# les valeurs de coupes potentielles sont ses valeurs uniques
			if len(uniqueValues) == 1 or (self.index_fact != None and column in self.index_fact) :
				potentialSplits[column] = uniqueValues
			else:
				# sinon ce sont toutes les médianes entre deux valeurs uniques
				potentialSplits[column] = []
				for i in range(len(uniqueValues)):
					if i != 0:
						currentValue = uniqueValues[i]
						previousValue = uniqueValues[i - 1]
						potentialSplits[column].append((currentValue + previousValue) / 2) # on prend la médiane comme potentielle coupe
		return potentialSplits


	def train(self, train, train_labels, current_depth = 0, max_depth = None): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode récursive qui va entrainer le modèle en créant un arbre de décision.
		Paramètres:
			train est une matrice de type Numpy et de taille nxm, avec 
				n: le nombre d'exemple d'entrainement dans le dataset
				m: le nombre d'attributs (le nombre de caractéristiques)
			train_labels : est une matrice numpy de taille nx1
			current_depth: taille actuelle de l'arbre
			max_depth: la taille maximale de l'arbre
		Retourne:
			l'arbre de décision actuel (dictionnaire)
		"""
		# on nomme les attributs pour améliorer la lisibilité
		if self.names == None :
			self.names = ["Attribut " + str(i) for i in range(train.shape[1])]
		
		# si l'arbre est assez profond ou si une seule classe on classifie, en prenant la classe majoritaire
		if (max_depth != None and current_depth == max_depth ) or len(np.unique(train_labels)) == 1:
			return self.classifyData(train_labels)
		else : # sinon on crée un noeud avec deux branches
			current_depth += 1
			# potentielles valeurs de coupes pour les attributs (continus ou catégoriels)
			potentialSplits = self.getPotentialSplits(train)
			# racine trouvée en calculant les gains et valeur de coupe associée
			racine, best_split_value = self.determineBestColumn(train, train_labels, potentialSplits)
			# initialisation du noeud
			decisionSubTree = {}
			# on traite les attributs comme des attributs continus, l'arbre de décision est fait sur un intervalle
			# 1) branche gauche
			questionInf = "{} <= {}".format(self.names[racine], best_split_value) 
			indexInf = np.where(train[:, racine]<=best_split_value)[0]
			dataInf = train[indexInf, :]  
			labelsInf = train_labels[indexInf]
			decisionSubTree[questionInf] = self.train(dataInf, labelsInf, current_depth, max_depth)
			# 2) branche droite
			questionSup = "{} > {}".format(self.names[racine], best_split_value) 
			indexSup = np.where(train[:, racine]>best_split_value)[0]
			dataSup = train[indexSup, :]  # classes des individus dont la valeur de l'attribut est i
			labelsSup = train_labels[indexSup]
			decisionSubTree[questionSup] = self.train(dataSup, labelsSup, current_depth, max_depth)
			return decisionSubTree

	def predict(self, x, decision_tree):
		"""
		Méthode qui prédit la classe d'un exemple x donné en entrée.
		Paramètres:
			x: exemple qui est de taille 1xm
			decision_tree: arbre de décision (dict)
		Retourne:
			L'étiquette prédite.
		"""
		if not isinstance(decision_tree, dict):
			return decision_tree
		
		questions = list(decision_tree.keys())
		
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
		Méthode qui va évaluer notre modèle sur les données X.
        Paramètres:
		    X: matrice de type Numpy et de taille nxm, avec 
                n: le nombre d'exemple de test dans le dataset
                m: le nombre d'attributs (le nombre de caractéristiques)
		    y: est une matrice numpy de taille nx1
			decision_tree: arbre de décision utilisé
		"""
		y_pred = np.array([self.predict(x, decision_tree) for x in X])
		show_metrics(y, y_pred)

	def pruningLeaves(self, decision_tree, data, labels, alpha = 0.05):
		"""
		Méthode qui fait l'élagage des noeuds.
		Paramètres:
			decision_tree: un arbre de décision (dict)
			data: les données, matrice numpy de taille nxm
		 	labels: les classes des données, matrice numpy de taille nx1
			alpha: le seuil alpha pour le test chi2
		Retourne:
			un arbre élagué 1 fois en se basant sur chi square
		"""
		isLeaf = True
		fusion = False
		pruned = False
		pred_node = []
		for key in list(decision_tree.keys()):
			if isinstance(decision_tree[key], dict):
				isLeaf = False
				parent = key
				sep = ' <= '
				question = parent.split(sep)
				if len(question)<2 : 
					sep = " > "
					question = parent.split(sep)
				attribute, value = question[0], question[1]
				attribute = np.where([i == attribute for i in self.names])[0][0]
				indexCorrect = eval("np.where(data[:, attribute]" + sep + value +")")[0]

				look_parent, pruned, fusion = self.pruningLeaves(decision_tree[parent], data[indexCorrect,:], labels[indexCorrect]) # appel récursif évalué lorsqu'on a deux feuilles
				if pruned or fusion:
					decision_tree[parent] = look_parent 
			else : pred_node.append(decision_tree[key])
		
		if isLeaf: 
			# si deux feuilles
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
			# degré de libertés
			dof = (len(uniques_root) - 1)
			if delta < chi2.isf(q=alpha, df=dof): 
				pruned = True
				return self.classifyData(labels), pruned, False # on fait l'élagage, on rejette l'hypothèse et on garde la classe majoritaire

		# ici permet de réduire la taille de l'abre si deux branches d'un noeud prédisent la même classe
		if len(pred_node) > 1 : 
			if len(np.unique(pred_node)) == 1 :
				fusion = True
				decision_tree = self.classifyData(labels)
			
		return decision_tree, pruned, fusion

	def pruningTree(self, decision_tree, data, labels):
		"""
		Méthode qui fait l'élagage d'un arbre complet par appel à pruningLeaves.
		Paramètres:
			decision_tree: un arbre de décision (dict)
			data: les données, matrice numpy de taille nxm
		 	labels: les classes des données, matrice numpy de taille nx1
		Retourne:
			decision_tree: arbre élagué avec la méthode chi squarred
		"""
		if isinstance(decision_tree, dict):
			pruned = True
			new_decision_tree = decision_tree
			while pruned and new_decision_tree != "pruned":
				#continue l'élagage tant qu'il est possible ou jusqu'à ce que l'abre soit entièrement élagué 
				new_decision_tree, pruned, _ = self.pruningLeaves(decision_tree, data, labels)
				if isinstance(new_decision_tree, float) or isinstance(new_decision_tree, np.int64):
					new_decision_tree = "pruned"
				else : 
					decision_tree = new_decision_tree
		return decision_tree

	def build_learning_curve(self, train, train_labels, seed, do_pruning = False):
		"""
		C'est la méthode qui va permettre de voir si notre modèle apprend correctement.
		Nous entraînons le modèle 99 fois en utilisant un jeu d'entraînement de taille entre 1 et 99 et un 
		jeu de test sur les données restantes.
		Paramètres:
			train est une matrice de type Numpy et de taille 100xm, avec 
				m: le nombre d'attributs (le nombre de caractéristiques)
			train_labels : est une matrice numpy de taille 100x1
			seed: hasard utilisé
			do_pruning: booléen valant True si on souhaite faire l'élagage après l'entraînement
		Retourne:
			acc: liste des exactitudes à chaque itération
			size: liste des tailles du jeu de données d'entraînement utilisées à chaque itérations (nombre d'instances)
		"""
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

	def show_learning_curve(self, list_acc, list_size, dataset, do_pruning = False)	:
		"""
		Méthode qui permet d'enregistrer les courbes d'entraînement calculées avec
		build_learning_curve.
		Paramètres:
			list_acc: la liste des exactitudes entre les labels test et les prédictions sur le jeu de test
			list_size: la liste des tailles du jeu d'entraînement utilisés
			dataset: le nom du dataset (str)
			do_pruning: booléen, qui influe sur le nom du fichier
		"""
		size = np.mean(list_size, axis = 0)
		acc = np.mean(list_acc, axis = 0)
		print("dernière exactitude", acc[-1])
		plt.figure()
		plt.plot(size, acc)
		plt.xlabel("Taille du jeu d'entraînement")
		plt.ylabel("Exactitude sur les données test")

		if do_pruning : plt.savefig("learning_curve_pruned_{}.png".format(dataset))
		else : plt.savefig("learning_curve_{}.png".format(dataset))
		#plt.show()

	def extractEdgesFromTree(self, decision_tree, edges = [], edge_labels = {}, oldAttribute = None, olDvalue = None, sep = " <= ", index=0):
		"""
		Méthode récursive qui permet de réarranger l'abre de décision en ne retenant que les relations, appelée par drawTree.
		Paramètres:
			edges: les relations entre deux noeuds déjà trouvées
			edge_labels: les labels associés aux edges (règles sur une branche)
			oldAttribute: l'attribut du noeud parent
			oldValue: la valeur de coupe de la règle parente
			sep: la comparaison (str) (<= ou >)
			index: index ajouté au nom des attributs qui permet de différencier chaque noeud
		Retourne:
			edges: les nouvelles relations entre deux noeuds
			edge_labels: les nouveaux labels associés aux edges (règles sur une branche)
			value: la valeur de coupe de la règle actuelle
			index: index incrémenté
		"""
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
		"""
		Méthode qui permet de dessiner l'abre de décision
		Paramètres:
			decision_tree: arbre de décision (dict)
			dataset: nom du dataset (str)
			name: str utilisé pour le nom du fichier .png créé (par exemple "elague" ou "big")
		"""
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
				G, pos, edge_color='black', width=2, linewidths=2, font_size = 15,
				node_size=7000, node_color=['green' if node.split(" + ")[0] in ['0', '1', '2'] else 'pink' for node in G.nodes()],
				labels={node: node.split(" + ")[0].split(" <= ")[0] for node in G.nodes()}
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