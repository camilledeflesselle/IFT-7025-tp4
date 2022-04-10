import numpy as np
import sys
import load_datasets
import NaiveBayes # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés
import metrics
from sklearn import neighbors # utilisation pour comaparer

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester
"""

# --> Initialisation des paramètres
L = 5              # nombre de sous échantillons pour la validation croisée
repeat_kfold = 30  # K_max
train_ratio = 0.7  # ratio pour la séparation train/test
k = 8              # initialisation de k, utilisée si on ne cherche pas la meilleure valeur de k

# --> Initialisation des classifieurs avec leurs paramètres
classif_knn_iris = Knn.Knn(repeat_kfold=30, L=5, k=8)
classif_knn_wine = Knn.Knn(repeat_kfold=10, L=20, k=1)
classif_knn_abalone = Knn.Knn(repeat_kfold=30, L=10, k=18)

# --> Chargement des datasets (décommenter en fonction du jeu étudié)
# 1) jeu iris
#train, train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)
#classif_knn = classif_knn_iris

# 2) jeu wine
train, train_labels, test, test_labels = load_datasets.load_wine_dataset(train_ratio)
classif_knn = classif_knn_iris

# 3) jeu abalone
#train, train_labels, test, test_labels = load_datasets.load_abalone_dataset(train_ratio)
#classif_knn = classif_knn_iris

# --> Entrainement du classifieur
# Recherche des meilleurs kppv (décommenter si vous souhaitez le calculer)
#classif_knn.getBestKppv(train, train_labels)
#classif_knn.plotAccuracy()
classif_knn.train(train, train_labels)
"""
Après avoir fait l'entrainement, nous évaluons notre modèle sur 
les données d'entrainement.
"""
print("\n######################################\nDonnées d'entraînement avec K = {}".format(classif_knn.k))
classif_knn.evaluate(train, train_labels)

# --> Test du classifieur
"""
Finalement, nous évaluons votre modèle sur les données de test.
"""
print("\n######################################\nDonnées de test avec K = {}".format(classif_knn.k))
classif_knn.evaluate(test, test_labels)

# --> Comparaison avec le classifieur de la bibliothèque sklearn
knn_model = neighbors.KNeighborsClassifier(n_neighbors=classif_knn.k) # on prend le même nombre de voisins
knn_model.fit(train, train_labels)
predictions = knn_model.predict(test)
print("\n######################################\nRésultats sklearn avec K = {}".format(classif_knn.k))
metrics.show_metrics(test_labels, predictions)

