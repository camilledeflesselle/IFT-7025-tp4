import numpy as np
import sys
import load_datasets
import NaiveBayes # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés
import metrics

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initialisez vos paramètres

L = 5
repeat_kfold = 20
train_ratio = 0.7


# Initialisez/instanciez vos classifieurs avec leurs paramètres

classif_knn = Knn.Knn()


# Charger/lire les datasets

# jeu iris
train, train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)
# jeu wine
#train, train_labels, test, test_labels = load_datasets.load_wine_dataset(train_ratio)
# jeu abalone
#train, train_labels, test, test_labels = load_datasets.load_abalone_dataset(train_ratio)


# Entrainez votre classifieur
classif_knn.getBestKppv(train, train_labels)

print("\n######################################\nDonnées d'entraînement")

classif_knn.train(train, train_labels)
#classif_knn.plotAccuracy()
"""
Après avoir fait l'entrainement, nous évaluons notre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""

classif_knn.evaluate(train, train_labels)

# Tester votre classifieur
"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""

print("\n######################################\nDonnées de test")
classif_knn.evaluate(test, test_labels)





