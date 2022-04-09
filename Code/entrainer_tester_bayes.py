import numpy as np
import sys
import load_datasets
import NaiveBayes # importer la classe du classifieur bayesien
import NaiveBayes # importer la classe du NaiveBayes
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

classif_NaiveBayes = NaiveBayes.BayesNaif()


# Charger/lire les datasets

# jeu iris
#train, train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)
# jeu wine
#train, train_labels, test, test_labels = load_datasets.load_wine_dataset(train_ratio)
# jeu abalone
train, train_labels, test, test_labels = load_datasets.load_abalone_dataset(train_ratio)


# Entrainez votre classifieur
print("\n######################################\nDonnées d'entraînement")

classif_NaiveBayes.train(train, train_labels)
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

classif_NaiveBayes.evaluate(train, train_labels)

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

classif_NaiveBayes.evaluate(test, test_labels)



