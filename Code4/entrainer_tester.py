import numpy as np
import sys
import load_datasets
import DecisionTree # importer la classe de l'arbre de décision
import NeuralNet# importer la classe du DecisionTree
#importer d'autres fichiers et classes si vous en avez développés

from sklearn import neighbors # utilisation pour comaparer
from sklearn.naive_bayes import GaussianNB  #importer the Gaussian Naive Bayes model
from time import perf_counter

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester

test = DecisionTree.DecisionTree(namesAtt=["Att1", "Att2"])
train_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
train = np.array([[1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1], [1, 1, 1, 1, 1, 2,2, 2, 1, 1, 2, 2], [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]]).T
#print(test.determineBestColumn(train, train_labels))
decision_tree = test.train(train, train_labels)

print(decision_tree)

test.evaluate(train, train_labels, decision_tree)
"""
train_ratio = 0.7

# --> 1- Initialisation des classifieurs avec leurs paramètres
classif_decisionTree_iris = DecisionTree.DecisionTree(index_fact = None)
classif_decisionTree_wine = DecisionTree.DecisionTree(index_fact = None)
classif_decisionTree_abalone = DecisionTree.DecisionTree(index_fact = None)

# --> 2- Chargement du dataset
# 1) jeu iris
iris = load_datasets.load_iris_dataset(train_ratio)
# 2) jeu wine
wine = load_datasets.load_wine_dataset(train_ratio)
# 3) jeu abalone
abalone = load_datasets.load_abalone_dataset(train_ratio)

# index des variables factorielles

for dataset in ["iris", "wine", "abalone"]:
    train, train_labels, test, test_labels = eval(dataset)
    ############################################################################
    # DecisionTree
    ############################################################################
    classif_decisionTree = eval("classif_decisionTree_"+dataset)
    print("\n######################################\nClassification DecisionTree / Dataset étudié = {}".format(dataset))

    # --> Entraînement du classifieur
    tps1 = perf_counter()
    decision_tree = classif_decisionTree.train(train, train_labels)
    #print(decision_tree)

    # --> Evaluation sur les données d'entraînement
    #print("\n######################################\nEvaluation sur les données d'entraînement")
    #classif_decisionTree.evaluate(train, train_labels, decision_tree)

    # --> Evaluation sur les données de test
    print("\n######################################\nEvaluation sur les données de test")
    classif_decisionTree.evaluate(test, test_labels, decision_tree)
    tps2 = perf_counter() # utilisé pour calculer les performances, avec seulement l'évaluation sur les données test et pas de print
    print("\nTemps d'exécution :", tps2-tps1)






