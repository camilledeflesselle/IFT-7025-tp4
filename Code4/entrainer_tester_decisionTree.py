############################################################################
# DecisionTree
############################################################################
# Ce fichier permet de tester les résultats de l'arbre de décisions sur les 
# différents datasets.
############################################################################

import numpy as np
import sys
import load_datasets
import DecisionTree # importer la classe de l'arbre de décision
import NeuralNet# importer la classe du DecisionTree
#importer d'autres fichiers et classes si vous en avez développés
import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree # utilisés pour comparer
from time import perf_counter
import matplotlib.pyplot as plt
import math

"""
Nous allons :
1- Initialiser les classifieurs avec leurs paramètres
2- Charger les datasets
3- Entraîner chaque classifieur
4- Le tester
"""
train_ratio = 0.7

# --> 1- Initialisation des classifieurs avec leurs paramètres
classif_decisionTree_iris = DecisionTree.DecisionTree(names = ["Longueur sépale", "Largeur sépale", "Longueur pétale", "Largeur pétale"], index_fact = None, conversion_labels={'0': 'Iris-setosa', '1' : 'Iris-versicolor', '2' : 'Iris-virginica'})
classif_decisionTree_wine = DecisionTree.DecisionTree(names = ["Acidité fixe", "Acide volatile", "Acide citrique", "Sucre résiduel", "Ch de sodium", "Dioxyde de soufre libre", "Dioxyde de soufre total", "densité", "pH", "sulfate de potassium", "alcool"],index_fact = None, conversion_labels=None)
classif_decisionTree_abalone = DecisionTree.DecisionTree(index_fact = [0], names = ["Sexe", "Longueur coquille", "Diamètre coquille", "Hauteur", "Poids total", "Poids chair", "Poids viscères", "Poids coquille"], conversion_labels=None)

# --> 2- Chargement du dataset
# 1) jeu iris
iris = load_datasets.load_iris_dataset(train_ratio, normalize_data=False)
# 2) jeu wine
wine = load_datasets.load_wine_dataset(train_ratio, normalize_data=False)
# 3) jeu abalone
abalone = load_datasets.load_abalone_dataset(train_ratio, normalize_data=False)

# index des variables factorielles

for dataset in ["abalone"]:
    train, train_labels, test, test_labels = eval(dataset)

    classif_decisionTree = eval("classif_decisionTree_"+dataset)
    print("\n######################################\nClassification DecisionTree / Dataset étudié = {}".format(dataset))

    # --> Entraînement du classifieur
    tps1 = perf_counter()
    decision_tree = classif_decisionTree.train(train, train_labels)
    tps2 = perf_counter()
    #if dataset == "iris" : classif_decisionTree.drawTree(decision_tree, dataset)
    #print(decision_tree)

    # --> Evaluation sur les données d'entraînement
    #print("\n######################################\nEvaluation sur les données d'entraînement")
    #classif_decisionTree.evaluate(train, train_labels, decision_tree)

    # --> Evaluation sur les données de test
    print("\n######################################\nEvaluation sur les données de test")
    print("\nTemps d'apprentissage :", tps2-tps1)
    classif_decisionTree.evaluate(test, test_labels, decision_tree)
    tps3 = perf_counter() # utilisé pour calculer les performances, avec seulement l'évaluation sur les données test et pas de print
    print("\nTemps d'exécution :", tps3-tps1)

    tps4 = perf_counter()
    decision_tree = classif_decisionTree.predict(test[0,:], decision_tree)
    tps5 = perf_counter()
    print("\nTemps de prédiction d'un exemple :", tps5-tps4)

    # --> Arbre élagué
    classif_decisionTree = eval("classif_decisionTree_"+dataset)
    tps1 = perf_counter()
    decision_tree = classif_decisionTree.train(train, train_labels)
    decision_tree = classif_decisionTree.pruningTree(decision_tree, train, train_labels)
    #classif_decisionTree.drawTree(decision_tree, dataset, "elague")
    tps2 = perf_counter()

    # --> Evaluation sur les données de test
    print("\n######################################\nEvaluation sur les données de test / arbre élagué")
    print("\nTemps d'apprentissage :", tps2-tps1)
    classif_decisionTree.evaluate(test, test_labels, decision_tree)
     
    tps3 = perf_counter()
    print("\nTemps d'exécution :", tps3-tps1)

    tps4 = perf_counter()
    decision_tree = classif_decisionTree.predict(test[0,:], decision_tree)
    tps5 = perf_counter()
    print("\nTemps de prédiction d'un exemple :", tps5-tps4)

    # --> Avec sklearn
    model_decisionTree = DecisionTreeClassifier()
    model_decisionTree = model_decisionTree.fit(train, train_labels)
    plot_tree(model_decisionTree)
    plt.show()
    predictions = model_decisionTree.predict(test)
    print("\n######################################\nRésultats sklearn Decision Tree")
    metrics.show_metrics(test_labels, predictions)




