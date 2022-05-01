############################################################################
# DecisionTree
############################################################################
# Ce fichier permet de générer les courbes d'apprentissage pour DecisionTree.
############################################################################

from ctypes import sizeof
import numpy as np
import sys
import load_datasets
import DecisionTree # importer la classe de l'arbre de décision

from sklearn import neighbors # utilisation pour comaparer
from sklearn.naive_bayes import GaussianNB  #importer the Gaussian Naive Bayes model
from time import perf_counter

train_ratio = 0.7

# --> 1- Initialisation des classifieurs avec leurs paramètres
classif_decisionTree_iris = DecisionTree.DecisionTree(names = ["Longueur sépale", "Largeur sépale", "Longueur pétale", "Largeur pétale"], index_fact = None, conversion_labels={'0': 'Iris-setosa', '1' : 'Iris-versicolor', '2' : 'Iris-virginica'})
classif_decisionTree_wine = DecisionTree.DecisionTree(names = ["Acidité fixe", "Acide volatile", "Acide citrique", "Sucre résiduel", "Ch de sodium", "Dioxyde de soufre libre", "Dioxyde de soufre total", "densité", "pH", "sulfate de potassium", "alcool"],index_fact = None, conversion_labels=None)
classif_decisionTree_abalone = DecisionTree.DecisionTree(names = ["Sexe", "Longueur coquille", "Diamètre coquille", "Hauteur", "Poids total", "Poids chair", "Poids viscères", "Poids coquille"], index_fact = None, conversion_labels=None)

# --> 2- Chargement du dataset
# 1) jeu iris
iris = load_datasets.load_iris_dataset(train_ratio, normalize_data=False)
# 2) jeu wine
wine = load_datasets.load_wine_dataset(train_ratio, normalize_data=False)
# 3) jeu abalone
abalone = load_datasets.load_abalone_dataset(train_ratio, normalize_data=False)

for dataset in ["iris", "wine", "abalone"]:
    train, train_labels, test, test_labels = eval(dataset)
    list_acc, list_size = [], []
    for seed in range(20):
        ############################################################################
        # DecisionTree
        ############################################################################
        classif_decisionTree = eval("classif_decisionTree_"+dataset)

        train, train_labels = train[0:100],  train_labels[0:100]
        print("\n######################################\nClassification DecisionTree / Dataset étudié = {}".format(dataset))
        print(seed)
        acc, size = classif_decisionTree.build_learning_curve(train, train_labels, seed, do_pruning = True)
        list_acc.append(acc)
        list_size.append(size)

    classif_decisionTree.show_learning_curve(list_acc, list_size, dataset, do_pruning = True)





