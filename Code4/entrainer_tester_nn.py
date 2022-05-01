
############################################################################
# NeuralNet
############################################################################
# Ce fichier permet de tester les résultats du réseau de neurones sur les 
# différents datasets.
############################################################################

import numpy as np
import sys
import load_datasets
import NeuralNet # importer la classe du NeuralNet
import metrics
from time import perf_counter
import matplotlib.pyplot as plt

"""
C'est le fichier main duquel nous allons tout lancer pour les réseaux de neurones.
Nous allons :
1- Initialiser les classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""

train_ratio = 0.7
init_weight_null = False

# --> 1- Initialisation des classifieurs avec leurs paramètres
#        nb_entrees          : taille de la couche d'entrée (nombre d'attributs)
#        nb_sorties          : taille de la couche de sortie (nombre de classes)
#        nb_hidden_layers    : nombre de couches cachées
#        nb_neurones         : nombre de neurones dans chaque couche cachée
#        batch_size          : taille de batch utilisée
#        epochs              : nombre d'époques utilisé
#        learning_rate       : taux d'apprentissage
#        weight_null         : booléen valant True si l'on souhaite une initialisation nulle des poids du réseau

classif_nn_iris = NeuralNet.NeuralNet(nb_entrees = 4, nb_sorties = 3, nb_hidden_layers = 2, nb_neurones =4, batch_size = 16, epochs = 1000, learning_rate = 0.5, weight_null=init_weight_null)
classif_nn_wine = NeuralNet.NeuralNet(nb_entrees = 11, nb_sorties = 2, nb_hidden_layers = 1, nb_neurones = 6, batch_size = 64, epochs = 20000, learning_rate = 0.5, weight_null=init_weight_null)
classif_nn_abalone = NeuralNet.NeuralNet(nb_entrees = 8, nb_sorties = 3, nb_hidden_layers = 1, nb_neurones = 10, batch_size = 64, epochs = 1000, learning_rate = 0.5, weight_null=init_weight_null)

# --> 2- Chargement du dataset
# 1) jeu iris
iris = load_datasets.load_iris_dataset(train_ratio)
# 2) jeu wine
wine = load_datasets.load_wine_dataset(train_ratio)
# 3) jeu abalone
abalone = load_datasets.load_abalone_dataset(train_ratio)

def binarize(z):
    """
    fonction qui permet de binariser les labels (écrits sous forme de vecteurs encodés)
    """
    return(z[:, None] == np.arange(z.max()+1)).astype(int)


for dataset in ["iris", "wine", "abalone"]:
    train, train_labels, test, test_labels = eval(dataset)

    # on convertit les classes en vecteurs one-hot (bit à 1 à l'index de la classe)
    train_labels, test_labels = binarize(train_labels), binarize(test_labels) 

    # recherche des meilleurs paramétrages
    #best_params = NeuralNet.grid_search(train, train_labels)
    #print(best_params)
   
    classif_nn = eval("classif_nn_"+dataset)
    print("\n######################################\nClassification NeuralNet / Dataset étudié = {}".format(dataset))

    # --> Entraînement du classifieur
    tps1 = perf_counter()
    history = classif_nn.train(train.T, train_labels.T, print_every = 200)
    tps2 = perf_counter()

    # --> Evaluation sur les données d'entraînement
    print("\n######################################\nEvaluation sur les données d'entraînement")
    #classif_nn.evaluate(train, train_labels)

    # --> Evaluation sur les données de test
    print("\n######################################\nEvaluation sur les données de test")
    classif_nn.evaluate(test.T, test_labels.T)
    tps3 = perf_counter() # utilisé pour calculer les performances, avec seulement l'évaluation sur les données test et pas de print
  
    print("\nTemps d'apprentissage :", tps2-tps1)
    print("\nTemps d'exécution :", tps3-tps1)
    
    tps1 = perf_counter()
    y_pred = classif_nn.predict(test[0,:].T)
    tps2 = perf_counter() # utilisé pour calculer les performances, avec seulement l'évaluation sur un exemple de données test
    print("\nTemps de prédiction d'un exemple :", tps2-tps1)
    
    # courbes loss et accuracy
    #NeuralNet.plot_history(history)
   

