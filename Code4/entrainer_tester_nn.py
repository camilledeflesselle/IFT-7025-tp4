import numpy as np
import sys
import load_datasets
import NeuralNet# importer la classe du NeuralNet
#importer d'autres fichiers et classes si vous en avez développés
import metrics
from time import perf_counter
import matplotlib.pyplot as plt
import NN
from sklearn import datasets



"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester

test = NeuralNet.NeuralNet(namesAtt=["Att1", "Att2"])
train_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
train = np.array([[1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1], [1, 1, 1, 1, 1, 2,2, 2, 1, 1, 2, 2], [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]]).T
#print(test.determineBestColumn(train, train_labels))
decision_tree = test.train(train, train_labels)

print(decision_tree)

test.evaluate(train, train_labels, decision_tree)
"""
train_ratio = 0.7
init_weight_null = False

# --> 1- Initialisation des classifieurs avec leurs paramètres
classif_nn_iris = NN.NeuralNet(nb_entrees = 4, nb_sorties = 3, nb_hidden_layers = 2, nb_neurones =4, batch_size = 16, epochs = 1000, learning_rate = 0.5, weight_null=init_weight_null)
classif_nn_wine = NN.NeuralNet(nb_entrees = 11, nb_sorties = 2, nb_hidden_layers = 1, nb_neurones = 6, batch_size = 64, epochs = 20000, learning_rate = 0.5, weight_null=init_weight_null)
classif_nn_abalone = NN.NeuralNet(nb_entrees = 8, nb_sorties = 3, nb_hidden_layers = 1, nb_neurones = 10, batch_size = 64, epochs = 1000, learning_rate = 0.5, weight_null=init_weight_null)

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


for dataset in ["abalone"]:
   
    train, train_labels, test, test_labels = eval(dataset)

    train_labels, test_labels = binarize(train_labels), binarize(test_labels)
    best_params = NN.grid_search(train, train_labels)
    print(best_params)
   

    #print(np.array([train_labels]))
    ############################################################################
    # NeuralNet
    ############################################################################
    classif_nn = eval("classif_nn_"+dataset)
    print("\n######################################\nClassification NeuralNet / Dataset étudié = {}".format(dataset))

    # --> Entraînement du classifieur
    tps1 = perf_counter()
    history = classif_nn.train(train.T, train_labels.T)
  
    #NN.plot_history(history)
    # --> Evaluation sur les données d'entraînement
    print("\n######################################\nEvaluation sur les données d'entraînement")
    #classif_nn.evaluate(train, train_labels)

    # --> Evaluation sur les données de test
    print("\n######################################\nEvaluation sur les données de test")
    classif_nn.evaluate(test.T, test_labels.T)
    tps2 = perf_counter() # utilisé pour calculer les performances, avec seulement l'évaluation sur les données test et pas de print
    print("\nTemps d'exécution :", tps2-tps1)

    tps1 = perf_counter()
    y_pred = classif_nn.predict(test[0,:].T)
    tps2 = perf_counter() # utilisé pour calculer les performances, avec seulement l'évaluation sur les données test et pas de print
    print("\nTemps de prédiction d'un exemple :", tps2-tps1)
    
   

