import numpy as np
import sys
import load_datasets
import NeuralNet# importer la classe du NeuralNet
#importer d'autres fichiers et classes si vous en avez développés
import metrics
from time import perf_counter
import matplotlib.pyplot as plt

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

# --> 1- Initialisation des classifieurs avec leurs paramètres
classif_nn_iris = NeuralNet.NeuralNet()
classif_nn_wine = NeuralNet.NeuralNet()
classif_nn_abalone = NeuralNet.NeuralNet()

# --> 2- Chargement du dataset
# 1) jeu iris
iris = load_datasets.load_iris_dataset(train_ratio)
# 2) jeu wine
wine = load_datasets.load_wine_dataset(train_ratio)
# 3) jeu abalone
abalone = load_datasets.load_abalone_dataset(train_ratio)

# index des variables factorielles

for dataset in ["iris"]:
    train, train_labels, test, test_labels = eval(dataset)
    ############################################################################
    # NeuralNet
    ############################################################################
    classif_nn = eval("classif_nn_"+dataset)
    print("\n######################################\nClassification NeuralNet / Dataset étudié = {}".format(dataset))

    # --> Entraînement du classifieur
    tps1 = perf_counter()
    history = classif_nn.train(train, train_labels, batch_size = 64, epochs=10, learning_rate=0.1)

    # --> Evaluation sur les données d'entraînement
    print("\n######################################\nEvaluation sur les données d'entraînement")
    classif_nn.evaluate(train, train_labels)

    # --> Evaluation sur les données de test
    print("\n######################################\nEvaluation sur les données de test")
    classif_nn.evaluate(test, test_labels)
    tps2 = perf_counter() # utilisé pour calculer les performances, avec seulement l'évaluation sur les données test et pas de print
    print("\nTemps d'exécution :", tps2-tps1)

    """
    # --> Avec sklearn
    model_NeuralNet = NeuralNetClassifier()
    model_NeuralNet = model_NeuralNet.fit(train, train_labels)
    plot_tree(model_NeuralNet)
    plt.show()
    predictions = model_NeuralNet.predict(test)
    print("\n######################################\nRésultats sklearn Decision Tree")
    metrics.show_metrics(test_labels, predictions)
    """




