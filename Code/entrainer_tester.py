import numpy as np
import sys
import load_datasets
import NaiveBayes # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés
import metrics
from sklearn import neighbors # utilisation pour comaparer
from sklearn.naive_bayes import GaussianNB  #importer the Gaussian Naive Bayes model
from time import perf_counter

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

# --> 1- Initialisation des classifieurs avec leurs paramètres
classif_knn_iris = Knn.Knn(repeat_kfold=30, L=5, k=8)
classif_knn_wine = Knn.Knn(repeat_kfold=10, L=20, k=1)
classif_knn_abalone = Knn.Knn(repeat_kfold=30, L=10, k=18)
classif_NaiveBayes_iris = NaiveBayes.BayesNaif()
classif_NaiveBayes_wine = NaiveBayes.BayesNaif()
classif_NaiveBayes_abalone = NaiveBayes.BayesNaif()

# --> 2- Chargement du dataset
# 1) jeu iris
iris = load_datasets.load_iris_dataset(train_ratio)
# 2) jeu wine
wine = load_datasets.load_wine_dataset(train_ratio)
# 3) jeu abalone
abalone = load_datasets.load_abalone_dataset(train_ratio)

for dataset in ["iris", "wine", "abalone"]:
    train, train_labels, test, test_labels = eval(dataset)
    ############################################################################
    # KNN
    ############################################################################
    classif_knn = eval("classif_knn_"+dataset)
    print("\n######################################\nClassification KNN / Dataset étudié = {}".format(dataset))

    # --> 3- Entrainement du classifieur
    # 1) Recherche du meilleur k (décommenter si vous souhaitez le calculer)
    #classif_knn.getBestKppv(train, train_labels)
    #classif_knn.plotAccuracy()
    # 2) Entraînement avec ce K
    tps1 = perf_counter()
    classif_knn.train(train, train_labels)

    # --> Evaluation sur les données d'entrainement'
    #print("\n######################################\nEvaluation sur les données d'entraînement avec K = {}".format(classif_knn.k))
    #classif_knn.evaluate(train_iris, train_labels_iris)
    # --> 4- Evaluation sur les données de test
    print("\n######################################\nEvaluation sur les données de test avec K = {}".format(classif_knn.k))
    classif_knn.evaluate(test, test_labels)
    tps2 = perf_counter() # utilisé pour calculer les performances, avec seulement l'évaluation sur les données test et pas de print
    print("\nTemps d'exécution :", tps2-tps1)

    # --> Comparaison avec le classifieur de la bibliothèque sklearn
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=classif_knn.k) # on prend le même nombre de voisins
    knn_model.fit(train, train_labels)
    predictions = knn_model.predict(test)
    print("\n######################################\nRésultats sklearn avec K = {}".format(classif_knn.k))
    metrics.show_metrics(test_labels, predictions)


    ############################################################################
    # Bayes Naif
    classif_NaiveBayes = eval("classif_NaiveBayes_"+dataset)
    print("\n############################################################################\n--> Bayes Naif / Dataset étudié = {}".format(dataset))
    
    # --> Entraînement du classifieur
    tps1 = perf_counter()
    classif_NaiveBayes.train(train, train_labels)

    # --> Evaluation sur les données d'entraînement
    #print("\n######################################\nEvaluation sur les données d'entraînement")
    #classif_NaiveBayes.evaluate(train, train_labels)

    # --> Evaluation sur les données de test
    print("\n######################################\nEvaluation sur les données de test")
    classif_NaiveBayes.evaluate(test, test_labels)
    tps2 = perf_counter() # utilisé pour calculer les performances, avec seulement l'évaluation sur les données test et pas de print
    print("\nTemps d'exécution :", tps2-tps1)

    # --> Avec sklearn
    model_naif = GaussianNB()
    model_naif.fit(train, train_labels)
    predictions = model_naif.predict(test)
    print("\n######################################\nRésultats sklearn Bayes Naïf")
    metrics.show_metrics(test_labels, predictions)




