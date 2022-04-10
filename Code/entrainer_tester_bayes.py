import numpy as np
import sys
import load_datasets
import NaiveBayes # importer la classe du classifieur bayesien
import metrics
from sklearn.naive_bayes import GaussianNB  #importer the Gaussian Naive Bayes model
from time import perf_counter

# 1) iris
# --> Initialisation des classifieurs
train_ratio = 0.7
classif_NaiveBayes = NaiveBayes.BayesNaif()

# --> Chargement des datasets
# jeu iris
train, train_labels, test, test_labels = load_datasets.load_iris_dataset(train_ratio)
dataset = "iris"
print("\n############################################################################\n--> Bayes Naif / Dataset étudié = {}".format(dataset))

# --> Entrainez votre classifieur
tps1 = perf_counter()
classif_NaiveBayes.train(train, train_labels)

# --> Evaluation sur les données d'entraînement
print("\n######################################\nEvaluation sur les données d'entraînement")
#classif_NaiveBayes.evaluate(train, train_labels)

# --> Evaluation sur les données de test
print("\n######################################\nEvaluation sur les données de test")
classif_NaiveBayes.evaluate(test, test_labels)
tps2 = perf_counter()
print("\nTemps d'exécution :", tps2-tps1)

# --> Avec sklearn
model_naif = GaussianNB()
model_naif.fit(train, train_labels)
predictions = model_naif.predict(test)
print("\n######################################\nRésultats sklearn Bayes Naïf")
metrics.show_metrics(test_labels, predictions)

# 2) wine
# --> Initialisation des classifieurs
classif_NaiveBayes = NaiveBayes.BayesNaif()

# --> Chargement des datasets
# jeu wine
train, train_labels, test, test_labels = load_datasets.load_wine_dataset(train_ratio)
dataset = "wine"
print("\n############################################################################\n--> Bayes Naif / Dataset étudié = {}".format(dataset))
# --> Entrainez votre classifieur
tps1 = perf_counter()
classif_NaiveBayes.train(train, train_labels)
# --> Evaluation sur les données d'entraînement
print("\n######################################\nEvaluation sur les données d'entraînement")
#classif_NaiveBayes.evaluate(train, train_labels)
# --> Evaluation sur les données de test
print("\n######################################\nEvaluation sur les données de test")
classif_NaiveBayes.evaluate(test, test_labels)
tps2 = perf_counter()
print("\nTemps d'exécution :", tps2-tps1)

# --> Avec sklearn
model_naif = GaussianNB()
model_naif.fit(train, train_labels)
predictions = model_naif.predict(test)
print("\n######################################\nRésultats sklearn Bayes Naïf")
metrics.show_metrics(test_labels, predictions)

# 3) abalone
# --> Initialisation des classifieurs
classif_NaiveBayes = NaiveBayes.BayesNaif()
# --> Chargement des datasets
# jeu abalone
train, train_labels, test, test_labels = load_datasets.load_abalone_dataset(train_ratio)
dataset = "abalone"
print("\n############################################################################\n--> Bayes Naif / Dataset étudié = {}".format(dataset))
# --> Entrainez votre classifieur
tps1 = perf_counter()
classif_NaiveBayes.train(train, train_labels)
# --> Evaluation sur les données d'entraînement
print("\n######################################\nEvaluation sur les données d'entraînement")
#classif_NaiveBayes.evaluate(train, train_labels)
# --> Evaluation sur les données de test
print("\n######################################\nEvaluation sur les données de test")
classif_NaiveBayes.evaluate(test, test_labels)
tps2 = perf_counter()
print("\nTemps d'exécution :", tps2-tps1)

# --> Avec sklearn
model_naif = GaussianNB()
model_naif.fit(train, train_labels)
predictions = model_naif.predict(test)
print("\n######################################\nRésultats sklearn Bayes Naïf")
metrics.show_metrics(test_labels, predictions)







