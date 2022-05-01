# IFT-7025-tp4

## Auteurs
- Lucas Chollet
- Camille Deflesselle

## Description des classes

### Classe DecisionTree

Cette classe est le classifieur implémenté dans le fichier *DecisionTree.py*. Il s'agit de la classification en utilisant la méthode d'arbres de décision. Son initialisation peut prendre en argument trois paramètres facultatifs :
- **names** : noms de chaque attribut utilisés lors de la construction de l'arbre (meilleure compréhension)
- **index_fact** : liste des index des attributs qui sont des attributs catégoriels
- **conversion_labels** : dictionnaire de conversion pour l'affichage de l'arbre avec les étiquettes initiales

 Les méthodes de cette classe sont les suivantes :
 - **calculateEntropy** : calcule l'entropie d'un vecteur
 - **calculateGlobalEntropie** : calcule l'entropie associée à un attribut
 - **classifyData** : renvoie la classe majoritaire d'un vecteur
 - **determineBestColumn** : calcule le gain d'information sur chaque attribut pour différentes valeurs de coupes et renvoie l'attribut qui maximise le gain et la valeur de coupe associée
 - **getPotentialSplits** : retourne les différentes valeurs de coupes par attribut
 - **train** : permet d'identifier les données d'entraînement
 - **predict** : permet de prédire la classe d'une instance
 - **evaluate** : permet de stocker toutes les prédictions d'un jeu de données test et d'évaluer les performances de l'algorithme (Accuracy, Précision, Rappel, Score F1 et matrice de confusion)
 - **pruningLeaves** : méthode qui fait un élagage avec le test chi squarred
 - **pruningTree** : méthode qui fait l'élagage d'un arbre complet par appel à pruningLeaves.
 - **build_learning_curve** et **show_learning_curve** : permettent de voir si notre modèle apprend correctement
 - **extractEdgesFromTree** et **drawTree** : permettent de représenter l'arbre de décision (.png)
 
 ### Classe NeuralNet

Cette classe est le classifieur implémenté dans le fichier *NeuralNet.py*. Il s'agit de la classification avec réseau de neurones. Son initialisation prend en entrée les arguments suivants :
- **nb_entrees** : taille de la couche d'entrée (nombre d'attributs)
- **nb_sorties**  : taille de la couche de sortie (nombre de classes)
- **nb_hidden_layers** : nombre de couches cachées
- **nb_neurones**  : nombre de neurones dans chaque couche cachée
- **batch_size** : taille de batch utilisée
- **epochs** : nombre d'époques utilisé
- **learning_rate** : taux d'apprentissage
- **weight_null** : booléen valant True si l'on souhaite une initialisation nulle des poids du réseau
- **seed**  : graine pour fixer le hasard

 Les méthodes de cette classe sont les suivantes :
 - **train** : permet d'entraîner le modèle avec les données d'entraînement
 - **predict** : permet de prédire la classe d'une instance
 - **evaluate** : permet de stocker toutes les prédictions d'un jeu de données test et d'évaluer les performances de l'algorithme (Accuracy, Précision, Rappel, Score F1 et matrice de confusion)
 - **forward** : réalise la propagation avant
 - **update_deltas**  : renvoie la liste contenant les valeurs de delta de chaque couche
 - **back_propagation** : réalise la propagation arrière et calcul du gradient 

## Répartition des tâches de travail entre les membres d’équipe
Pour faciliter notre collaboration, nous avons créé un dépôt git privé, sur lequel se trouve tout notre travail.

Pour ce projet, l'un des membres de l'équipe a implémenté la classe DecisionTree et l'autre la classe NeuralNetwork.
Quant aux fonctions dédiées au chargement des datasets, nous les avons écrit ensemble.

Nous avons implémenté une boucle d'entraînement/test (fichier *entrainer_tester_decisionTree.py*) en utilisant la classe DecisionTree sur les trois jeux de données étudiés en travaillant ensemble sur le fichier. De même nous avons utilisé le fichier *entrainer_tester_nn.py* pour la classe NeuralNet. Ces fichies nous ont permis de connaître les temps d'exécution des différents classifieurs (temps d'entraînement + évaluation sur les données test).

Par ailleurs, nous avions au préalable implémenté le fichier *metrics.py* qui nous permet d'afficher les différentes métriques de performances, que nous utilisons dans nos classes lors de l'évaluation.

## Explication des difficultés rencontrées dans ce travail

Globalement, pour ce travail tout s'est bien déroulé. Nos réflexions se sont essentiellement tournées vers le choix des hyperparamètres pour l'implémentation de l'algorithme NeuralNetwork et sur la création de l'arbre de décision et plus particulièrement son élagage. Pour le premier jeu de données, iris dataset, qui ne contient que 150 instances, les résultats sont très bons et rapides à obtenir, sachant que le jeu de données est plutôt équilibré.

Finalement, nous pensons nous être bien approprié ces deux algorithmes d'apprentissage et avoir bien compris leur fonctionnement.
