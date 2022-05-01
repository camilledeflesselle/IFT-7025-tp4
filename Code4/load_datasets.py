import numpy as np
import math

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def load_iris_dataset(train_ratio, seed=1, normalize_data = True):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    np.random.seed(seed) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    # le code ici pour lire le dataset
    f = open('datasets/bezdekIris.data', 'r')
    lines = [line.strip() for line in f.readlines()]
    f.close()
    lines = [line.split(",") for line in lines if line]
    data=np.array([line[:4] for line in lines], dtype=float)
    # l'étiquette à prédire est la dernière colonne
    labels=np.array([conversion_labels[line[-1]] for line in lines], dtype=int)
    nrow = len(data)
    indices = np.arange(nrow)

    # on normalise les données pour plus de précision 
    if normalize_data : data = NormalizeData(data)

	  # les exemples sont ordonnés dans le fichier du dataset, ils sont ordonnés par type de fleur
    # on utilise donc la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test
    np.random.shuffle(indices)
    split = math.floor(train_ratio * nrow)

    train_idx,    test_idx    = indices[:split],   indices[split:]
    train,        test        = data[train_idx],   data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    # la fonction retourne 4 matrices de type Numpy. 
    return (train, train_labels, test, test_labels)
	
	
def load_wine_dataset(train_ratio, seed=1, normalize_data = True):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    np.random.seed(seed) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/binary-winequality-white.csv', 'r')
    lines = [line.strip() for line in f.readlines()]
    f.close()
    lines = [line.split(",") for line in lines if line]
    data=np.array([line[:-1] for line in lines], dtype=float)

    # on normalise les données pour plus de précision 
    if normalize_data : data = NormalizeData(data)

    # l'étiquette à prédire est la dernière colonne
    labels=np.array([line[-1] for line in lines], dtype=int)
    nrow = len(data)
    indices = np.arange(nrow)
    np.random.shuffle(indices)
    split = math.floor(train_ratio * nrow)

    train_idx,    test_idx    = indices[:split],   indices[split:]
    train,        test        = data[train_idx],   data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
	
	# La fonction retourne 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)

def load_abalone_dataset(train_ratio, seed=1, normalize_data = True):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le reste des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    np.random.seed(seed) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    f = open('datasets/abalone-intervalles.csv', 'r') 
    lines = [line.strip() for line in f.readlines()]
    f.close()

    lines = [line.split(",") for line in lines if line]
    conversion_sexe = {'M': 0, 'F' : 1, 'I' : 2}
    for line in lines : # on convertit les strings en integer
      line[0] = conversion_sexe[line[0]]
    data=np.array([line[:-1] for line in lines], dtype=float)

    # on normalise les données pour plus de précision 
    if normalize_data : data = NormalizeData(data)

    # l'étiquette à prédire est la dernière colonne
    labels=np.array([line[-1] for line in lines], dtype=float)

    nrow = len(data)
    indices = np.arange(nrow)
    np.random.shuffle(indices)
    split = math.floor(train_ratio * nrow)

    train_idx,    test_idx    = indices[:split],   indices[split:]
    train,        test        = data[train_idx],   data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    # La fonction retourne 4 matrices (ou vecteurs) de type Numpy.
    return (train, train_labels, test, test_labels)
