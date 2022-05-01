import math
from matplotlib import pyplot as plt
# numpy for vector and matrix manipulations
import numpy as np
from metrics import show_metrics
# tqdm est une barre de progression (si la librairie n'est pas installée -> pip install tqdm)
from tqdm import tqdm

# fonctions auxiliaires utilisées pour le réseau de neurones
def activation_relu(x,  derivative=False):
    """
    Fonction d'activation relu (ici pas utilisée)
    Cela prend en compte deux modes: normal et le mode "derivative".
    Applique l'opération choisie aux vecteurs
    
    Paramètres:
    z: vecteur de pre-activation pour une couche l,
        il est de taille (n[l], batch_size)

    Retourne: 
    l'activation de chaque élément de l'entrée z
    """
    if derivative :
        y=x
        np.piecewise(y,[activation_relu(y)==0,activation_relu(y)==y],[0,1])
        return y
    else :
        return (np.abs(x)+x)/2

def activation_sigmoid(z, derivative=False):
    """
    Fonction d'activation Sigmoid
    Cela prend en compte deux modes: normal et le mode "derivative".
    Applique l'opération choisie aux vecteurs
    
    Paramètres:
    z: vecteur de pre-activation pour une couche l,
        il est de taille (n[l], batch_size)

    Retourne: 
    l'activation de chaque élément de l'entrée z
    """
    if derivative:
        return activation_sigmoid(z) * (1 - activation_sigmoid(z))
    else:
        return 1 / (1 + np.exp(-z))

def cost_function(y_true, y_pred):
    """
    Fonction qui calcule l'erreur MSE (Mean Square Error) entre un vecteur de labels réels et un vecteur de prédictions
    Paramètres:
    y_true: les labels réels
    y_pred: les prédictions
    Retourne:
    cost: un float, valeur représentant la perte et qui nous permet de tracer la perte.
    """
    n = y_pred.shape[1]
    cost = (1./(2*n)) * np.sum((y_true - y_pred) ** 2)
    return cost

def cost_function_prime(y_true, y_pred):
    """
    Computes the derivative of the loss function w.r.t the activation of the output layer
    Parameters:
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    cost_prime: derivative of the loss w.r.t. the activation of the output
    shape: (n[L], batch_size)    
    """
    cost_prime = y_pred - y_true
    return cost_prime

class NeuralNet():     
    '''
    classe réseaux de neurones
    '''

    def __init__(self, nb_entrees = 1, nb_sorties = 3, nb_hidden_layers = 1, nb_neurones = 3, batch_size = 16, epochs = 1000, learning_rate = 0.5, weight_null = False, seed=42):
        """
		C'est un Initializer. 
        Paramètres:
            nb_entrees          : taille de la couche d'entrée (nombre d'attributs)
            nb_sorties          : taille de la couche de sortie (nombre de classes)
            nb_hidden_layers    : nombre de couches cachées
            nb_neurones         : nombre de neurones dans chaque couche cachée
            batch_size          : taille de batch utilisée
            epochs              : nombre d'époques utilisé
            learning_rate       : taux d'apprentissage
            weight_null         : booléen valant True si l'on souhaite une initialisation nulle des poids du réseau
            seed                : hasard pour fixer le hasard
		"""
        self.seed = seed
        np.random.seed(self.seed)
        self.size = [nb_entrees] # couche entrée
        for i in range(nb_hidden_layers): self.size.append(nb_neurones) # couches cachées
        self.size.append(nb_sorties) # couche de sortie
        print(self.size)

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        if not weight_null :
            # les poids sont les connexions entre deux couches qui se suivent
            # on initizlise les poids avec une distribution gaussienne de moyenne nulle et d'écart type 1/sqrt(n)
            self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
            self.biases = [np.random.rand(n, 1) for n in self.size[1:]]
        else :
            self.weights = [np.zeros((self.size[i], self.size[i-1])) for i in range(1, len(self.size))]
            self.biases = [np.zeros((n, 1)) for n in self.size[1:]]
       

    def forward(self, input):
        '''
        Méthode qui réalise la propagation avant des poids.
        Paramètres:
            input: données qui permette d'enrichir le réseau
            de taille (input_shape, batch_size)
        Retourne:
            a: l'activation de la couche de sortie de taille (output_shape, batch_size)
            pre_activations: la liste des pré activations par couche, de taille (n[l], batch_size)
            avec n[l] le nombre de neurones dans une couche l
            activations: liste des activations par couche
            each of shape (n[l], batch_size), where n[l] is the number 
            of neuron at layer l
        '''
        a = input
        pre_activations = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a  = activation_sigmoid(z)
            pre_activations.append(z)
            activations.append(a)
        return a, pre_activations, activations

    def update_deltas(self, pre_activations, y_true, y_pred):
        """
        Méthode qui renvoie la liste contenant les valeurs de delta de chaque couche
        Paramètres:
            pre_activations: liste des pré-activations. correspondant chacun à une couche
            y_true: valeurs réelles des étiquettes
            y_pred: valeurs de prédiction des étiquettes
        Retourne:
            deltas: une liste de deltas par couche
        """
        delta_L = cost_function_prime(y_true, y_pred) * activation_sigmoid(pre_activations[-1], derivative=True)
        deltas = [0] * (len(self.size) - 1)
        deltas[-1] = delta_L
        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]) * activation_sigmoid(pre_activations[l], derivative=True) 
            deltas[l] = delta
        return deltas

    def back_propagation(self, deltas, pre_activations, activations):
        """
        Méthode qui réalise la back propagation et calcule le gradient avec les poids 
        et les biais du réseau.
        Paramètres:
            deltas: liste des deltas calculés avec update_deltas
            pre_activations: list des pre-activations par couche
            activations: liste des activations par couche
        Retourne:
            dW: liste des gradients sur les poids du réseau
            db: liste des gradients sur les biais du réseau
        """
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.size)):
            dW_l = np.dot(deltas[l], activations[l-1].transpose()) 
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db

    def train(self, train, train_labels, print_every=10, tqdm_=False):
       	"""
		C'est la méthode qui va entrainer le modèle.
        Paramètres:
            train: matrice de type Numpy et de taille mxn, avec 
                n: le nombre d'exemples d'entrainement dans le dataset
                m: le nombre d'attributs (le nombre de caractéristiques)
            train_labels: matrice numpy de taille cxn
                c: nombre de classes 
                --> par exemple :
                        train_labels.T[0,:] = [1 0 0]
                        est un vecteur encodé si on a trois classes
        Retourne:
            history: l'historique de l'entraînement (perte, époques, accuracy)
		"""
        history_train_losses = []
        history_train_accuracies = []
        batch_size = self.batch_size
        epochs = self.epochs 
        learning_rate = self.learning_rate

        if tqdm_:
            epoch_iterator = tqdm(range(epochs))
        else:
            epoch_iterator = range(epochs)

        for e in epoch_iterator:
            if train.shape[1] % batch_size == 0:
                n_batches = int(train.shape[1] / batch_size)
            else:
                n_batches = int(train.shape[1] / batch_size ) - 1

            idx = np.arange(train.shape[1])
            np.random.shuffle(idx)

            train = train[:, idx]
            train_labels = train_labels[:,idx]

            batches_x = [train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            batches_y = [train_labels[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]

            train_losses = []
            train_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases] 
            
            for batch_x, batch_y in zip(batches_x, batches_y):
                batch_y_pred, pre_activations, activations = self.forward(batch_x)
                deltas = self.update_deltas(pre_activations, batch_y, batch_y_pred)
                dW, db = self.back_propagation(deltas, pre_activations, activations)
                for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                batch_y_train_pred = self.predict(batch_x)
             
                train_loss = cost_function(batch_y, batch_y_train_pred)
                train_losses.append(train_loss)

                true = np.argmax(batch_y, axis = 0)
                predictions = np.argmax(batch_y_train_pred, axis=0) 
                train_accuracy = np.mean(true == predictions)

                train_accuracies.append(train_accuracy)


            # weight update
            for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            history_train_losses.append(np.mean(train_losses))
            history_train_accuracies.append(np.mean(train_accuracies))

            if e % print_every == 0:    
                print('Epoque {} / {} | perte entraînement : {} | exactitude entraînement : {}'.format(
                    e, epochs, np.round(np.mean(train_losses), 3), np.round(np.mean(train_accuracies), 3)))

        history = {'epochs': epochs,
                   'train_loss': history_train_losses, 
                   'train_acc': history_train_accuracies
                   }
        return history
        
    def predict(self, a):
        '''
        Utilisation de l'état actuel du réseau de neurones pour faire les prédictions
        Paramètres:
            a: matrice d'entrée, de taille: (input_shape, batch_size)
        Retourne:
            prédictions: matrice de prédictions (probabilité pour chacune des classes) de taille (input_shape, batch_size)
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation_sigmoid(z)
        return a
        
    def evaluate(self, X, y):
        """
		Méthode qui va évaluer votre modèle sur les données X.
        Paramètres:
		    X: matrice de type Numpy et de taille nxm, avec 
                n: le nombre d'exemple de test dans le dataset
                m: le nombre d'attributs (le nombre de caractéristiques)
		    y: est une matrice numpy de taille nx1
		"""
        y_pred_proba = self.predict(X)
        y_pred = np.argmax(y_pred_proba, axis = 0) # on convertit les probabilités en classes
        y = np.argmax(y, axis = 0)
        show_metrics(y, y_pred)

def plot_history(history, show_valid = False):
    """
    Fonction qui permet d'afficher graphiquement l'historique d'un entraînement, 
    la perte et l'exactitude en fonction du nombre d'époques.
    Paramètres:
        history: l'historique d'entraînement (dict)
        show_valid: booléen, True si on souhaite l'affichage des courbes de validation (si elle existe)
    """
    n = history['epochs']
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    n = 4000
    plt.plot(range(history['epochs'])[:n], history['train_loss'][:n], label='Perte en entraînement')
    if show_valid : plt.plot(range(history['epochs'])[:n], history['valid_loss'][:n], label='Perte en validation')
    plt.xlabel('Époques')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(history['epochs'])[:n], history['train_acc'][:n], label='Exactitude en entraînement')
    if show_valid : plt.plot(range(history['epochs'])[:n], history['valid_acc'][:n], label='Exactitude en validation')
    plt.xlabel('Époques')
    plt.legend()
    plt.show()


def grid_search(train, train_labels, list_batch_size=[64], list_epochs=[1000], list_learning_rate=[0.5], k = 10, list_nb_neurones = [10], list_nb_hidden_layers = range(1, 6)):
    """
    Fonction qui permet de rechercher les meilleurs hyperparamètres.
    Permet d'afficher l'erreur moyenne lors de la validation croisée en fonction du paramètre testé.
    Paramètres:
        train: matrice de type Numpy et de taille nxm, avec 
                n: le nombre d'exemples d'entrainement dans le dataset
                m: le nombre d'attributs (le nombre de caractéristiques)
        train_labels: matrice numpy de taille nxc avec c le nombre de classe
        list_batch_size: liste des tailles de batch testées
        list_epochs: liste contenant les nombres d'époques testées
        list_learning_rate: liste contenant les taux d'apprentissage testés
        k: liste contenant le nombre de k-folds
        list_nb_neurones: liste contenant les nombres de neurones dans les couches cachées testés
        list_nb_hidden_layers: liste contenant les nombres testés de couches cachées 
    Retourne:
    Renvoie un tuple contenant les paramètres qui minimisent cette erreur (best_nb_neurones, best_hidden_layers, best_batch_size, best_epochs, best_learning_rate)
    """
    fold_indices = np.arange(train.shape[0])
    valid_indices = np.array_split(fold_indices, k)

    max_mean_accuracy = None
    
    for learning_rate in list_learning_rate:
        for batch_size in list_batch_size :
            for epochs in list_epochs:

                # initialisation d'un tableau qui contient les moyennes des erreurs en validation et en entraînement
                error_valid = []
                error_train = []
                for nb_hidden_layers in list_nb_hidden_layers:
                    for nb_neurones in list_nb_neurones:
                        # 2-1) boucle de validation croisée
                        # listes des accuracies pour tous les k-folds
                        all_accuracy_valid = []
                        all_accuracy_train = []
                        for i_fold in range(k):
                            model = NeuralNet(train.shape[1], train_labels.shape[1], nb_hidden_layers, nb_neurones, batch_size, epochs, learning_rate, weight_null = False)
                            idx = valid_indices[i_fold]
                            # séparation train/validation
                            valid_data, valid_lab = train[idx, :], train_labels[idx, :]
                            train_data, train_lab = np.delete(train, idx, axis =0), np.delete(train_labels, idx, axis = 0)
                            # a) entraînement du modèle
                            history = model.train(train_data.T,train_lab.T, tqdm_=False)
                            # b) prédiction sur les données de validation
                            valid_pred = model.predict(valid_data.T)
                            # c) calcul de l'exactitude2 
                            all_accuracy_valid.append(np.mean(np.argmax(valid_lab.T, axis = 0) == np.argmax(valid_pred, axis=0)))
                            all_accuracy_train.append(history['train_acc'][-1])
                            
                        # 2-2) moyenne des exactitudes pour les hyperparamètres
                        mean_accuracy = np.mean(all_accuracy_valid)
                        error_valid.append(1-mean_accuracy)
                        error_train.append(1-np.mean(all_accuracy_train))
                        if max_mean_accuracy == None or max_mean_accuracy < mean_accuracy :
                            # nous retenons le maximum des exactitudes
                            max_mean_accuracy = mean_accuracy 
                            # et les hyperparamètres associés
                            best_hidden_layers = nb_hidden_layers
                            best_nb_neurones = nb_neurones
                            best_batch_size = batch_size
                            best_epochs = epochs
                            best_learning_rate = learning_rate
    
    if len(list_nb_neurones) > 1 :
        plt.plot(list_nb_neurones, error_train, label ='Erreur moyenne en entraînement')
        plt.plot(list_nb_neurones, error_valid, label ='Erreur moyenne en validation')
        plt.xlabel('Nombre de neurones')
    elif len(list_nb_hidden_layers) > 1 :
        plt.plot(list_nb_hidden_layers, error_train, label ='Erreur moyenne en entraînement')
        plt.plot(list_nb_hidden_layers, error_valid, label ='Erreur moyenne en validation')
        plt.xlabel('Nombre de couches cachées')
    elif len(list_epochs) > 1 :
        plt.plot(list_epochs, error_train, label ='Erreur moyenne en entraînement')
        plt.plot(list_epochs, error_valid, label ='Erreur moyenne en validation')
        plt.xlabel("Nombre d'époques")
    plt.legend()
    plt.show()
    return best_nb_neurones, best_hidden_layers, best_batch_size, best_epochs, best_learning_rate