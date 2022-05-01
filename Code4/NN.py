"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""
import math
from matplotlib import pyplot as plt
# numpy for vector and matrix manipulations
import numpy as np
from metrics import show_metrics
# tqdm est une barre de progression (si la librairie n'est pas installée -> pip install tqdm)
from tqdm import tqdm

def activation_relu(x,  derivative=False):
    """
    Fonction d'activation relu
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
    
    Entrées:
    ---
    z: vecteur de pre-activation pour une couche l,
        il est de taille (n[l], batch_size)

    Retourne: 
    pontwize activation on each element of the input z
    """
    if derivative:
        return activation_sigmoid(z) * (1 - activation_sigmoid(z))
    else:
        return 1 / (1 + np.exp(-z))

def cost_function(y_true, y_pred):
    """
    Computes the Mean Square Error between a ground truth vector and a prediction vector
    Parameters:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    ---
    cost: a scalar value representing the loss
    """
    n = y_pred.shape[1]
    cost = (1./(2*n)) * np.sum((y_true - y_pred) ** 2)
    return cost

def cost_function_prime(y_true, y_pred):
    """
    Computes the derivative of the loss function w.r.t the activation of the output layer
    Parameters:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    ---
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
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
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
            # ils sont pris aléatoirement entre -1 et 1
            self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
            self.biases = [np.random.rand(n, 1) for n in self.size[1:]]
        else :
            self.weights = [np.zeros((self.size[i], self.size[i-1])) for i in range(1, len(self.size))]
            self.biases = [np.zeros((n, 1)) for n in self.size[1:]]
       

    def forward(self, input):
        '''
        Perform a feed forward computation 

        Parameters
        ---
        input: data to be fed to the network with
        shape: (input_shape, batch_size)

        Returns
        ---
        a: ouptut activation (output_shape, batch_size)
        pre_activations: list of pre-activations per layer
        each of shape (n[l], batch_size), where n[l] is the number 
        of neuron at layer l
        activations: list of activations per layer
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
        Computes a list containing the values of delta for each layer using 
        a recursion
        Parameters:
        ---
        pre_activations: list of of pre-activations. each corresponding to a layer
        y_true: ground truth values of the labels
        y_pred: prediction values of the labels
        Returns:
        ---
        deltas: a list of deltas per layer
        
        """
        delta_L = cost_function_prime(y_true, y_pred) * activation_sigmoid(pre_activations[-1], derivative=True)
        deltas = [0] * (len(self.size) - 1)
        deltas[-1] = delta_L
        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]) * activation_sigmoid(pre_activations[l], derivative=True) 
            deltas[l] = delta
        return deltas

    def backpropagate(self, deltas, pre_activations, activations):
        """
        Applies back-propagation and computes the gradient of the loss
        w.r.t the weights and biases of the network

        Parameters:
        ---
        deltas: list of deltas computed by update_deltas
        pre_activations: a list of pre-activations per layer
        activations: a list of activations per layer
        Returns:
        ---
        dW: list of gradients w.r.t. the weight matrices of the network
        db: list of gradients w.r.t. the biases (vectors) of the network
    
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

    def train1(self, X, y, batch_size, epochs, learning_rate, train_split=0.8, print_every=10, tqdm_=False, plot_every=None):
        """
        Trains the network using the gradients computed by back-propagation
        Splits the data in train and validation splits
        Processes the training data by batches and trains the network using batch gradient descent

        Parameters:
        ---
        X: input data
        y: input labels
        batch_size: number of data points to process in each batch
        epochs: number of epochs for the training
        learning_rate: value of the learning rate
        validation_split: percentage of the data for validation
        print_every: the number of epochs by which the network logs the loss and accuracy metrics for train and validations splits
        tqdm_: use tqdm progress-bar
        plot_every: the number of epochs by which the network plots the decision boundary
    
        Returns:
        ---
        history: dictionary of train and validation metrics per epoch
            train_acc: train accuracy
            test_acc: validation accuracy
            train_loss: train loss
            test_loss: validation loss

        This history is used to plot the performance of the model
        """
        history_train_losses = []
        history_train_accuracies = []
        history_test_losses = []
        history_test_accuracies = []

        nrow = X.shape[1]
        indices = np.arange(nrow)
        split = math.floor(train_split * nrow)
        train_idx,    test_idx    = indices[:split],   indices[split:]
        x_train,        x_test        = X[:, train_idx],   X[:, test_idx]
        y_train, y_test = y[:, train_idx], y[:, test_idx]
		
        if tqdm_:
            epoch_iterator = tqdm(range(epochs))
        else:
            epoch_iterator = range(epochs)

        for e in epoch_iterator:
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1] / batch_size)
            else:
                n_batches = int(x_train.shape[1] / batch_size ) - 1

            idx = np.arange(x_train.shape[1])
            np.random.shuffle(idx)

            x_train = x_train[:, idx]
            y_train = y_train[:,idx]

            batches_x = [x_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            batches_y = [y_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            
            train_losses = []
            train_accuracies = []
            
            test_losses = []
            test_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases] 
            
            for batch_x, batch_y in zip(batches_x, batches_y):
                batch_y_pred, pre_activations, activations = self.forward(batch_x)
                deltas = self.update_deltas(pre_activations, batch_y, batch_y_pred)
                dW, db = self.backpropagate(deltas, pre_activations, activations)
                for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                batch_y_train_pred = self.predict(batch_x)
                print(batch_y)
                print(batch_y_train_pred)
                train_loss = cost_function(batch_y, batch_y_train_pred)
                train_losses.append(train_loss)
                train_accuracy = np.mean(batch_y[0] == batch_y_train_pred[0])
                train_accuracies.append(train_accuracy)
                
                batch_y_test_pred = self.predict(x_test)

                test_loss = cost_function(y_test, batch_y_test_pred)
                test_losses.append(test_loss)
                test_accuracy = np.mean(y_test[0] == batch_y_test_pred[0])
                test_accuracies.append(test_accuracy)


            # weight update
            for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            history_train_losses.append(np.mean(train_losses))
            history_train_accuracies.append(np.mean(train_accuracies))
            
            history_test_losses.append(np.mean(test_losses))
            history_test_accuracies.append(np.mean(test_accuracies))


            if not plot_every:
                if e % print_every == 0:    
                    print('Epoch {} / {} | train loss: {} | train accuracy: {} | val loss : {} | val accuracy : {} '.format(
                        e, epochs, np.round(np.mean(train_losses), 3), np.round(np.mean(train_accuracies), 3), 
                        np.round(np.mean(test_losses), 3),  np.round(np.mean(test_accuracies), 3)))
            else:
                if e % plot_every == 0:
                    self.plot_decision_regions(x_train, y_train, e, 
                                                np.round(np.mean(train_losses), 4), 
                                                np.round(np.mean(test_losses), 4),
                                                np.round(np.mean(train_accuracies), 4), 
                                                np.round(np.mean(test_accuracies), 4), 
                                                )
                    plt.show()                    
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        history = {'epochs': epochs,
                   'train_loss': history_train_losses, 
                   'train_acc': history_train_accuracies,
                   'valid_loss': history_test_losses,
                   'valid_acc': history_test_accuracies
                   }
        print(history_train_accuracies)
        return history
        
    def train(self, x_train, y_train, print_every=10, tqdm_=False):
       	"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
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
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1] / batch_size)
            else:
                n_batches = int(x_train.shape[1] / batch_size ) - 1

            idx = np.arange(x_train.shape[1])
            np.random.shuffle(idx)

            x_train = x_train[:, idx]
            y_train = y_train[:,idx]

            batches_x = [x_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            batches_y = [y_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]

            train_losses = []
            train_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases] 
            
            for batch_x, batch_y in zip(batches_x, batches_y):
                batch_y_pred, pre_activations, activations = self.forward(batch_x)
                deltas = self.update_deltas(pre_activations, batch_y, batch_y_pred)
                dW, db = self.backpropagate(deltas, pre_activations, activations)
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
                print('Epoch {} / {} | train loss: {} | train accuracy: {}'.format(
                    e, epochs, np.round(np.mean(train_losses), 3), np.round(np.mean(train_accuracies), 3)))

        history = {'epochs': epochs,
                   'train_loss': history_train_losses, 
                   'train_acc': history_train_accuracies
                   }
        return history
        
    def predict(self, a):
        '''
        Use the current state of the network to make predictions

        Parameters:
        ---
        a: input data, shape: (input_shape, batch_size)

        Returns:
        ---
        predictions: vector of output predictions
        '''
        #print(a)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation_sigmoid(z)
       
        #predictions = (a > 0.5).astype(int)
        return a
        
    def evaluate(self, X, y):
        """
		c'est la méthode qui va évaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
        y_pred_proba = self.predict(X)
        y_pred = np.argmax(y_pred_proba, axis = 0)
        y = np.argmax(y, axis = 0)
        show_metrics(y, y_pred)

def plot_history(history, show_valid = False):
    n = history['epochs']
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    n = 4000
    plt.plot(range(history['epochs'])[:n], history['train_loss'][:n], label='Perte en entraînement')
    if show_valid : plt.plot(range(history['epochs'])[:n], history['valid_loss'][:n], label='Perte en validation')
    #plt.title('Perte en entraînement et en validation')
    #plt.grid(1)
    plt.xlabel('Époques')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(history['epochs'])[:n], history['train_acc'][:n], label='Exactitude en entraînement')
    if show_valid : plt.plot(range(history['epochs'])[:n], history['valid_acc'][:n], label='Exactitude en validation')
    #plt.title('Exactitudes en entraînement et en validation')
    #plt.grid(1)
    plt.xlabel('Époques')
    plt.legend()
    plt.show()


def grid_search(train, train_labels, list_batch_size=[64], list_epochs=[1000], list_learning_rate=[0.5], k = 10, list_nb_neurones = range(2, 12), list_nb_hidden_layers = [1]):
    """
    Fonction qui permet de rechercher les meilleurs hyperparamètres
    """
    fold_indices = np.arange(train.shape[0])
    valid_indices = np.array_split(fold_indices, k)
   
	# initialisation d'un tableau qui contient les exactitudes pour chaque échantillon
    
    print(train.shape)
    max_mean_accuracy = None
    
    for learning_rate in list_learning_rate:
        for batch_size in list_batch_size :
            for epochs in list_epochs:

                error_valid = []
                error_train = []
                for nb_hidden_layers in list_nb_hidden_layers:
                    for nb_neurones in list_nb_neurones:
                        # 2-1) boucle de validation croisée
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
    else : 
        plt.plot(list_nb_hidden_layers, error_train, label ='Erreur moyenne en entraînement')
        plt.plot(list_nb_hidden_layers, error_valid, label ='Erreur moyenne en validation')
        plt.xlabel('Nombre de couches cachées')
    plt.legend()
    plt.show()
    return best_nb_neurones, best_hidden_layers, best_batch_size, best_epochs, best_learning_rate