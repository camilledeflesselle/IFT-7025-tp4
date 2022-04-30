# -*- coding: utf-8 -*-
# Author: Ahmed BESBES 
# <ahmed.besbes@hotmail.com>
#

# matplotlib for plotting
import math
from matplotlib import pyplot as plt

# numpy for vector and matrix manipulations
import numpy as np

from metrics import show_metrics

# tqdm is progress-bar. make sure it's installed: pip install tqdm
from tqdm import tqdm


def activation(z, derivative=False):
    """
    Sigmoid activation function:
    It handles two modes: normal and derivative mode.
    Applies a pointwize operation on vectors
    
    Parameters:
    ---
    z: pre-activation vector at layer l
        shape (n[l], batch_size)

    Returns: 
    pontwize activation on each element of the input z
    """
    if derivative:
        return activation(z) * (1 - activation(z))
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



class NeuralNetwork(object):     
    '''
    This is a custom neural netwok package built from scratch with numpy. 
    It allows training using SGD, inference and live plotting of the decision boundary.
    This code is not optimized and should not be used with real-world examples.
    It's written for educational purposes only.

    The Neural Network as well as its parameters and training method and procedure will 
    reside in this class.

    Parameters
    ---
    size: list of number of neurons per layer

    Examples
    ---
    >>> import NeuralNetwork
    >>> nn = NeuralNetword([2, 3, 4, 1])
    
    This means :
    1 input layer with 2 neurons
    1 hidden layer with 3 neurons
    1 hidden layer with 4 neurons
    1 output layer with 1 neuron
    
    '''

    def __init__(self, size, seed=42):
        '''
        Instantiate the weights and biases of the network
        weights and biases are attributes of the NeuralNetwork class
        They are updated during the training
        '''
        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]

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
            a  = activation(z)
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
        delta_L = cost_function_prime(y_true, y_pred) * activation(pre_activations[-1], derivative=True)
        deltas = [0] * (len(self.size) - 1)
        deltas[-1] = delta_L
        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]) * activation(pre_activations[l], derivative=True) 
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

    def train(self, X, y, batch_size, epochs, learning_rate, train_split=0.8, print_every=10, tqdm_=False, plot_every=None):
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
            print(idx)
            x_train = x_train[:, idx]
            y_train = y_train[:,idx]
			#print(x_train)
            """
            x_train, y_train = shuffle(x_train.T, y_train.T)
           
            x_train, y_train = x_train.T, y_train.T
            """
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
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        predictions = (a > 0.5).astype(int)
        return predictions
        
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
        y_pred = self.predict(X)[0]
        show_metrics(y, y_pred)

def plot_history(history):
    n = history['epochs']
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    n = 4000
    plt.plot(range(history['epochs'])[:n], history['train_loss'][:n], label='Perte en entraînement')
    plt.plot(range(history['epochs'])[:n], history['valid_loss'][:n], label='Perte en validation')
    plt.title('Perte en entraînement et en validation')
    #plt.grid(1)
    plt.xlabel('Époques')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(history['epochs'])[:n], history['train_acc'][:n], label='Exactitude en entraînement')
    plt.plot(range(history['epochs'])[:n], history['valid_acc'][:n], label='Exactitude en validation')
    plt.title('Exactitudes en entraînement et en validation')
    #plt.grid(1)
    plt.xlabel('Époques')
    plt.legend()
    plt.show()