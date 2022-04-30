"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

from operator import indexOf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
from metrics import show_metrics
import math

# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones
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

class NeuralNet: #nom de la class à changer

	def __init__(self,size = [2, 3, 1], seed= 1, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
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
	
	def compute_deltas(self, pre_activations, y_true, y_pred):
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
		dW = []
		db = []
		deltas = [0] + deltas
		for l in range(1, len(self.size)):
			dW_l = np.dot(deltas[l], activations[l-1].transpose()) 
			db_l = deltas[l]
			dW.append(dW_l)
			db.append(np.expand_dims(db_l.mean(axis=1), 1))
		return dW, db
		
	def plot_decision_regions(self, X, y, iteration, train_loss, val_loss, train_acc, val_acc, res=0.01):
		"""
        Plots the decision boundary at each iteration (i.e. epoch) in order to inspect the performance
        of the model
        Parameters:
        ---
        X: the input data
        y: the labels
        iteration: the epoch number
        train_loss: value of the training loss
        val_loss: value of the validation loss
        train_acc: value of the training accuracy
        val_acc: value of the validation accuracy
        res: resolution of the plot
        Returns:
        ---
        None: this function plots the decision boundary
        """
		X, y = X.T, y.T 
		x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
		y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
		xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                            np.arange(y_min, y_max, res))
							
		Z = self.predict(np.c_[xx.ravel(), yy.ravel()].T)
		Z = Z.reshape(xx.shape)
		plt.contourf(xx, yy, Z, alpha=0.5)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1),  alpha=0.2)
		message = 'iteration: {} | train loss: {} | val loss: {} | train acc: {} | val acc: {}'.format(iteration,
                                                                                                     train_loss, 
                                                                                                     val_loss, 
                                                                                                     train_acc, 
                                                                                                     val_acc)
		plt.title(message)


	def train(self, train, train_labels, batch_size, epochs, learning_rate, train_split=0.8, print_every = 10, plot_every = None, tqdm_=False): #vous pouvez rajouter d'autres attributs au besoin
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
		history_test_losses = []
		history_test_accuracies = []
		
		nrow = len(train_labels)
		indices = np.arange(nrow)
    
		split = math.floor(train_split * nrow)
		train_idx,    test_idx    = indices[:split],   indices[split:]
		x_train,        x_test        = train[train_idx, :],   train[test_idx, :]
		y_train, y_test = train_labels[train_idx], train_labels[test_idx]
		print(x_train.shape[0])
		if tqdm_:
			epoch_iterator = tqdm(range(epochs))
		else:
			epoch_iterator = range(epochs)
			
		for e in epoch_iterator:
			if x_train.shape[0] % batch_size == 0:
				n_batches = int(x_train.shape[0] / batch_size)
			else:
				n_batches = int(x_train.shape[0] / batch_size ) - 1
			
			#idx = np.arange(x_train.shape[0])
			#print(idx)
			#idx = np.random.shuffle(idx)
			#x_train= x_train[idx, :]
			#y_train = y_train[idx]
			#print(x_train)
			batches_x = [x_train[batch_size*i:batch_size*(i+1), :] for i in range(0, n_batches)]
			batches_y = [y_train[batch_size*i:batch_size*(i+1), :] for i in range(0, n_batches)]
			print(batches_x)
			train_losses = []
			train_accuracies = []
			
			test_losses = []
			test_accuracies = []
			
			dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
			db_per_epoch = [np.zeros(b.shape) for b in self.biases] 
			
			for batch_x, batch_y in zip(batches_x, batches_y):
				batch_y_pred, pre_activations, activations = self.forward(batch_x)
				deltas = self.compute_deltas(pre_activations, batch_y, batch_y_pred)
				dW, db = self.backpropagate(deltas, pre_activations, activations)
				for i, (dw_i, db_i) in enumerate(zip(dW, db)):
					dw_per_epoch[i] += dw_i / batch_size
					db_per_epoch[i] += db_i / batch_size
					
				batch_y_train_pred = [self.predict(x) for x in batch_x]
				
				train_loss = cost_function(batch_y, batch_y_train_pred)
				train_losses.append(train_loss)
				print(batch_y)
				print(batch_y_train_pred)
				train_accuracy = np.mean(batch_y == batch_y_train_pred)
				train_accuracies.append(train_accuracy)
				
				batch_y_test_pred = [self.predict(x) for x in x_test]
				
				test_loss = cost_function(y_test, batch_y_test_pred)
				test_losses.append(test_loss)
				test_accuracy = np.mean(y_test == batch_y_test_pred)
				test_accuracies.append(test_accuracy)
			
			# weight update
			for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
				self.weights[i] = self.weights[i] - learning_rate * dw_epoch
				self.biases[i] = self.biases[i] - learning_rate * db_epoch
				
			history_train_losses.append(np.mean(train_losses))
			print(train_losses)
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
					
		self.plot_decision_regions(train, train_labels, e, 
                                    np.round(np.mean(train_losses), 4), 
                                    np.round(np.mean(test_losses), 4),
                                    np.round(np.mean(train_accuracies), 4), 
                                    np.round(np.mean(test_accuracies), 4), 
                                    )
		
		history = {'epochs': epochs,
                   'train_loss': history_train_losses, 
                   'train_acc': history_train_accuracies,
                   'test_loss': history_test_losses,
                   'test_acc': history_test_accuracies
                   }
		return history
        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, x) + b
			x = activation(z)
		predictions = (x > 0.5).astype(int)
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
		y_pred = self.predict(X) 
		show_metrics(y, y_pred)
	
	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.