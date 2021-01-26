import numpy as np
import random
import pickle
import os

class Network(object):
    def __init__(self, sizes = None, weights = None, biases = None, cross_entropy = True):
        self.cross_entropy = cross_entropy
        if sizes:
            
            self.num_layers = len(sizes)
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = weights
            self.biases = biases
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
                
        return a
    
    def backprop(self, mini_batch):
        nabla_b = [np.tile(np.zeros(b.shape), len(mini_batch)) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        X_array=[]
        Y_array=[]
        for x, y in mini_batch:
            X_array.append(x)
            Y_array.append(y)
        
        X = np.concatenate(X_array, axis = 1) 
        Y = np.concatenate(Y_array, axis = 1)
        activation = X
        
        activations = [X]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+np.repeat(b, len(mini_batch), 1)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #input()
        #quit()
        delta = self.cost_derivative(activations[-1], Y, zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        print('Starting training...')
        test_data = list(test_data)
        training_data = list(training_data)
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            i = 0
            for batch in mini_batches:
                i+=1
                
                nabla_b, nabla_w = self.backprop(batch)
                #update weights and biases 
                nabla_b = [np.sum(b, 1) for b in nabla_b]
                self.weights = [w-(eta/mini_batch_size)*nw for nw,w in zip(nabla_w, self.weights)]
                self.biases = [b-(eta/mini_batch_size)*np.expand_dims(nb, 1) for nb,b in zip(nabla_b, self.biases)]
            if test_data:
                print("Epoch {0}: {1} / {2}".format(e+1, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(e))
        
        filepath = os.path.dirname(__file__)+'\w_b'
        with open(filepath, 'wb') as file:
            pickle.dump((self.weights, self.biases), file)  
                
    def cost_derivative(self, output_activations, y, zs):
        if self.cross_entropy:
            return (output_activations - y)
        else:
            return(output_activations - y) * sigmoid_prime(zs)
        
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x==y) for x, y in test_results)

def sigmoid(z):
        a = 1/(1 + np.exp(-z))
        return a
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))