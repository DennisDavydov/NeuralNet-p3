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
        #print Y
        activation = X
        activations = [X]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+np.tile(b, len(mini_batch))
            zs.append(z)
            activation = sigmoid(z)
            
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], Y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)