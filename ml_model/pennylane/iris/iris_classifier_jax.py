import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

from templates.layers.naive_strong_entangler import StronglyEntanglingLayers
from ml_model.pennylane.classifier_jax import ClassifierJax

from sklearn.utils import shuffle

import jax
from jax import vmap
from jax import numpy as jnp
from jax.config import config

import optax

config.update("jax_enable_x64", True)

num_qubits = 2
dev = qml.device('default.mixed', wires=num_qubits)

@qml.qnode(dev)
def circuit(x, weights, num_qubits, p=None):
    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=True)
    StronglyEntanglingLayers(weights=weights, wires=range(num_qubits), p=p)

    return qml.expval(qml.PauliZ(wires=0))

class IrisClassifierJax(ClassifierJax):
    def __init__(self, num_qubits=2, num_layers=3, p=None, dev=None, weights=None, best_weights=None,
                 opt=None, batch_size=16, lr=1e-2, qml_circuit=circuit):
        super().__init__(num_qubits, num_layers, p, dev, weights, best_weights, opt, batch_size, lr, qml_circuit)
        
    def load_data(self, train_ratio=0.75):
        data = np.loadtxt("./data/iris/iris.txt")
        X = data[:, 0:4]
        Y = data[:, -1]
        
        X = jnp.array(X)
        Y = jnp.array(Y)
        
        X, Y = shuffle(X, Y, random_state=42)

        self.X = X
        self.Y = Y

        num_data = len(Y)
        num_train = int(train_ratio * num_data)

        self.X_train = X[:num_train]
        self.Y_train = Y[:num_train]

        self.X_test = X[num_train:]
        self.Y_test = Y[num_train:]

        return
        
    def variational_classifier(self, weights, x, p=None):
        return self.qml_circuit(x, weights['circuit_weights'], num_qubits=self.num_qubits, p=p) + weights['bias']

    def accuracy(self, labels, predictions):
        return super().accuracy(labels, predictions)
    
    def cost(self, weights, X, Y, p=None):
        predictions = vmap(self.variational_classifier, in_axes=(None, 0, None), out_axes=0)(weights, X, p)
        return optax.l2_loss(predictions - Y).mean()
        
    def predict(self, X, use_best=False):
        def _predict(weights, x):
            return jnp.sign(self.variational_classifier(weights, x, self.p))
        
        X = jnp.array(X)
        if use_best:
            predictions = vmap(_predict, in_axes=(None, 0), out_axes=0)(self.best_weights, X)
        else:
            predictions = vmap(_predict, in_axes=(None, 0), out_axes=0)(self.weights, X)
            
        return predictions
    
    def test(self, X, Y, use_best=False):
        predictions = self.predict(X, use_best=use_best)
        return self.accuracy(Y, predictions)
    
    def train(self, epoch):
        X_train, Y_train = self.X_train, self.Y_train
        X_test, Y_test = self.X_test, self.Y_test
        
        num_train = len(X_train)
        
        self.init_weights()
        
        # opt
        opt = optax.adam(learning_rate=self.lr) if self.opt is None else self.opt
        opt_state = opt.init(self.weights)
        
        acc_train_history, acc_test_history = [], []
        acc_train = self.test(X_train, Y_train)
        acc_val = self.test(X_test, Y_test)
        print(f"Inital:   0 | Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}")
        
        for it in range(epoch):
            # Update the weights by one optimizer step
            batch_index = np.random.randint(0, num_train, (self.batch_size,))
            X_train_batch = X_train[batch_index]
            Y_train_batch = Y_train[batch_index]
            
            loss_value, gradient = jax.value_and_grad(self.cost)(self.weights, X_train_batch, Y_train_batch, self.p)
            updates, opt_state = opt.update(gradient, opt_state, self.weights)
            self.weights = optax.apply_updates(self.weights, updates)            
            
            acc_train = self.test(X_train, Y_train)
            acc_val = self.test(X_test, Y_test)

            acc_train_history.append(acc_train)
            acc_test_history.append(acc_val)

            if self.best_test_acc < acc_val:
                self.best_test_acc = acc_val
                self.best_weights = self.weights

            print(f"Iter: {(it + 1):5d} | Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}")
         
        

if __name__ == "__main__":
    lr = 1e-1
    num_layers = 3
    p = [0.1, 0.1]
    classifier = IrisClassifierJax(num_qubits=num_qubits, num_layers=num_layers, p=p, lr=lr)
    classifier.load_data()
    classifier.train(60)