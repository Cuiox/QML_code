import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
#from ml_model.pennylane.classifier_jax import circuit

from templates.layers.naive_strong_entangler import StronglyEntanglingLayers
from ml_model.pennylane_jax.classifier_jax import ClassifierJax

from sklearn.utils import shuffle

import jax
from jax import vmap
from jax import numpy as jnp
from jax.config import config

import random

import optax

config.update("jax_enable_x64", True)

num_qubits = 2
dev = qml.device('default.mixed', wires=num_qubits)

@qml.qnode(dev)
def circuit(x, weights, num_qubits, p=None):
    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=True)
    StronglyEntanglingLayers(weights=weights, wires=range(num_qubits), p=p)

    return qml.expval(qml.PauliZ(wires=0))

class ClassifierNoiseDriftJax(ClassifierJax):
    def __init__(self, num_qubits=2, num_layers=3, p=None, dev=None, weights=None, best_weights=None, 
                 opt=None, batch_size=16, lr=0.01, qml_circuit=circuit):
        super().__init__(num_qubits, num_layers, p, dev, weights, best_weights, opt, batch_size, lr, qml_circuit)
        
    def noise_mapping_classifier(self, weights, x, p):
        if isinstance(p, float):
            p = np.full(self.num_qubits, p)
        weight_shape = StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits)
        q_weights = {
            'circuit_weights': self.neural_network_mapping(weights['circuit_weights'], p).reshape(weight_shape),
            'bias': self.neural_network_mapping(weights['bias'], p).reshape(())
        }   # 此 q_weights 为量子电路的参数
        return self.variational_classifier(q_weights, x, p)
    
    # Initialize all layers for a fully-connected neural network with sizes "sizes"
    def init_network_params(self, sizes, key, scale=1e-2):
        # A helper function to randomly initialize weights and biases for a dense neural network layer
        def random_layer_params(m, n, key, scale=1e-2):
            w_key, b_key = jax.random.split(key)
            return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(b_key, (n,))
        
        keys = jax.random.split(key, len(sizes))
        return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    def relu(self, x):
        return jnp.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))

    def neural_network_mapping(self, params, x):
        # per-example predictions
        activations = x
        for w, b in params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = self.sigmoid(outputs)
    
        final_w, final_b = params[-1]
        logits = jnp.dot(final_w, activations) + final_b
        return logits #- logsumexp(logits)
    
    def cost(self, weights, X, Y, p=None):
        predictions = vmap(self.noise_mapping_classifier, in_axes=(None, 0, None), out_axes=0)(weights, X, p)
        return optax.l2_loss(predictions - Y).mean()
    
    def init_weights(self):
        # TODO 隐藏层作为参数
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        weight_shape = StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)
        
        self.weights = {
            'circuit_weights': self.init_network_params([num_qubits, 5, num_layers * num_qubits * 3], jax.random.PRNGKey(random.randint(0, 100)), scale=0.01),
            'bias': self.init_network_params([num_qubits, 3 ,1], jax.random.PRNGKey(random.randint(0, 100)), scale=0.01)
        } # self.weights 为神经网络参数
        
    def predict(self, X, use_best = False, p = None):
        def _predict(weights, x):
            return jnp.sign(self.noise_mapping_classifier(weights, x, p))
        
        X = jnp.array(X)
        if use_best:
            predictions = vmap(_predict, in_axes=(None, 0), out_axes=0)(self.best_weights, X)
        else:
            predictions = vmap(_predict, in_axes=(None, 0), out_axes=0)(self.weights, X)
            
        return predictions
    
    def output(self, X):
        return vmap(self.noise_mapping_classifier, in_axes=(None, 0, None), out_axes=0)(self.weights, X, self.p)
    
    def test(self, X, Y, use_best=False, p=None):
        p = np.zeros(self.num_qubits) if p is None else p
        predictions = self.predict(X, use_best=use_best, p=p)
        return self.accuracy(Y, predictions)
        
    def train(self):
        # for testing
        self.init_seed()
        self.init_weights()
        
        print(f"num_layer = {self.num_layers}, num_qubit = {self.num_qubits}")
        print(f"weights 4 circuit:\n{self.weights['circuit_weights']}")
        print(f"weights 4 bias:\n{self.weights['bias']}")
        
        weight_shape = StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits)
        p = np.array([0.1, 0.1])
        q_weights = {
            'circuit_weights': self.neural_network_mapping(self.weights['circuit_weights'], p).reshape(weight_shape),
            'bias': self.neural_network_mapping(self.weights['bias'], p).reshape(())
        }
        print(f"weights circuit:\n{q_weights['circuit_weights']}")
        print(f"weights bias:\n{q_weights['bias']}")
        
if __name__ == "__main__":
    num_layers = 2
    classifier = ClassifierNoiseDriftJax(num_qubits=num_qubits, num_layers=num_layers)
    
    classifier.train()
    