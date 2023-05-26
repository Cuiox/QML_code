import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

from templates.layers.naive_strong_entangler import StronglyEntanglingLayers
from ml_model.pennylane_jax.classifier import Classifier

import jax
from jax import vmap
from jax import numpy as jnp
from jax.config import config

import optax

import random

config.update("jax_enable_x64", True)

num_qubits = 2
dev = qml.device('default.mixed', wires=num_qubits)

@qml.qnode(dev)
def circuit(x, weights, num_qubits, p=None):
    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=True)
    StronglyEntanglingLayers(weights=weights, wires=range(num_qubits), p=p)

    return qml.expval(qml.PauliZ(wires=0))

class ClassifierJax(Classifier):
    def __init__(self, num_qubits=2, num_layers=3, p=None, dev=None, weights=None, best_weights=None,
                 opt=None, batch_size=16, lr=1e-2, qml_circuit=circuit):
        super().__init__(num_qubits, num_layers, p, dev, weights, best_weights, opt, batch_size, lr, qml_circuit)
        
    def load_data(self):
        pass
    
    #def variational_classifier(self, weights, x, p=None):
    #    return self.qml_circuit(x, weights['circuit_weights'], num_qubits=self.num_qubits, p=p) + weights['bias']

    #def accuracy(self, labels, predictions):
    #    return super().accuracy(labels, predictions)
    
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
    
    def output(self, X):
        return vmap(self.variational_classifier, in_axes=(None, 0, None), out_axes=0)(self.weights, X, self.p)
    
    def test(self, X, Y, use_best=False):
        predictions = self.predict(X, use_best=use_best)
        return self.accuracy(Y, predictions)
    
    def init_seed(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def init_weights(self):
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        weight_shape = StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)
        self.weights = {
            'circuit_weights': 0.1 * jax.random.normal(shape=weight_shape, key=jax.random.PRNGKey(random.randint(0, 10)), dtype=jnp.float64),
            'bias':  jnp.array(0, dtype=jnp.float64)
        }

    def train(self):
        pass

if __name__ == "__main__":
    pass