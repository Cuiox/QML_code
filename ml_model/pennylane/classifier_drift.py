import pennylane as qml
from pennylane import numpy as np
import numpy
from pennylane.optimize import NesterovMomentumOptimizer
from ml_model.pennylane.classifier_jax import circuit
from ml_model.pennylane.classifier_jax import circuit

from templates.layers.naive_strong_entangler import StronglyEntanglingLayers
from ml_model.pennylane.classifier import Classifier

from sklearn.utils import shuffle

import random

import optax

num_qubits = 2
dev = qml.device('default.mixed', wires=num_qubits)

@qml.qnode(dev)
def circuit(x, weights, num_qubits, p=None):
    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=True)
    StronglyEntanglingLayers(weights=weights, wires=range(num_qubits), p=p)

    return qml.expval(qml.PauliZ(wires=0))

class ClassifierNoiseDrift(Classifier):
    def __init__(self, num_qubits=2, num_layers=3, p=None, dev=None, weights=None, best_weights=None, 
                 opt=None, batch_size=16, lr=0.01, qml_circuit=circuit):
        super().__init__(num_qubits, num_layers, p, dev, weights, best_weights, opt, batch_size, lr, qml_circuit)
        
    def noise_mapping_classifier(self, params_weights, parmas_bias, x, p):
        """_summary_

        Args:
            params_weights (_type_): network param for weights
            parmas_bias (_type_): network param for bias
            x (_type_): input x
            p (np array | float): p

        Returns:
            _type_: _description_
        """
        if isinstance(p, float):
            p = np.full(self.num_qubits, p)
        weight_shape = StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits)
        
        weights = self.neural_network_mapping(params_weights, p).reshape(weight_shape)
        bias = self.neural_network_mapping(parmas_bias, p).reshape(())

        return self.variational_classifier(weights, bias, x, p)
    
    # Initialize all layers for a fully-connected neural network with sizes "sizes"
    def init_network_params(self, sizes, scale=1e-2):
        # A helper function to randomly initialize weights and biases for a dense neural network layer
        def random_layer_params(m, n, scale=1e-2):
            # w, b
            mean = 0  # Mean of the distribution
            stddev = 1  # Standard deviation of the distribution
            return scale * np.array(np.random.normal(loc=mean, scale=stddev, size=(n, m)), requires_grad=True), scale * np.array(np.random.normal(mean, stddev, (n,)), requires_grad=True)
        
        return [random_layer_params(m, n, scale) for m, n in zip(sizes[:-1], sizes[1:])]

    def relu(self, x):
        return max(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def neural_network_mapping(self, params, x):
        # per-example predictions
        activations = x
        for w, b in params[:-1]:
            outputs = np.dot(w, activations) + b
            activations = self.sigmoid(outputs)
    
        final_w, final_b = params[-1]
        logits = np.dot(final_w, activations) + final_b
        return logits #- logsumexp(logits)
    
    def cost(self, params_weights, parmas_bias, X, Y, p=None):
        outputs = [self.noise_mapping_classifier(params_weights, parmas_bias, x, p) for x in X]
        #return optax.l2_loss(outputs - Y).mean()
        return self.square_loss(Y, outputs)
    
    def init_weights(self):
        # TODO 隐藏层作为参数
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        weight_shape = StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)
        
        self.params_weights = self.init_network_params([num_qubits, 5, num_layers * num_qubits * 3], scale=0.01)
        self.parmas_bias = self.init_network_params([num_qubits, 3 ,1], scale=0.01)
    
    def predict(self, x, p):
        return np.sign(self.noise_mapping_classifier(self.params_weights, self.parmas_bias, x, p))
    
    def output(self, x, p):
        return self.noise_mapping_classifier(self.params_weights, self.parmas_bias, x, p)
    
    def batch_predict(self, X, p):
        predictions = [self.predict(x, p) for x in X]
        return predictions
    
    def batch_output(self, X, p):
        outputs = [self.output(x, p) for x in X]
        return outputs
    
    def test(self, X, Y, use_best=False, p=None):
        p = np.zeros(self.num_qubits) if p is None else p
        predictions = [self.predict(x, p) for x in X]
        return self.accuracy(Y, predictions)

    def train(self):
        # for testing
        self.init_seed()
        self.init_weights()
        
        print(f"num_layer = {self.num_layers}, num_qubit = {self.num_qubits}")
        print(f"weights 4 circuit:\n{self.params_weights}")
        print(f"weights 4 bias:\n{self.parmas_bias}")
        
        weight_shape = StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits)
        p = np.array([0.1, 0.1])
        q_weights = {
            'circuit_weights': self.neural_network_mapping(self.params_weights, p).reshape(weight_shape),
            'bias': self.neural_network_mapping(self.parmas_bias, p).reshape(())
        }
        print(f"weights circuit:\n{q_weights['circuit_weights']}")
        print(f"weights bias:\n{q_weights['bias']}")
        
if __name__ == "__main__":
    num_layers = 2
    classifier = ClassifierNoiseDrift(num_qubits=num_qubits, num_layers=num_layers)
    
    classifier.train()
    