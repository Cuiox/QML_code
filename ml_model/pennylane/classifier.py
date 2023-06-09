"""
https://pennylane.ai/blog/2022/06/how-to-choose-your-optimizer/
"""
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

from templates.layers.naive_strong_entangler import StronglyEntanglingLayers

num_qubits = 2
dev = qml.device('default.mixed', wires=num_qubits)

@qml.qnode(dev)
def circuit(x, weights, num_qubits, p=None):
    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=True)
    StronglyEntanglingLayers(weights=weights, wires=range(num_qubits), p=p)

    return qml.expval(qml.PauliZ(wires=0))

class Classifier(object):
    def __init__(self, num_qubits=2, num_layers=3, p=None, dev=None, weights=None, best_weights=None,
                 opt=None, batch_size=16, lr=1e-2, qml_circuit=circuit):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.p = p
        self.dev = dev
        self.weights = weights
        self.best_weights = best_weights
        self.opt = opt
        self.batch_size = batch_size
        self.lr = lr
        self.qml_circuit = qml_circuit
        self.best_test_acc = 0
        
    def load_data(self):
        pass
    
    def variational_classifier(self, weights, bias, x, p=None):
        #print(f"In here: {weights}")
        return self.qml_circuit(x, weights, num_qubits=self.num_qubits, p=p) + bias

    def accuracy(self, labels, predictions):
        num_success = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                num_success = num_success + 1
        accuracy = num_success / len(labels)
        return accuracy
    
    def num_success(self, labels, predictions):
        num_success = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                num_success = num_success + 1
        
        return num_success
    
    def init_seed(self, seed=42):
        np.random.seed(seed)
        # random.seed(seed)
        
    def init_weights(self):
        num_qubits = self.num_qubits
        num_layers = self.num_layers
        weight_shape = StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)
        self.weights = {
            'circuit_weights': 0.01 * np.random.randn(*weight_shape, requires_grad=True),
            'bias':  np.array(0.0, requires_grad=True)
        } # pennylane 不能对字典进行 grad
        
        self.circuit_weights = 0.01 * np.random.randn(*weight_shape, requires_grad=True)
        self.bias = np.array(0.0, requires_grad=True)
        

    def square_loss(self, labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss
    
    def cost(self, weights, bias, features, labels, p=None):
        outputs = [self.variational_classifier(weights, bias, f, p) for f in features]
        return self.square_loss(labels, outputs)
    
    def predict(self, x):
        return np.sign(self.variational_classifier(self.circuit_weights, self.bias, x, self.p))
    
    def test(self, X, Y):
        predictions = [self.predict(x) for x in X]
        return self.accuracy(Y, predictions)
    
    def train(self):
        pass

if __name__ == "__main__":
    pass