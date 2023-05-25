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
    
    def variational_classifier(self, weights, x, p=None):
        return self.qml_circuit(x, weights['circuit_weights'], num_qubits=self.num_qubits, p=p) + weights['bias']

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
    
    def square_loss(self, labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss
    
    def cost(self, weights, features, labels, p=None):
        predictions = [self.variational_classifier(weights, f, p) for f in features]
        return self.square_loss(labels, predictions)
    
    def predict(self, x):
        return np.sign(self.variational_classifier(self.weights, x, self.p))
    
    def train(self):
        pass

if __name__ == "__main__":
    pass