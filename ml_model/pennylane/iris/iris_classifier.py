import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from ml_model.pennylane.classifier import circuit

from templates.layers.naive_strong_entangler import StronglyEntanglingLayers
from ml_model.pennylane.classifier import Classifier

from sklearn.utils import shuffle

num_qubits = 2
dev = qml.device('default.mixed', wires=num_qubits)

@qml.qnode(dev)
def circuit(x, weights, num_qubits, p=None):
    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=True)
    StronglyEntanglingLayers(weights=weights, wires=range(num_qubits), p=p)

    return qml.expval(qml.PauliZ(wires=0))

class IrisClassifier(Classifier):    
    def __init__(self, num_qubits=2, num_layers=3, p=None, dev=None, weights=None, best_weights=None, 
                 opt=None, batch_size=16, lr=0.01, qml_circuit=circuit):
        """_summary_

        Args:
            num_qubits (int, optional): _description_. Defaults to 2.
            num_layers (int, optional): _description_. Defaults to 3.
            p (_type_, optional): _description_. Defaults to None.
            dev (_type_, optional): _description_. Defaults to None.
            weights (_type_, optional): _description_. Defaults to None.
            best_weights (_type_, optional): _description_. Defaults to None.
            opt (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 16.
            lr (float, optional): _description_. Defaults to 0.01.
            qml_circuit (_type_, optional): _description_. Defaults to circuit.
        """
        super().__init__(num_qubits, num_layers, p, dev, weights, best_weights, opt, batch_size, lr, qml_circuit)
        
    def load_data(self, train_ratio=0.75):
        data = np.loadtxt("./data/iris/iris.txt")
        X = data[:, 0:4]
        Y = data[:, -1]
        
        X = np.array(X)
        Y = np.array(Y)
        
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
    
    def train(self, epoch):
        X_train, Y_train = self.X_train, self.Y_train
        X_test, Y_test = self.X_test, self.Y_test

        num_train = len(X_train)
        
        self.init_seed()
        self.init_weights()
        
        opt = NesterovMomentumOptimizer(stepsize=0.01, momentum=0.9)
        #opt = qml.GradientDescentOptimizer(stepsize=0.1)
        batch_size = 16 #self.batch_size
        for it in range(epoch):
            # Update the weights by one optimizer step
            batch_index = np.random.randint(0, num_train, (batch_size,))
            feats_train_batch = X_train[batch_index]
            Y_train_batch = Y_train[batch_index]
            
            print(f"bias = {self.bias}")
            
            new_list, prev_cost = opt.step_and_cost(self.cost, self.circuit_weights, self.bias,
                np.array(feats_train_batch, requires_grad=False), np.array(Y_train_batch, requires_grad=False), np.array(self.p, requires_grad=False))
            
            self.circuit_weights, self.bias = new_list[0], new_list[1]
            
            print(f"cost = {prev_cost}")
            
            #grads = opt.compute_grad(self.cost, self.weights)
            #print(f"{grads}")
            
            # Compute predictions on train and validation set
            # Compute accuracy on train and validation set
            acc_train = self.test(X_train, Y_train)
            acc_val = self.test(X_test, Y_test)

            print(
                "Iter: {:5d} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
                "".format(it + 1, acc_train, acc_val)
            )
        
        return
    
    def train_v2(self, epoch):
        """
        将 step_and_cost 分成 compute_grad + apply_grad
        """        
        X_train, Y_train = self.X_train, self.Y_train
        X_test, Y_test = self.X_test, self.Y_test

        num_train = len(X_train)
        
        self.init_seed()
        self.init_weights()
        
        opt = NesterovMomentumOptimizer(stepsize=0.01, momentum=0.9)
        #opt = qml.GradientDescentOptimizer(stepsize=0.1)
        batch_size = 16 #self.batch_size
        prev_cost = 0
        for it in range(epoch):
            # Update the weights by one optimizer step
            batch_index = np.random.randint(0, num_train, (batch_size,))
            feats_train_batch = X_train[batch_index]
            Y_train_batch = Y_train[batch_index]
            
            print(f"bias = {self.bias}")
        
            grad, prev_cost = opt.compute_grad(self.cost,  (self.circuit_weights, self.bias, np.array(feats_train_batch, requires_grad=False), np.array(Y_train_batch, requires_grad=False), np.array(self.p, requires_grad=False)), kwargs={})
            print(f"cost = {prev_cost}")
            #print(f"grad = {grad}")
            new_weights, new_bias = opt.apply_grad(grad, (self.circuit_weights, self.bias))
            #print(f"new weights = {new_weights}")
            
            self.circuit_weights, self.bias = new_weights, new_bias
            #print(f"new bias = {self.bias}")
            # Compute accuracy on train and validation set
            acc_train = self.test(X_train, Y_train)
            acc_val = self.test(X_test, Y_test)

            print(
                "Iter: {:5d} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
                "".format(it + 1, acc_train, acc_val)
            )
        
        return
        
if __name__ == "__main__":
    num_layers = 3
    p = [0.001, 0.001]
    classifier = IrisClassifier(num_qubits=num_qubits, num_layers=num_layers, p=p)
    classifier.load_data()
    classifier.train(60)