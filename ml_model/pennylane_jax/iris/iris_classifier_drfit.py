import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from ml_model.pennylane.classifier_drift import circuit
from ml_model.pennylane.classifier_jax import circuit

from templates.layers.naive_strong_entangler import StronglyEntanglingLayers
from ml_model.pennylane.classifier_drift import ClassifierNoiseDriftJax

from sklearn.utils import shuffle

import jax
from jax import vmap
from jax import numpy as jnp
from jax.config import config

import optax

from noise_channel.create_noise_history import create_history_fake

config.update("jax_enable_x64", True)

num_qubits = 2
dev = qml.device('default.mixed', wires=num_qubits)

@qml.qnode(dev)
def circuit(x, weights, num_qubits, p=None):
    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=True)
    StronglyEntanglingLayers(weights=weights, wires=range(num_qubits), p=p)

    return qml.expval(qml.PauliZ(wires=0))

class IrisClassifierNoiseDriftJax(ClassifierNoiseDriftJax):
    def __init__(self, num_qubits=2, num_layers=3, p=None, dev=None, weights=None, best_weights=None, 
                 opt=None, batch_size=16, lr=0.01, qml_circuit=circuit):
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
    
    def train(self, epoch, p_history):
        X_train, Y_train = self.X_train, self.Y_train
        X_test, Y_test = self.X_test, self.Y_test
        
        num_train = len(X_train)
        
        self.init_weights()
        
        # opt
        opt = optax.adam(learning_rate=self.lr) if self.opt is None else self.opt
        opt_state = opt.init(self.weights)
        
        for index, p in enumerate(p_history):
            print(f'Now in noise[{index}]: {p}')
            acc_train = self.test(X_train, Y_train, p=p)
            acc_val = self.test(X_test, Y_test, p=p)
            print(f"Inital:   0 | Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}")
        
            for it in range(epoch):
                # Update the weights by one optimizer step
                batch_index = np.random.randint(0, num_train, (self.batch_size,))
                X_train_batch = X_train[batch_index]
                Y_train_batch = Y_train[batch_index]

                loss_value, gradient = jax.value_and_grad(self.cost)(self.weights, X_train_batch, Y_train_batch, p)
                updates, opt_state = opt.update(gradient, opt_state, self.weights)
                self.weights = optax.apply_updates(self.weights, updates)            

                acc_train = self.test(X_train, Y_train)
                acc_val = self.test(X_test, Y_test)

                if self.best_test_acc < acc_val:
                    self.best_test_acc = acc_val
                    self.best_weights = self.weights

                print(f"Iter: {(it + 1):5d} | Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}")
            self.weights = self.best_weights
            
        return
    
if __name__ == "__main__":
    start, end, step = 0.0, 0.2, 0.01

    def create_random_p(n, L, start, end):
        p = np.random.uniform(start, end, size=(n, L))
        p[0] = np.ones((L,)) * ((end-start) * 0.5)
        #p = jnp.array(p)        
        return p
    # history= create_history_fake(num_qubits = 2, num_history=10, depolarizing = {'avg_1q': 0.1, 'scale_1q': 0.3})
    #history= create_history_fake(num_qubits=2, num_history=10, depolarizing = {'range': [start, end]})
    ## 随机构造10个noiseinfo, 为depolarizing类型, start, end = 0, 1
    #history[0] =  {
    #    'depolarizing': {
    #        'error_1q': jnp.ones((2,))*((end - start)*0.5),
    #    }
    #}
    p = create_random_p(n=10, L=2, start=0.0, end=0.1)
    print(p)


    classifier = IrisClassifierNoiseDriftJax()
    classifier.load_data()
    classifier.train(epoch=10, p_history=p)