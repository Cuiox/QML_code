"""
dataloader
img 2 numpy

resize  8
n_qubit 6

"""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
#from ml_model.pennylane_jax.classifier_jax import circuit

from templates.layers.naive_strong_entangler import StronglyEntanglingLayers
from ml_model.pennylane_jax.classifier_jax import ClassifierJax

from sklearn.utils import shuffle

import jax
from jax import vmap
from jax import numpy as jnp
from jax.config import config

import optax

config.update("jax_enable_x64", True)

num_qubits = 6
dev = qml.device('default.mixed', wires=num_qubits)

@qml.qnode(dev)
def circuit(x, weights, num_qubits, p=None):
    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=True)
    StronglyEntanglingLayers(weights=weights, wires=range(num_qubits), p=p)

    return qml.expval(qml.PauliZ(wires=0))

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, size=10):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 遍历文件夹，收集图片路径和标签
        for label in [0, 1]:
            label_dir = os.path.join(root_dir, f"{label}")
            count = 0
            for filename in os.listdir(label_dir):
                if count < size:
                    count = count + 1
                    self.image_paths.append(os.path.join(label_dir, filename))
                    self.labels.append(-1 if label == 0 else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        #image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path).convert("L")
        
        if self.transform:
            image = self.transform(image)

        image_array = numpy.array(image).flatten()
        
        return image_array, label

def collate_fn(batch):
    images = []
    labels = []
    
    for image, label in batch:
        images.append(image)
        labels.append(label)
    
    return numpy.array(images), numpy.array(labels)

class MnistClassifierJax(ClassifierJax):
    def __init__(self, num_qubits=6, num_layers=3, p=None, dev=None, weights=None, best_weights=None, 
                 opt=None, batch_size=10, lr=0.01, qml_circuit=circuit,
                 img_size=8, test_sub_size=10, train_sub_size=100):
        """Mnist Classifier

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
            img_size (int, optional): Resize 后的图片大小. Defaults to 8.
            test_sub_size (int, optional): test dataset 0 或 1 的训练样本数量. Defaults to 10.
            train_sub_size (int, optional): _description_. Defaults to 50.
        """
        super().__init__(num_qubits, num_layers, p, dev, weights, best_weights, opt, batch_size, lr, qml_circuit)
        self.img_size = img_size
        self.test_sub_size = test_sub_size
        self.train_sub_size = train_sub_size
        
        self.opt_state = None
        
    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        test_dataset = CustomDataset(root_dir=f'./data/mnist/size_{self.img_size}/test/', transform=transform, size=self.test_sub_size)
        train_dataset = CustomDataset(root_dir=f'./data/mnist/size_{self.img_size}/train/', transform=transform, size=self.train_sub_size)
        
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        
        return

    def train(self, epochs=10):
        
        def epochTrain(epoch):
            
            batch_losses = []
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                loss_value, gradient = jax.value_and_grad(self.cost)(self.weights, data, target, self.p)
                updates, self.opt_state = self.opt.update(gradient, self.opt_state, self.weights)
                self.weights = optax.apply_updates(self.weights, updates)
                
                batch_losses.append(loss_value)
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(self.train_dataloader)}]", end="\t")
                print(f"Loss: {loss_value:0.7f}")
            
            epoch_loss = np.mean(batch_losses)
            return epoch_loss
        
        def epochTest():
            
            batch_losses, test_acc = [], 0
            for batch_idx, (data, target) in enumerate(self.test_dataloader):
                outputs = self.output(data)
                #predictions = self.predict(data)
                predictions = jnp.sign(outputs)
                batch_acc = self.num_success(target, predictions)
                batch_loss = optax.l2_loss(outputs - target).mean()
                
                batch_losses.append(batch_loss)
                test_acc = test_acc + batch_acc
                
            test_loss = np.mean(batch_losses)
            return test_loss, test_acc
        
        # init weights
        self.init_weights()
        
        # opt
        self.opt = optax.adam(learning_rate=self.lr) if self.opt is None else self.opt
        self.opt_state = self.opt.init(self.weights)
        
        best_acc = 0
        train_losses, test_losses, test_accs = [], [], []
        for epoch in range(epochs):
            train_loss = epochTrain(epoch=epoch)
            test_loss, test_acc = epochTest()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            print(f"Epoch [{epoch+1}/{epochs}]\t train_loss: {train_loss:0.5f}\t test_loss: {test_loss:0.5f}\t test_acc: {test_acc}/{2*self.test_sub_size}")
            
            if self.best_test_acc < test_acc:
                self.best_test_acc = test_acc
                self.best_weights = self.weights
                

if __name__ == "__main__":
    lr = 1e-1
    num_layers = 3
    p = 0.01
    classifier = MnistClassifierJax(num_qubits=num_qubits, num_layers=num_layers, p=p, lr=lr)
    classifier.load_data()
    classifier.train(2)