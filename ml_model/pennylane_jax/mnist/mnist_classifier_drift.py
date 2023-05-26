"""
https://docs.ray.io/en/latest/ray-core/examples/plot_parameter_server.html

ray 
同步更新
    求梯度的平均
异步更新
    不同的异步更新算法
    
"""

import ray

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch


import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

from templates.layers.naive_strong_entangler import StronglyEntanglingLayers
from ml_model.pennylane_jax.mnist.mnist_classifier_jax import MnistClassifierJax
from ml_model.pennylane_jax.classifier_drift import ClassifierNoiseDriftJax

from sklearn.utils import shuffle

import jax
from jax import vmap
from jax import numpy as jnp
from jax.config import config

from jax import grad, jit

import optax

ray.init(ignore_reinit_error=True)

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
        """CustomDataset

        Args:
            root_dir (_type_): 图片数据的根路径
                ./root_dir/0/ ./root_dir/1/
            transform (_type_, optional): _description_. Defaults to None.
            size (int, optional): 每种数字的size. Defaults to 10.
        """
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
                else:
                    break

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        #image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path).convert("L") # 灰度图
        
        if self.transform:
            image = self.transform(image)

        image_array = numpy.array(image).flatten() # 转为 np.array 并 flatten
        
        return image_array, label

def collate_fn(batch):
    # 转为 np.array
    images = []
    labels = []
    
    for image, label in batch:
        images.append(image)
        labels.append(label)
    
    return numpy.array(images), numpy.array(labels)

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0, seed=42, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed
        self.collate_fn = collate_fn
        self.current_idx = 0
        self.indices = list(range(len(dataset)))

        if shuffle:
            self.shuffle_indices()

    def __iter__(self):
        return self

    def get_batch(self, indices):
        # 在 get_batch 时已经转为 np.array 了, 故不再需要 collate_fn
        batch = [self.dataset[idx] for idx in indices]

        images, labels = zip(*batch)
        images = np.stack(images)
        labels = np.array(labels)

        return images, labels
    
    def __next__(self):
        if self.current_idx >= len(self.dataset):
            self.current_idx = 0 # 下一个 epoch 从 0 开始
            raise StopIteration

        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch = self.get_batch(batch_indices)        
        images, labels = batch
                
        if self.num_workers > 0: # turn to mini-batch
            num_samples = len(labels)
            mini_batch_size = num_samples // self.num_workers
            if mini_batch_size == 0: # num_samples < self.num_workers 的情况
                mini_batch_size = 1

            mini_batches = []
            for i in range(self.num_workers):
                start_idx = i * mini_batch_size
                end_idx = (i + 1) * mini_batch_size
                if end_idx > num_samples:
                    end_idx = num_samples
                #print(f"[worker {i}]: start_idx = {start_idx}, end_idx = {end_idx}")
                mini_batch = images[start_idx:end_idx], labels[start_idx:end_idx]
                mini_batches.append(mini_batch)

            batch = mini_batches
            
        # Increment the current index
        self.current_idx += self.batch_size
        
        return batch
    
    def shuffle_indices(self):
        torch.manual_seed(self.seed)  # Set a seed for reproducibility
        self.indices = torch.randperm(len(self.dataset), out=torch.tensor(self.indices))

    def shuffle_batch(self, batch):
        for mini_batch in batch:
            torch.manual_seed(0)  # Set a seed for reproducibility
            torch.randperm(len(mini_batch), out=torch.tensor(mini_batch))
                
def sum_data(data_list):
    {'bias': [(np.array([[2.2250463e-06, 2.2250463e-06, 2.2250463e-06, 2.2250463e-06, 2.2250463e-06, 2.2250463e-06],
                         [2.1467455e-05, 2.1467455e-05, 2.1467455e-05, 2.1467455e-05, 2.1467455e-05, 2.1467455e-05],
                         [2.4220581e-05, 2.4220581e-05, 2.4220581e-05, 2.4220581e-05, 2.4220581e-05, 2.4220581e-05]]), 
               np.array([0.0002225 , 0.00214675, 0.00242206])), # 第一层的 w, b
           
              (np.array([[-0.5139741, -0.5180101, -0.5182382]]), np.array([-1.0336919]))
              ], # 第二层的 w, b
     
    } # 每个 data 的结构
    result = {}  # 存储求和结果的字典
    
    for data in data_list:
        for key, value in data.items(): # value is a list
            if key not in result:
                # 如果结果字典中不存在该键，则直接将键值对添加到结果字典中
                result[key] = value
            else:
                # 如果结果字典中已存在该键，则对应的数组进行相加操作(0对应w, 1对应b)
                result[key] = [(result[key][i][0] + value[i][0], result[key][i][1] + value[i][1]) for i in range(len(value))]
    
    return result

class ClassifierModel(ClassifierNoiseDriftJax):
    def __init__(self, num_qubits=6, num_layers=3, p=None, dev=None, weights=None, best_weights=None, 
                 opt=None, batch_size=16, lr=0.01, qml_circuit=circuit,
                 img_size=8, test_sub_size=10, train_sub_size=100, num_workers=10):
        """每个 DataWorker 会初始化一个 Model

        Args:
            num_qubits (int, optional): _description_. Defaults to 6.
            num_layers (int, optional): _description_. Defaults to 3.
            p (_type_, optional): _description_. Defaults to None.
            dev (_type_, optional): _description_. Defaults to None.
            weights (_type_, optional): _description_. Defaults to None.
            best_weights (_type_, optional): _description_. Defaults to None.
            opt (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 10.
            lr (float, optional): _description_. Defaults to 0.01.
            qml_circuit (_type_, optional): _description_. Defaults to circuit.
            img_size (int, optional): _description_. Defaults to 8.
            test_sub_size (int, optional): _description_. Defaults to 10.
            train_sub_size (int, optional): _description_. Defaults to 100.
        """
        super().__init__(num_qubits, num_layers, p, dev, weights, best_weights, opt, batch_size, lr, qml_circuit)
        self.img_size = img_size
        self.test_sub_size = test_sub_size
        self.train_sub_size = train_sub_size
        self.num_workers = num_workers
        
        self.opt_state = None
        
    def set_weights(self, weights):
        self.weights = weights

class MnistClassifierNoiseDriftJax(ClassifierNoiseDriftJax):
    def __init__(self, num_qubits=2, num_layers=3, p=None, dev=None, weights=None, best_weights=None, 
                 opt=None, batch_size=16, lr=0.01, qml_circuit=circuit,
                 img_size=8, test_sub_size=10, train_sub_size=100, num_workers=10):
        """继承自 classifier_drift.py, 并定制化 load_data 和 train

        Args:
            num_qubits (int, optional): _description_. Defaults to 6.
            num_layers (int, optional): _description_. Defaults to 3.
            p (_type_, optional): _description_. Defaults to None.
            dev (_type_, optional): _description_. Defaults to None.
            weights (_type_, optional): _description_. Defaults to None.
            best_weights (_type_, optional): _description_. Defaults to None.
            opt (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 10.
            lr (float, optional): _description_. Defaults to 0.01.
            qml_circuit (_type_, optional): _description_. Defaults to circuit.
            img_size (int, optional): _description_. Defaults to 8.
            test_sub_size (int, optional): _description_. Defaults to 10.
            train_sub_size (int, optional): _description_. Defaults to 100.
        """
        super().__init__(num_qubits, num_layers, p, dev, weights, best_weights, opt, batch_size, lr, qml_circuit)
        self.img_size = img_size
        self.test_sub_size = test_sub_size
        self.train_sub_size = train_sub_size
        self.num_workers = num_workers
        
        self.opt_state = None
        
    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        test_dataset = CustomDataset(root_dir=f'./data/mnist/size_{self.img_size}/test/', transform=transform, size=self.test_sub_size)
        train_dataset = CustomDataset(root_dir=f'./data/mnist/size_{self.img_size}/train/', transform=transform, size=self.train_sub_size)
        
        self.test_dataloader_4_eval = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        self.test_dataloader = CustomDataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)
        self.train_dataloader = CustomDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers)
        
        return  
    
    def train(self, epochs=10):
        def epochTest(dataloader, weights):
            print(f"start test")
            batch_losses, test_acc = [], 0
            for batch_idx, (data, target) in enumerate(dataloader):
                print(f"[{batch_idx+1}/{len(dataloader)}]")
                outputs = vmap(self.noise_mapping_classifier, in_axes=(None, 0, None), out_axes=0)(weights, data, self.p)
                predictions = jnp.sign(outputs)
                batch_acc = self.num_success(target, predictions)
                batch_loss = optax.l2_loss(outputs - target).mean()
                
                batch_losses.append(batch_loss)
                test_acc = test_acc + batch_acc
                
            test_loss = np.mean(batch_losses)
            return test_loss, test_acc
        
        self.init_weights() # in classifier_drift.py

        ps = ParameterServer.remote(initial_params=self.weights, lr=1e-2)
        workers = [DataWorker.remote(id=i+1, p=0.01) for i in range(self.num_workers)]
        
        # train
        current_weights = ps.get_params.remote()
        for epoch in range(epochs):
            test_loss, test_acc = epochTest(self.test_dataloader_4_eval, ray.get(current_weights))
            print(f"Epoch [{epoch}/{epochs}]\t test_loss: {test_loss:0.5f}\t test_acc: {test_acc}/{2*self.test_sub_size}")
            for batch_idx, batch in enumerate(self.train_dataloader):
                print(f"[{epoch+1}/{epochs}] | [{batch_idx+1}] ")
                #mini_batches = [batch[i] for i in range(self.num_workers)]
                gradients = [workers[i].compute_gradients.remote(current_weights, batch[i]) for i in range(self.num_workers)]
                gradients_value = ray.get(gradients) # 同步
                #for i in range(self.num_workers):
                #    print(f"[{epoch}]/[{batch_idx}]/[{i}]: {gradients_value[i]}")
                summed_gradients = sum_data(gradients_value)
                #print(f"{summed_gradients}")
                current_weights = ps.apply_gradients.remote(summed_gradients)
            
        test_loss, test_acc = epochTest(self.test_dataloader_4_eval, ray.get(current_weights))
        print(f"Final Test | test_loss: {test_loss:0.5f}\t test_acc: {test_acc}/{2*self.test_sub_size}")
    

@ray.remote
class ParameterServer(object):
    def __init__(self, initial_params=None, lr=1e-2):
        """ParameterServer

        Args:
            initial_params (_type_, optional): _description_. Defaults to None.
            lr (_type_, optional): _description_. Defaults to 1e-2.
        """
        self.params = initial_params
        self.optimizer = optax.adam(learning_rate=lr)
        self.opt_state = self.optimizer.init(self.params)

    def apply_gradients(self, gradients):
        updates, self.opt_state = self.optimizer.update(gradients, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        return self.params

    def get_params(self):
        return self.params

@ray.remote
class DataWorker(object):
    def __init__(self, id=0, p=None):
        """DataWorker

        Args:
            id (int, optional): _description_. Defaults to 0.
            p (_type_, optional): p for the model. Defaults to None.
        """
        self.id = id
        self.p = p
        self.model = ClassifierModel()

    def compute_gradients(self, weights, mini_batch):
        #self.model.set_weights(weights)
        data, target = mini_batch
        loss_value, gradient = jax.value_and_grad(self.model.cost)(weights, data, target, self.p)

        return gradient
        
    def get_id(self):
        return self.id

def test():
    num_qubits = 6
    num_layers = 3
    p = 0.01
    lr = 1e-1
    num_workers = 10
    test_sub_size = 20
    train_sub_size = 100
    batch_size = 20
    classifier = MnistClassifierNoiseDriftJax(num_qubits=num_qubits, num_layers=num_layers, p=p, lr=lr, batch_size=batch_size, 
                                              num_workers=num_workers, test_sub_size=test_sub_size, train_sub_size=train_sub_size)
    classifier.load_data()
    classifier.train(epochs=20)
    
    ray.shutdown() # Clean up Ray resources and processes before the next example.

if __name__ == "__main__":
    #main()
    test()
    
"""
{'bias': [(Array([[2.2250463e-06, 2.2250463e-06, 2.2250463e-06, 2.2250463e-06, 2.2250463e-06, 2.2250463e-06],
                  [2.1467455e-05, 2.1467455e-05, 2.1467455e-05, 2.1467455e-05, 2.1467455e-05, 2.1467455e-05],
                  [2.4220581e-05, 2.4220581e-05, 2.4220581e-05, 2.4220581e-05, 2.4220581e-05, 2.4220581e-05]]
                  , dtype=float32), 
           Array([0.0002225 , 0.00214675, 0.00242206], dtype=float32)), 
           
          (Array([[-0.5139741, -0.5180101, -0.5182382]], dtype=float32), Array([-1.0336919], dtype=float32))], 

'circuit_weights': [(Array([[ 1.9237439e-05,  1.9237439e-05,  1.9237439e-05,  1.9237439e-05, 1.9237439e-05,  1.9237439e-05],
                    ...
                            [ 1.0205437e-05,  1.0205437e-05,  1.0205437e-05,  1.0205437e-05, 1.0205437e-05,  1.0205437e-05]]
                            , dtype=float32), 
                     Array([ 0.00192374, -0.00088524, -0.00032948,  0.00060027,  0.00102054], type=float32)), 
         
         (Array([[ 2.15363343e-06,  2.15312662e-06,  2.13509338e-06, 2.15067689e-06,  2.14474812e-06],
          ...
                 [ 8.07312574e-20,  8.07122646e-20,  8.00362642e-20, 8.06204346e-20,  8.03981815e-20]], dtype=float32), 
         Array([ 4.29322790e-06,  4.17438394e-04,  4.18947229e-06,  7.96445238e-05, -8.31138459e-04,  8.05015879e-05, -5.99780906e-05, -5.72774385e-04,
       ...
        1.18584613e-20,  1.02829800e-18,  1.20702195e-20,  1.62630326e-19,
        4.17417836e-18,  1.60936260e-19], dtype=float32))
        ]
}
"""