import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt

data_path = 'data/' 

cifar10 = datasets.CIFAR10(data_path, train=True, download=False) 
#cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

len(cifar10)

img, label = cifar10[99] 
print(label)

plt.imshow(img) 
plt.show()


#Comienza la transformación de las imágenes a tensores para pytorch

from torchvision import transforms 
dir(transforms)

to_tensor = transforms.ToTensor() 
img_t = to_tensor(img) 
img_t.shape


#Se ha creado un tensor de una imagen, ahora se hace con todo el dataset

tensor_cifar10 = datasets.CIFAR10(data_path, 
                                  train=True, 
                                  download=False, 
                                  transform=transforms.ToTensor())

#Se comprueba que está bien

imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3) 
imgs.shape


#Se puede calcular la media por canal, la desviación típica, ...
print(imgs.view(3, -1).mean(dim=1), imgs.view(3, -1).std(dim=1))


#Normalización
transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))

transformed_cifar10 = datasets.CIFAR10( data_path, 
                                       train=True, 
                                       download=False,
                                       transform=transforms.Compose([ transforms.ToTensor(), 
                                                                     transforms.Normalize((0.4915, 0.4823, 0.4468), 
                                                                                          (0.2470, 0.2435, 0.2616)) ]))

transformed_cifar10_val = datasets.CIFAR10( data_path, 
                                       train=False, 
                                       download=False,
                                       transform=transforms.Compose([ transforms.ToTensor(), 
                                                                     transforms.Normalize((0.4915, 0.4823, 0.4468), 
                                                                                          (0.2470, 0.2435, 0.2616)) ]))
img_t, _ = transformed_cifar10[99]
plt.imshow(img_t.permute(1, 2, 0)) 
plt.show()


#Se crea una red sólo para distinguir entre pájaros y aviones

label_map = {0: 0, 2: 1} 
class_names = ['airplane', 'bird'] 
cifar2 = [(img, label_map[label]) 
          for img, label in transformed_cifar10
          if label in [0, 2]] 
cifar2_val = [(img, label_map[label]) for img, label in transformed_cifar10_val if label in [0, 2]]

#Construcción del modelo con una capa ocula de 512 neuronas
import torch.nn as nn

n_out = 2

model = nn.Sequential(
    nn.Linear(3072, 512,), 
    nn.Tanh(),
    nn.Linear(512, n_out,))

#Se desea representar las salidas como probabilidades
def softmax(x): 
    return torch.exp(x) / torch.exp(x).sum()

x = torch.tensor([1.0, 2.0, 3.0])
softmax(x)

softmax = nn.Softmax(dim=1)

model =  nn.Sequential( nn.Linear(3072, 512), 
                       nn.Tanh(), 
                       nn.Linear(512, 2), 
                       nn.Softmax(dim=1))


#Se observa un ejemplo, y se ve que efectivamente es un pájaro (o avión)

img, _ = cifar2[0]
plt.imshow(img.permute(1, 2, 0)) 
plt.show()

img_batch = img.view(-1).unsqueeze(0)

out = model(img_batch) 
out

_, index = torch.max(out, dim=1)
index

loss = nn.NLLLoss()

img, label = cifar2[0]
out = model(img.view(-1).unsqueeze(0))
loss(out, torch.tensor([label]))


from torch import optim

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

model = nn.Sequential( nn.Linear(3072, 512), 
                      nn.Tanh(), 
                      nn.Linear(512, 2), 
                      nn.LogSoftmax(dim=1))

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.NLLLoss()

n_epochs = 10

for epoch in range(n_epochs): 
    for imgs, labels in train_loader:

        batch_size = imgs.shape[0] 
        outputs = model(imgs.view(batch_size, -1)) 
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()

    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
    

#Validación en Test. Se cuenta los correctos frente al total (se supone el dataset bien balanceado)

val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

correct = 0 
total = 0

with torch.no_grad(): 
    for imgs, labels in val_loader: 
        batch_size = imgs.shape[0] 
        outputs = model(imgs.view(batch_size, -1)) 
        _, predicted = torch.max(outputs, dim=1) 
        total += labels.shape[0] 
        correct += int((predicted == labels).sum())

print("Accuracy: %f", correct / total)


#Análisis del número de parámetros del modelo

numel_list = [p.numel() 
              for p in model.parameters() 
              if p.requires_grad == True] 
sum(numel_list), numel_list