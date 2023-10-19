## load mnist

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
import wandb as wb
from skimage.io import imread

def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))

def plot(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()

## mnist

train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:, None, :, :] / 255
X_test = X_test[:, None, :, :] / 255

plt.imshow(X[1000, :].reshape(28, 28))

X = X.reshape(X.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

##Transpose

X = X.T

##Reshape image tensor


X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)

batch_size = 64
x = X[:, 0:64]
M = GPU(np.random.rand(10, 784))
y = M @ x

##random y=m@x model

y = torch.argmax(y, 0)
accuracy = torch.sum((y == Y[0:batch_size])) / batch_size

##randome walk

m_best = 0
acc_best = 0

y = torch.argmax(y, axis=0)
acc = ((y == Y)).sum() / len(Y)

if acc > acc_best:
    print(acc.item())
    m_best = m
    acc_best = acc
### code
view code [here](https://colab.research.google.com/drive/1qOSO5U4iMzSX_pQ0Ka-Twm39Wumksc87)
