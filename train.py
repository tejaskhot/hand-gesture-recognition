import numpy as np
import random
import math
import sys
import os
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.utils.data as data_utils

data_dir = "depthGestRecog/"
label_dict={}
# find mappings for every label string : class id
for label in os.listdir(data_dir):
    for sub_dir in os.listdir(os.path.join(data_dir,label)):
        class_num = int(os.listdir(os.path.join(data_dir,label,sub_dir))[0][5:7])
        label_dict[label] = class_num
x = []
y = []
for root, _, files in os.walk(data_dir):
    for i in range(len(files)):
        x.append((plt.imread((os.path.join(root,files[i])))))
        y.append((int(files[i][5:7]) -1 ))

x = np.asarray(x, dtype=np.float32)
y = np.asarray(y)

# get train-val splits
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print('# train examples : ', len(X_train))
print('# test examples : ', len(X_test))

# Visualize a few entries from dataset
fig, ax = plt.subplots(11, 11, figsize=(11, 11))
# pick a random subset of the train set
indices = np.random.choice(len(X_train), 11*11)
for i, axi in enumerate(ax.flat):
    axi.imshow(X_train[indices[i]])
    axi.set(xticks=[], yticks=[])

# dataset utils
def convert2tensor(data, target, batch_size):
    tensor_data = torch.from_numpy(data)
    tensor_data = tensor_data.float()
    tensor_data = tensor_data.unsqueeze(1)
    tensor_target = torch.from_numpy(target)

    loader = data_utils.TensorDataset(tensor_data, tensor_target)
    loader_dataset = data_utils.DataLoader(loader, batch_size=batch_size, shuffle=True)
    return loader_dataset

batch_size = 64
train_loader = convert2tensor(X_train, y_train, batch_size)
test_loader = convert2tensor(X_test, y_test, batch_size)

# fix seed for reproducible results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# GestureNet is a very simple CNN equipped with ReLU non-linearity, dropout and batch normalization
# after each layer with no additional tricks. It works surprisingly well and trains in ~10-15 mins.
# The loss function used is cross-entropy which is a standard for multi-class classification tasks.
# Optimizer used is Adam because it typically leads to faster convergence than SGD.
# No schedule for learning rate is used here because the results are pretty good as is.

class GestureNet(nn.Module):
    def __init__(self):
        super(GestureNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(64 * 25 * 25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

model = GestureNet()
optimizer = optim.Adam(model.parameters(), lr=0.003)

if torch.cuda.is_available():
    model = model.cuda()

def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1)% 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))


def evaluate(data_loader):
    with torch.no_grad():
        loss = 0
        correct = 0
        total = 0

        for data, target in data_loader:
            data, target = Variable(data), Variable(target)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)

            loss += F.cross_entropy(output, target, reduction='sum').item()

            correct += (output.max(-1)[1] == target).sum().item()
            total += len(output)

        loss /= total

        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            loss, correct, total,
            100. * correct / total))

    return loss, (100. * correct / total)

n_epochs = 50
train_loss = []
val_loss = []
train_acc = []
val_acc = []
for epoch in range(n_epochs):
    train(epoch)
    t_loss, t_acc = evaluate(train_loader)
    v_loss, v_acc = evaluate(test_loader)

    train_loss.append(t_loss)
    train_acc.append(t_acc)
    val_loss.append(v_loss)
    val_acc.append(v_acc)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Loss')
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='val')
plt.xlabel('epochs')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='val')
plt.xlabel('epochs')
plt.ylabel('% accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('metrics.png')
plt.show()