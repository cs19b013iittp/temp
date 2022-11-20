import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"


  
class Dyn_CNN(nn.Module):
    def __init__(self, config, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = len(config)
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(in_channels=i, out_channels=o, stride=s, kernel_size=k, padding=p) for i, o, s, k, p in config])
        self.relu_layers = nn.ModuleList([nn.ReLU() for i in range(len(config))])
        self.fc_in_h = input_dim[0]
        self.fc_in_w = input_dim[1]
        for i,o,s,f,p in config:       #stride=(heightxwidth)
            self.fc_in_h = ((self.fc_in_h-f[0]+2*p)//s)+1
            self.fc_in_w = ((self.fc_in_w-f[1]+2*p)//s)+1
        # print('in_features = ' , config[-1][1]*self.fc_in_h*self.fc_in_w)
        self.fc = nn.Linear(in_features = config[-1][1]*self.fc_in_h*self.fc_in_w, out_features=num_classes)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        self.conv_out_feature_maps = 0
        for i,l in enumerate(self.conv_layers):
            x = self.conv_layers[i](x)
            x = self.relu_layers[i](x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc(x)
        x = self.softmax(x)
        return x
def get_datasets():
    train_data = datasets.FashionMNIST(
        root="data",
        train = True,
        download = True,
        transform = ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root = "data",
        train = True,
        download = True,
        transform = ToTensor()
    )
    
    return train_data, test_data

def get_dataloaders(train_data, test_data):
    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    return train_dataloader, test_dataloader

def loss_fun(y_pred, y_actual):
  v = -(y_actual * torch.log(y_pred+1e-10))
  v = torch.sum(v)
  return v

def get_optim(DynCNN_model, lr=1e-6):
    optim = torch.optim.SGD(DynCNN_model.parameters(), lr = lr)
    return optim
    
def get_num_classes(train_data):
    return len(train_data.classes)

def get_input_dim(test_dataloader):
    _x, _y = None,None

    for X, y in test_dataloader:
        _x = X.shape
        _y = y.shape
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return _x[2],_x[3] #heightxwidth
def get_model(config, input_dim, num_classes):
    model = Dyn_CNN(config, input_dim, num_classes)
    return model

def train_network(train_dataloader, model, optim, loss_fn, epochs=5):
    print('Training Model ...\n\n')
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs , labels = inputs.to(device), labels.to(device)
            optim.zero_grad()
            outputs = model(inputs)
            tmp = torch.nn.functional.one_hot(labels, num_classes= 10)
            loss = loss_fn(outputs, tmp)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            # return
        print("[Epoch : {}/{}] loss = ".format(epoch+1,epochs),running_loss)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from torchmetrics import Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import accuracy

def test_network(dataloader, model, loss_fun):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            tmp = torch.nn.functional.one_hot(y, num_classes= 10)
            pred = model(X)
            test_loss += loss_fun(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss/= num_batches
    correct/=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    accuracy1 = Accuracy().to(device)
    print('Accuracy :', accuracy1(pred,y))
    precision = Precision(average = 'macro', num_classes = 10).to(device)
    print('precision :', precision(pred,y))

    recall = Recall(average = 'macro', num_classes = 10).to(device)
    print('recall :', recall(pred,y))
    f1_score = F1Score(average = 'macro', num_classes = 10).to(device)
    print('f1_score :', f1_score(pred,y))
    return accuracy1,precision, recall, f1_score
# train_data, test_data = get_datasets()
# train_dataloader, test_dataloader = get_dataloaders(train_data, test_data)
# num_classes = get_num_classes(test_data)
# input_dim = get_input_dim(train_dataloader)
# config = [(1, 20, 1, (5,5), 0), (20, 50, 1, (5,5), 0)]
# model = get_model(config, input_dim, num_classes)
# model = model.to(device)
# optim = get_optim(model)
# train_network(train_dataloader, model, optim, loss_fun)
# test_network(test_dataloader, model, loss_fun)
