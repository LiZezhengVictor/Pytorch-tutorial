# MNIST项目实战改良
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
import torch.optim as optim

batch_size = 64
epoch = 1

class myMnist(Dataset):
    def __init__(self,filepath,transform,train = True):
        data = pd.read_csv(filepath)
        self.x = data.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:, :, :, None]
        self.y = torch.from_numpy(data.iloc[:,0].values)
        self.train = train
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.train:
            return self.transform(self.x[index]),self.y[index]
        else:
            return self.transform(self.x[index])

transform= transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(0.5,))])

train_data = myMnist('data/mnist_train.csv',transform,train=True)
test_data = myMnist('data/mnist_test.csv',transform,train=False)
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=batch_size,drop_last=False)

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__() 
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(64*4*4,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,10)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)   #conv --> fc
        x = self.fc(x)
        return x

model = net()
optimizer = optim.Adam(model.parameters(),lr=0.005)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
print(model)

def train(epoch):
    model.train()
    for batch,(data,label) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch + 1) * len(data), len(train_loader.dataset),
                       100. * (batch + 1) / len(train_loader), loss.item()))
               
    exp_lr_scheduler.step()

for e in range(epoch):
    train(e)

def prediciton(data_loader):
    model.eval()
    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):
        output = model(data)
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
    return test_pred

test_pred = prediciton(test_loader)
print("***********")
print(test_pred)
print("***********")
#torch.save(model.state_dict(),'mynet1.pth')