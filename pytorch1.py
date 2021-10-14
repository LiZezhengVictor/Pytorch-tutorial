#如何编写一个最简单的神经网络

#1.导入库
import torch
import torch.nn as nn
import numpy as np

#2.准备数据和标签
inputmat = np.array([[0,0],[0,1],[1,0],[1,1]])
inputmat = torch.from_numpy(inputmat).float()
output = np.array([1,0,0,1])
output = torch.from_numpy(output).float()
print(inputmat,output)

#3.设计网络结构
myNet = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
    )
print(myNet)

#4.设计optimazer和loss
opt = torch.optim.SGD(myNet.parameters(),lr=0.05)
func_loss = nn.MSELoss()

#5.训练
for epoch in range(5000):
    out = myNet(inputmat)
    loss = func_loss(out,output)
    opt.zero_grad()
    loss.backward()
    opt.step()

print('e')
print(myNet(inputmat).data)