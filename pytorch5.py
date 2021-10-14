#保存和加载模型
import torch
import torch.nn as nn
from torchvision import models

class myNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module('conv1',nn.Conv2d(3,16,3))
        self.add_module('conv2',nn.Conv2d(16,128,3))
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

model = myNet1()

#保存、加载模型
torch.save(model,'mynet.pth')
model = torch.load('mynet.pth')

#仅保存模型参数和加载模型
model1 = models.AlexNet(10)
print(model1)
torch.save(model1.state_dict(),'mynet1.pth')
model = myNet1()
paremeters = torch.load('mynet.pth')
model.load_state_dict(paremeters)