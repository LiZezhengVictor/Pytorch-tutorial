#更多的构建神经网络模型的方法
import torch
import torch.nn as nn

#1.add_module
class myNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module('conv1',nn.Conv2d(3,16,3))
        self.add_module('conv2',nn.Conv2d(16,128,3))
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



#2.modulelist
class myNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(128,64),nn.Linear(64,64),nn.Linear(64,10)]
        )
    
    def forward(self,x):
        for f in self.linears:
            x = f(x)
        return x
# net = myNet()
# print(net)

#3.VGG构建（仅参考）
vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            '''这里视情况也可以改成其他类型如linear'''
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class myNet3(nn.Module):
    def __init__(self):
        super(myNet3,self).__init__()
        self.vgg = nn.ModuleList(vgg(vgg_cfg,3)) #ModuleList

    def forward(self,x):
        for l in self.vgg:
            x = l(x)

#查看网络结构
#直接看
net = myNet2()
print(net)

#module
net = myNet2()
for item in net.modules():
    print(item)

#named_modules看层的名字
net = myNet2()
for (i,j) in net.named_modules():
    print(i)

