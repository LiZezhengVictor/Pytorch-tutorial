#介绍Dataset和DataLoader的基本用法

import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np

class myDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.tensor([[1,2],[2,3],[3,3],[8,7]])
        self.label = torch.tensor([0,0,0,1])
    
    def __getitem__(self, index):
        return self.data[index],self.label[index]
    
    def __len__(self):
        return len(self.data.numpy())

mydata = myDataset()
data = DataLoader(dataset=mydata,batch_size=2,shuffle=True)

for i,(j,k) in enumerate(data):
    print((j,k))






