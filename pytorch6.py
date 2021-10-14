#torchvision.transforms常用API
#用于预处理图片

from torchvision.datasets import ImageFolder
from torchvision import transforms

input_size = 224
path = ''

#构建处理图片的过程用compose，处理方法用transform中的方法
data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#使用上述方法处理文件夹path里的图片
image_datasets = ImageFolder(path, data_transforms) 
