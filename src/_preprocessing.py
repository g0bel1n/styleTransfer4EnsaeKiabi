import torchvision.transforms as transforms
from torch import float, FloatTensor, device
from PIL import Image

class Normalize(object):
    def __init__(self, mean =(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean 
        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class Denormalize(object):
    def __init__(self, mean =(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.clone().mul_(s).add_(m)     
        return tensor

def loader(path : str, device : device = device('cpu'), loader = None) -> FloatTensor:
    img = Image.open(path)

    if loader is None :
        loader=transforms.Compose([transforms.Resize((512,512)),
         transforms.ToTensor(),
          #Normalize()
          ]) 

    img=loader(img).unsqueeze(0)
    return img.to(device,float)