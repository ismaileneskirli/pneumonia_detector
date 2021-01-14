from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import sys


data_transforms = {

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')       #2 boyutlu resmi 3 boyuta çekiyorum.
    image = loader(image).float()      # resmi data_transofrms a sokuyoruz ve istediğimiz forma getiriyoruz.

    image.clone().detach()

    image = image.unsqueeze(0)

    return image   ## 4 boyutlu bir image dönüyoruz.

if __name__ == '__main__':
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.load_state_dict(torch.load('bestmodel.pt'))  #artık modeli train etmektense classification.py dan çıktı olarak aldığımız bestmodel.pt yi kullanıyoruz.
    ################
    model_ft.eval()
    path_image=sys.argv[1]   #### input olarak trainde kullanmadığım test datalarımı kullanıyorum.
    img = image_loader(data_transforms['val'],path_image)
    
    #print(model_ft(img))
    cla =  np.argmax(model_ft(img).detach().numpy())

    if(cla==0):
        print("Hasta Saglikli")
    else:
        print("Hasta Zaturreli")


