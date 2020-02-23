import json
import numpy as np
import torch
from torchvision import transforms, datasets, models
from network import Network
import PIL
from PIL import Image
from validate import validation
from torch import nn, optim
from parse_arguments import make_predict_parser
from network import make_network


img_path, cat_names, gpu, topk = make_predict_parser()

data_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),                                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#test_dataset = datasets.ImageFolder('flowers/test', transform=data_transforms)
#testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
if cat_names:
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
   
#Import pretrained model

if gpu:
    checkpoint = torch.load('checkpoint.pth')
else:
    checkpoint = torch.load('checkpoint.pth', map_location=lambda storage, loc: storage)

model = make_network(checkpoint['arch'], checkpoint['hidden_units'])
model.load_state_dict(checkpoint['model_state_dict'])
class_to_idx = checkpoint['class_to_idx']

def process_image(image):
    image = Image.open(image)
    image = data_transforms(image)
    return image.numpy()

def predict(image_path, model, topk=3):
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image.unsqueeze_(0)
    if gpu:
        image = image.to('cuda')
        model = model.to('cuda')
    output = model.forward(image)
    output = output.exp()
    probs,classes = output.topk(topk)    
    classes = classes.cpu().numpy()[0]
    probs = probs.detach().cpu().numpy()[0]
    idx_to_class = {_class: index for index, _class in class_to_idx.items()}
    classes = list(map(lambda x : idx_to_class[x], classes ))
    return probs, classes

probs, classes = predict(img_path, model, topk)
if cat_names:
    name_classes = list(map(lambda x: cat_to_name[x], classes))
    print(' Predicted species is {} with {:.2f}% confidence. \n '.format(name_classes[0], probs[0]*100))
    for num in range(1,topk):
        print('{} most likely species is {}'.format(num+1, name_classes[num]))
else:
    print(' Predicted class is {} with {:.2f}% confidence. \n '.format(classes[0], probs[0]*100))
    for num in range(1,topk):
        print('{} most likely class is {}'.format(num+1, classes[num]))
    



