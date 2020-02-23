import json
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from network import make_network
from validate import validation
from parse_arguments import make_train_parser

data_dir, arch, hidden_units, epochs, learning_rate, gpu, save_dir = make_train_parser()

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms)

trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
model = make_network(arch, hidden_units)
   
def train_network(model):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print_every = 40
    if gpu:
        model.to('cuda')
    curr_epoch = 0
    for e in range(epochs): 
        curr_epoch += 1
        training_loss = 0
        steps = 0
        for images, labels in iter(trainloader):
            steps += 1
            if gpu:
                images = images.to('cuda')
                labels = labels.to('cuda')
            optimizer.zero_grad()       
            output = model.forward(images)
            loss = criterion(output, labels)
            training_loss += loss
            loss.backward()
            optimizer.step()
        validation_loss, num_correct = validation(model, validationloader, criterion, gpu)
        print("epoch: {} \n total training loss: {:.4f} \n average training loss: {:.4f} \n total validation loss: {:.4f} \n average validation loss: {:.4f} \n validation accuracy: {:.2f}%".format(curr_epoch, training_loss, training_loss/len(training_dataset), validation_loss, validation_loss/len(validation_dataset), int(num_correct)*100/len(validation_dataset)))
            

train_network(model)

torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': training_dataset.class_to_idx,
            'arch': arch,
            'hidden_units': hidden_units
            }, save_dir)

