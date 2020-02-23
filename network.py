from torch import nn, optim
from torchvision import models
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_units, hidden_units):
        super().__init__()
        self.input = nn.Linear(input_units, hidden_units)
        self.hidden = nn.Linear(hidden_units,102)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x):
        x = self.dropout(x)
        x = self.input(x)
        x = F.relu(x)
        x = self.hidden(x)
        x = F.log_softmax(x, dim=1)
        
        return x

def make_network(arch, hidden_units):
    if arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_units = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_units = 9216
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = Network(input_units, hidden_units)
    return model