import torch

def validation(model, loader, criterion, gpu):
    model.eval()
    if gpu:
        model.to('cuda')
    loss = 0
    num_correct = 0
    for images,labels in iter(loader):
        if gpu:
            images = images.to('cuda')
            labels = labels.to('cuda')
        with torch.no_grad():
            output = model.forward(images)
        # .item() turns value from tensor to scalar
        loss += criterion(output, labels).item()
        ps = torch.exp(output)
        #labels.data gives label indices. ps.max(dim=1) returns the maximum probabilities tensor, and indices tensor. 
        #uses ps.max(dim=1)[1] to take just the indices for comparison with label indices.
        equality = (labels.data == ps.max(dim=1)[1])

        num_correct += equality.sum()
    return loss, num_correct
