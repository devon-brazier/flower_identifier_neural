import save_load
import preprocessing
import time
import torch
import argparse

from torchvision import models
from torch import nn, optim
from network_class import Network

parser = argparse.ArgumentParser(description='Neural Network Trainer')
parser.add_argument('--save_dir', action="store", dest='save_dir', default='checkpoint.pth')
parser.add_argument('--arch', action="store", dest="arch", default='vgg16')
parser.add_argument('--learnrate', action="store", dest="learnrate", type=float, default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=0)
parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=5)
parser.add_argument('--gpu', action="store_true", dest='gpu_bool', default=False, help='Set the GPU switch to TRUE')

options = parser.parse_args()
epochs = options.epochs

if options.arch == 'vgg11':
    model = models.vgg11(pretrained=True)
elif options.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
elif options.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif options.arch == 'vgg19':
    model = models.vgg19(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
class_input = model.classifier[0].in_features

model.classifier = Network(options.hidden_units, class_input)
criterion = nn.NLLLoss()
optimizer = optim.Adam(params=model.classifier.parameters(), lr=options.learnrate)

train_dataset, valid_dataset, test_dataset, trianloader, validloader, testloader = preprocessing.datasets()

def validation(model, valid_loader, criterion):
    valid_loss = 0
    accuracy = 0
    step = 0
    if options.gpu_bool:
        model.to("cuda")
    
    for flower_images, labels in valid_loader:
        if options.gpu_bool:
            flower_images, labels = flower_images.to("cuda"), labels.to("cuda")
        
        output = model.forward(flower_images)
        loss = criterion(output, labels)

        valid_loss += loss.item()
        probabilities = torch.exp(output)
        is_accurate = (probabilities.max(dim=1)[1] == labels.data)
        accuracy += is_accurate.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

#Training script
if options.gpu_bool:
    model.to("cuda")
print_every = int(len(trainloader))
steps = 0
running_loss = 0

for e in range(epochs):
    model.train()
    start = time.time()

    for flower_images, labels in trainloader:
        if options.gpu_bool:
            flower_images, labels = flower_images.to("cuda"), labels.to("cuda")
        steps += 1

        optimizer.zero_grad()

        output = model.forward(flower_images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            # Set model to evaluation mode, cancels the dropout to make sure the network is tested.
            model.eval()
            # Make sure the Tensor is not tracking operations on itself, to save memory.
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}".format(e+1, epochs),
                      "Training Loss: {}".format(running_loss/print_every),
                      "Validation Loss: {}".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            running_loss = 0
            model.train()
    print("Time Elapsed for epoch: {:.3f} seconds".format(time.time() - start))

# saves the model to the specified directory, if none specified it defaults to checkpoint.pth
save_load.save(options.arch, model, options.save_dir)
