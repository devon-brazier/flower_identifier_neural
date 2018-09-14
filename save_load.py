import torch
import preprocessing

from torchvision import models

def save(arch, model, directory):
    model.class_to_idx = preprocessing.train_dataset.class_to_idx
    checkpoint = {'arch': arch,
                  'class_to_idx': model.class_to_idx,
                  'features': model.features,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, directory)
    
def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    if checkpoint['arch'] == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.features = checkpoint['features']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(state_dict=checkpoint['state_dict'])
        
    return model