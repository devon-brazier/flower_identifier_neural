import save_load
import argparse
from preprocessing import process_image
import pandas as pd
import json
import torch

from PIL import Image

parser = argparse.ArgumentParser(description='Neural Network Predictor')
parser.add_argument('inputs', nargs=2, type=str)
parser.add_argument('--top_k', action="store", dest='top_k', default='1', type=int)
parser.add_argument('--category_names', action="store", dest='category_names', default='cat_to_name.json', type=str)
parser.add_argument('--gpu', action="store_true", dest='gpu_bool', default=False, help='Set the GPU switch to TRUE')

options = parser.parse_args()

# Stored the cat_to_name dictionary
with open(options.category_names, 'r') as f:
    cat_to_name = json.load(f)

def predict(image_path, checkpoint, top_k):
    model = save_load.load_checkpoint(checkpoint)

    model.eval()
    model.class_to_idx = {v: k for k, v in model.class_to_idx.items()}

    image = Image.open(image_path)
    processed_img = torch.from_numpy(process_image(image))
    processed_img = processed_img.unsqueeze(0).float()

    if options.gpu_bool:
        model.to('cuda')
        processed_img = processed_img.to('cuda')
    else:
        model.to('cpu')

    with torch.no_grad():
        output = model.forward(processed_img)

        x = torch.exp(output)
        x = x.topk(top_k, dim=1)

    classes = [model.class_to_idx[v] for v in x[1].cpu().numpy()[0]]
    probs = list(x[0].cpu().numpy()[0])
    species = [cat_to_name[k] for k in classes]
    return probs, classes, species

probs, classes, species = predict(options.inputs[0], options.inputs[1], options.top_k)
print(pd.DataFrame(data={'Probability':probs}, index=species))
