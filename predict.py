import torch
import helpers
import argparse
import json
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Path to image')
parser.add_argument('checkpoint', help='Path to checkpoint')
parser.add_argument('--top_k', type=int, default=1, help='Number of most likely classes')
parser.add_argument('--category_names', default='./cat_to_name.json', help='Path to category mapping')
parser.add_argument('--gpu', action="store_true", help='Use GPU')
args = parser.parse_args()

# Check if we can process on GPU
device = torch.device("cpu")
if args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print('GPU is not available')
        exit()


# Load category mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)



model = helpers.getModelFromCheckpoint(args.checkpoint)
helpers.showPredictions(args.input, model, cat_to_name, args.top_k, device)
