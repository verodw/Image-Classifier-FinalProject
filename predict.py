import argparse
import torch
from torchvision import models, transforms
from torch import nn
import numpy as np
from PIL import Image
import json
from collections import OrderedDict

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('Invalid architecture.')

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    img = Image.open(image_path)
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = img_transforms(img)
    img = img.unsqueeze(0)

    return img

def predict(image_path, model, topk=5, gpu=False):
    model.eval()
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    image = process_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    probabilities = torch.exp(output)
    top_probs, top_classes = probabilities.topk(topk, dim=1)

    top_probs = top_probs.cpu().numpy()[0]
    top_classes = top_classes.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_classes]

    return top_probs, top_classes

def load_category_names(category_names_file):
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flower name and probabilities from an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to JSON file with category names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference.')

    args = parser.parse_args()

    # Load the checkpoint
    model = load_checkpoint(args.checkpoint)
    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
    else:
        cat_to_name = None

    top_probs, top_classes = predict(args.image_path, model, topk=args.top_k, gpu=args.gpu)

    if cat_to_name:
        top_classes = [cat_to_name[str(cls)] for cls in top_classes]

    for i in range(args.top_k):
        print(f"{i+1}: {top_classes[i]} with probability {top_probs[i]:.4f}")