import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
import urllib.request
import torchvision
import torch.nn.functional as F


# ------------------------
# Model & Data
# ------------------------

def create_model():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

def load_data():
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root='./data/custom', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    return dataloader

def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url) as f:
        classes = [line.decode("utf-8").strip() for line in f.readlines()]
    return classes


# ------------------------
# Visualization
# ------------------------

def imshow(img, title=None):
    img = img.detach().cpu()  # Detach before numpy
    img = img / 2 + 0.5  # Unnormalize if needed
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')


# ------------------------
# Main Visualization + Analysis Logic
# ------------------------

def process_and_visualize(model, original_images, adv_images, imagenet_labels):
    with torch.no_grad():
        outputs_orig = model(original_images)
        outputs_adv = model(adv_images)

    preds_orig = torch.argmax(outputs_orig, dim=1)
    preds_adv = torch.argmax(outputs_adv, dim=1)

    success_count = torch.sum(preds_orig != preds_adv).item()

    for i in range(len(original_images)):
        plt.figure(figsize=(6, 3))

        probs_orig = F.softmax(outputs_orig[i], dim=0)
        probs_adv = F.softmax(outputs_adv[i], dim=0)

        label_orig = imagenet_labels[preds_orig[i].item()]
        label_adv = imagenet_labels[preds_adv[i].item()]

        conf_orig = probs_orig[preds_orig[i].item()].item() * 100
        conf_adv = probs_adv[preds_adv[i].item()].item() * 100

        # Original
        plt.subplot(1, 2, 1)
        imshow(original_images[i])
        plt.title(f"Original:\n{label_orig}\nConf: {conf_orig:.2f}%")

        # Adversarial
        plt.subplot(1, 2, 2)
        imshow(adv_images[i])
        plt.title(f"After Gaussian Noise:\n{label_adv}\nConf: {conf_adv:.2f}%")

        plt.tight_layout()
        plt.show()

    print(f"\n✅ Attack success on {success_count}/{len(original_images)} images")
    accuracy = 100 - (success_count / len(original_images)) * 100
    print(f"🎯 Accuracy after attack: {accuracy:.2f}%")
