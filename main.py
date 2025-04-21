import torch
import torch.nn as nn
from functions import create_model, load_data, load_imagenet_labels, process_and_visualize

# FGSM imports
from fgsm import custom_fgsm_attach
from fgsm_guassian import fgsm_gaussian_attack

def main():
    model = create_model()
    testloader = load_data()
    imagenet_labels = load_imagenet_labels()
    loss_fn = nn.CrossEntropyLoss()
    epsilon = 0.09

    # Get a batch
    data_iter = iter(testloader)
    images, labels = next(data_iter)

    # Choose attack type
    # adv_images = custom_fgsm_attach(model, loss_fn, images, labels, epsilon)
    adv_images = fgsm_gaussian_attack(model, images, labels, epsilon)

    # Process results & visualize
    process_and_visualize(model, images, adv_images, imagenet_labels)

if __name__ == "__main__":
    main()
