import torch
import torch.nn as nn

def fgsm_gaussian_attack(model, image, label, epsilon):
    # copy image to another variable
    copied_image = image.clone().detach().to(next(model.parameters()).device)
    label = label.to(image.device)

    # Create Gaussian noise (mean=0, std=1) multiplied by epsilon
    noise = torch.randn_like(copied_image) * epsilon

    # Add noise to the image
    adv_image = copied_image + noise

    # Clamp to keep pixel values in valid range [0, 1]
    adv_image = torch.clamp(adv_image, 0, 1)

    return adv_image
