import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Importing your custom FGSM function and foolbox attack method
from fgsm import custom_fgsm_attach

# Define and load the model (ResNet18 for example)
def create_model():
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()  # set model to evaluation mode
    return model

# Load CIFAR-10 dataset (as an example)
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # DataLoader for test dataset
    testloader = DataLoader(testset, batch_size=4, shuffle=False)
    
    return testloader

# The main function to tie everything together
def main():
    # Load the model and data
    model = create_model()
    testloader = load_data()

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Get a batch of test images and labels
    data_iter = iter(testloader)
    images, labels = next(data_iter)

    # Run custom FGSM attack
    epsilon = 0.1  # Perturbation magnitude
    adv_image_custom = custom_fgsm_attach(model, loss_fn, images, labels, epsilon)
    print("Adversarial Image (Custom FGSM): ", adv_image_custom)

if __name__ == "__main__":
    main()
