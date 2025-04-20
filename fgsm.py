import torch
import torch.nn as nn 


def custom_fgsm_attach(model, loss_fn, image, label, epsilon):
    #make sure our input image require gradients 
    image.requires_grad = True
    
    #forward pass
    output = model(image)
    loss = loss_fn(output, label)
    
    #backward pass; calculation gradient of loss w.r.t image
    model.zero_grad()
    loss.backward()
    
    image_grad = image.grad.data
    sign_data = image_grad.sign()
    
    #create adversarial attack
    adv_image = image + epsilon * sign_data
    
    #restrict the values between 0-1 if changes made to pixels
    adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image