import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon):
    """
    Fast Gradient Sign Method (FGSM).
    Generates a single-step adversarial perturbation.
    """
    images.requires_grad = True
    outputs = model(images)
    
    # Calculate Cross Entropy Loss
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    
    # Collect the sign of the data gradient
    sign_data_grad = images.grad.data.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_images = images + epsilon * sign_data_grad
    
    # Clip to maintain valid pixel range [0, 1]
    perturbed_images = torch.clamp(perturbed_images, 0.0, 1.0)
    
    return perturbed_images.detach()

def pgd_attack(model, images, labels, epsilon, alpha, iters):
    """
    Projected Gradient Descent (PGD).
    An iterative extension of FGSM with random starts.
    """
    # Start with a random noise initialization inside the epsilon ball
    perturbed_images = images.clone().detach() + torch.empty_like(images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0.0, 1.0)
    
    for i in range(iters):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        # Iterative update
        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
        
        # Projection: Ensure perturbation stays within epsilon bounds of original image
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        perturbed_images = torch.clamp(images + eta, min=0.0, max=1.0).detach()
            
    return perturbed_images
