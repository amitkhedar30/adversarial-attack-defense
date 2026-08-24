"""
Run this ONCE in your Kaggle session to precompute C&W adversarial
examples for a handful of demo images.

Why this exists: with 9-step binary search over c (~1000-1800 total
optimization steps per image), live C&W is fine on a Kaggle GPU but far
too slow on the CPU-only free tiers most dashboards deploy to (a slider
that takes 1-2 minutes to respond isn't an interactive demo). So instead
of running C&W live in the app, we precompute results here, once, and
the dashboard just displays them.

Prerequisites:
  - demo_images.pt already exported (run export_demo_images.py first)
  - resnet18_standard_cifar10.pt already in /kaggle/working/

Output: cw_gallery.pt -- copy this into dashboard/assets/ in your repo.
"""

import torch
import torch.optim as optim
from models.classifier import get_cifar10_resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GALLERY_IMAGES = 15  # subset of the 40 demo images to precompute


def cw_l2_attack(model, images, labels, targeted=False, kappa=0,
                  max_iterations=200, lr=0.01,
                  binary_search_steps=9, initial_const=1e-2,
                  c_upper_bound=1e10, c_lower_bound=0.0):
    """
    Same binary-search C&W L2 attack used in the main notebook (Cell A).
    Kept as a standalone copy here since this script runs independently
    of the notebook session.
    """
    device = images.device
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    batch_size = images.size(0)

    def atanh(x):
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)
        return 0.5 * torch.log((1 + x) / (1 - x))

    x_atanh = atanh((images * 2 - 1) * 0.999999)

    lower_bound = torch.full((batch_size,), c_lower_bound, device=device)
    upper_bound = torch.full((batch_size,), c_upper_bound, device=device)
    const = torch.full((batch_size,), initial_const, device=device)

    best_l2 = torch.full((batch_size,), 1e10, device=device)
    best_adv = images.clone()

    for _ in range(binary_search_steps):
        modifier = torch.zeros_like(images, requires_grad=True, device=device)
        optimizer = optim.Adam([modifier], lr=lr)
        step_success = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_iterations):
            optimizer.zero_grad()
            adv_images = 0.5 * (torch.tanh(modifier + x_atanh) + 1)
            l2_dist = torch.sum((adv_images - images) ** 2, dim=[1, 2, 3])

            outputs = model(adv_images)
            real_logits = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
            outputs_clone = outputs.clone()
            outputs_clone.scatter_(1, labels.unsqueeze(1), -float("inf"))
            other_logits = outputs_clone.max(1)[0]

            if targeted:
                f_loss = torch.clamp(other_logits - real_logits + kappa, min=0)
            else:
                f_loss = torch.clamp(real_logits - other_logits + kappa, min=0)

            loss = torch.sum(l2_dist + const * f_loss)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = outputs.argmax(1)
                success = (preds != labels) if not targeted else (preds == labels)
                step_success = step_success | success
                improved = success & (l2_dist < best_l2)
                best_l2 = torch.where(improved, l2_dist, best_l2)
                if improved.any():
                    best_adv[improved] = adv_images[improved].detach()

        for i in range(batch_size):
            if step_success[i]:
                upper_bound[i] = min(upper_bound[i].item(), const[i].item())
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
            else:
                lower_bound[i] = max(lower_bound[i].item(), const[i].item())
                if upper_bound[i] < c_upper_bound:
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    const[i] *= 10

    return best_adv


bundle = torch.load("/kaggle/working/demo_images.pt")
images, labels = bundle["images"], bundle["labels"]

model = get_cifar10_resnet18(num_classes=10)
model.load_state_dict(torch.load("/kaggle/working/resnet18_standard_cifar10.pt"))
model.to(DEVICE)
model.eval()

gallery = {}
for idx in range(min(N_GALLERY_IMAGES, len(images))):
    x = images[idx].unsqueeze(0).to(DEVICE)
    y = labels[idx].unsqueeze(0).to(DEVICE)

    adv = cw_l2_attack(model, x, y, max_iterations=200, binary_search_steps=9)
    l2_norm = torch.sqrt(torch.sum((adv - x) ** 2)).item()

    with torch.no_grad():
        pred = model(adv).argmax(1).item()

    gallery[idx] = {
        "adv_image": adv.squeeze(0).cpu(),
        "l2_norm": l2_norm,
        "prediction": pred,
    }
    print(f"[*] Image {idx}: L2={l2_norm:.4f}, pred={pred}, true={labels[idx].item()}")

torch.save(gallery, "/kaggle/working/cw_gallery.pt")
print(f"[*] Saved cw_gallery.pt with {len(gallery)} precomputed attacks")
print("[*] Copy this file into dashboard/assets/cw_gallery.pt")
