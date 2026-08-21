def pgd_attack_ce(model, X, y, epsilon, alpha, num_iter):
    """Inner loop: generate PGD examples maximizing Cross-Entropy."""
    X_adv = X.clone().detach().requires_grad_(True)
    
    for _ in range(num_iter):
        with torch.enable_grad():
            loss = F.cross_entropy(model(X_adv), y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        
        # PGD Step + Projection to L_inf ball
        X_adv = X_adv.detach() + alpha * torch.sign(grad.detach())
        X_adv = torch.min(torch.max(X_adv, X - epsilon), X + epsilon)
        X_adv = torch.clamp(X_adv, 0.0, 1.0) # Ensure valid image range
        X_adv.requires_grad_(True)
        
    return X_adv.detach()

def train_pgd_at(model, train_loader, optimizer, total_epochs, device):
    start_epoch = load_latest_checkpoint(model, optimizer, "PGD-AT")
    model.train()
    
    for epoch in range(start_epoch, total_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 1. Generate adversarial examples
            data_adv = pgd_attack_ce(model, data, target, epsilon=8/255, alpha=2/255, num_iter=10)
            
            # 2. Train on adversarial examples
            optimizer.zero_grad()
            output = model(data_adv)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
        print(f"PGD-AT Epoch {epoch} complete. Loss: {loss.item():.4f}")
        save_checkpoint(model, optimizer, epoch, "PGD-AT")
