def trades_loss(model, x_natural, y, optimizer, step_size, epsilon, perturb_steps, beta):
    """Calculates the TRADES loss."""
    # 1. Clean loss (Standard cross-entropy)
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    
    # 2. Generate adversarial examples maximizing KL divergence
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                               F.softmax(model(x_natural), dim=1),
                               reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
    x_adv = x_adv.detach()
    
    # 3. Calculate final KL penalty on the generated adversarial examples
    loss_robust = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                           F.softmax(model(x_natural), dim=1),
                           reduction='batchmean')
    
    # Combine clean loss and robust penalty
    loss = loss_natural + beta * loss_robust
    return loss

def train_trades(model, train_loader, optimizer, total_epochs, device):
    start_epoch = load_latest_checkpoint(model, optimizer, "TRADES")
    model.train()
    
    for epoch in range(start_epoch, total_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            loss = trades_loss(model, data, target, optimizer, 
                               step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0)
            loss.backward()
            optimizer.step()
            
        print(f"TRADES Epoch {epoch} complete. Loss: {loss.item():.4f}")
        save_checkpoint(model, optimizer, epoch, "TRADES")
