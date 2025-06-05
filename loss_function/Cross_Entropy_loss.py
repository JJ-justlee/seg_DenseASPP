import torch
import torch.nn.functional as F

def log_softmax_manual(x, dim):
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    log_den = torch.log(torch.exp(x_shifted).sum(dim=dim, keepdim=True) + 1e-12)
    return x_shifted - log_den

def log_softmax_manual_logsumexp(x, dim):
    return x - torch.logsumexp(x, dim=dim, keepdim=True)

def Cross_Entropy_loss(logits, targets):
    
    log_probs = log_softmax_manual(logits, dim=1)
    # # or
    # log_probs = log_softmax_manual_logsumexp(logits, dim=1)
    
    targets = targets.long().unsqueeze(1)
    loss_map = -log_probs.gather(1, targets)
    
    return loss_map.mean()

def Cross_Entropy_loss_ignore(logits, targets, return_map=False):
    
    log_probs = log_softmax_manual(logits, dim=1)
    # # or
    # log_probs = log_softmax_manual_logsumexp(logits, dim=1)
    
    targets_long = targets.long()
    
    mask = (targets != 255)
    safe_targets = targets_long.clone()
    safe_targets[~mask] = 0
    
    ce_loss_map = -log_probs.gather(1, safe_targets.unsqueeze(1))
    ce_loss_map = ce_loss_map.squeeze(1)

    if return_map:
        return ce_loss_map, mask, safe_targets

    loss = (ce_loss_map * mask.float()).sum() / mask.sum()
    
    return loss
