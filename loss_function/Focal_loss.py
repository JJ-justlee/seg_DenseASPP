import torch
import torch.nn.functional as F
from loss_function.Cross_Entropy_loss import Cross_Entropy_loss_ignore

def log_softmax_manual(x, dim):
    return x - torch.logsumexp(x, dim=dim, keepdim=True)

# 감마 2, 알파 0.25
def Focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | list[float] | None = None,
    ignore_index: int = 255,
):
    # CE loss
    log_probs   = log_softmax_manual(logits, dim=1)                 # (B,C,H,W)
    targets_long = targets.long()                                   # (B,H,W)
    mask     = (targets_long != ignore_index)
    
    safe_targets = targets_long.clone()
    safe_targets[~mask] = 0
    
    ce_map      = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)  # (B,H,W)

    # Focal factor
    pt = torch.exp(-ce_map)                 # p_t
    focal_factor = (1.0 - pt) ** gamma      # (1-p_t)^γ

    if alpha is not None:
        if isinstance(alpha, (list, tuple, torch.Tensor)):
            alpha = torch.as_tensor(alpha, dtype=logits.dtype, device=logits.device)
            alpha_t = alpha[safe_targets]   # (B,H,W)
        else:                               # 스칼라
            alpha_t = alpha
        focal_factor = focal_factor * alpha_t

    # 손실 계산
    loss_map = focal_factor * ce_map        # FL = (1-p_t)^γ * α_t * CE
    denom    = mask.sum().clamp(min=1)      # 0 나눗셈 방지
    loss     = (loss_map * mask.float()).sum() / denom
    return loss

# def Focal_loss(logits, targets, gamma=2.0, alpha=0.25):
#     ce_map, mask, safe_targets = Cross_Entropy_loss_ignore(logits, targets, return_map=True)
#     pt = torch.exp(-ce_map)
#     focal_factor = (1.0 - pt) ** gamma

#     if alpha is not None:
#         if isinstance(alpha, (list, tuple, torch.Tensor)):
#             alpha = torch.as_tensor(alpha, dtype=logits.dtype, device=logits.device)
#             alpha = alpha[safe_targets]
#         focal_factor = focal_factor * alpha

#     loss = (focal_factor * ce_map * mask.float()).sum() / mask.sum().clamp(min=1)
#     return loss