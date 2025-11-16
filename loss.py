import torch
import torch.nn.functional as F

def focal_distillation_naive_loss(output, old_output, target):
    # Compute predictions from the old model and create weights
    old_pred = torch.argmax(old_output, dim=1)
    sample_weights = (old_pred == target).type(output.dtype)
    
    # Compute the negative log-likelihood for the new model predictions
    log_prob = F.log_softmax(output, dim=1)
    # Gather log probabilities for the correct class
    log_loss = -log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    
    # Weight the loss per sample and take the mean
    loss = torch.mean(sample_weights * log_loss)
    return loss

def focal_distillation_fdkl_loss(output, old_output, target, alpha=1.0, beta=5.0, tau=100.0):
    # Compute predictions from the old model and create weights
    old_pred = torch.argmax(old_output, dim=1)
    sample_weights = (old_pred == target).type(output.dtype)
    
    # Temperature scaled logits and probabilities
    log_prob_new = F.log_softmax(output / tau, dim=1)
    prob_old = F.softmax(old_output / tau, dim=1)
    
    # Compute per-sample KL divergence
    kl_div = F.kl_div(log_prob_new, prob_old, reduction='none').sum(dim=1)
    
    # Weight and average the KL divergence loss
    loss = torch.mean((alpha + beta * sample_weights) * kl_div)
    return loss


def focal_distillation_fdlm_loss(output, old_output, target, alpha=1.0, beta=5.0):
    # Compute predictions from the old model and create weights
    old_pred = torch.argmax(old_output, dim=1)
    sample_weights = (old_pred == target).type(output.dtype)
    
    # Compute L2 loss between old and new outputs (mean squared error per sample divided by 2)
    l2_loss = ((output - old_output).pow(2).mean(dim=1)) / 2
    
    # Weight and average the L2 loss
    loss = torch.mean((alpha + beta * sample_weights) * l2_loss)
    return loss

def lm_loss(output, output2):
    # Compute L2 loss between old and new outputs (mean squared error per sample divided by 2)
    l2_loss = ((output - output2).pow(2).mean(dim=1)) / 2
    
    # Weight and average the L2 loss
    loss = torch.mean(l2_loss)
    return loss