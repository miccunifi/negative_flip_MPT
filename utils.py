import numpy as np

def compute_negative_flip_rate(new_logits, old_logits, targets):
    """
    Computes the negative flip rate (NFR).

    A negative flip is counted when:
      - The old model predicts correctly:  old_pred == target
      - The new model predicts incorrectly: new_pred != target

    Args:
        new_logits (ndarray): New model logits of shape (N, num_classes).
        old_logits (ndarray): Old model logits of shape (N, num_classes).
        targets (ndarray): Ground-truth labels of shape (N,).

    Returns:
        float: The negative flip rate (fraction of total samples).
    """
    # Compute predictions from logits
    new_preds = np.argmax(new_logits, axis=1)
    old_preds = np.argmax(old_logits, axis=1)
    
    # Create masks:
    # Mask for samples where the old model is correct.
    old_correct = (old_preds == targets)
    # Mask for samples where the new model is incorrect.
    new_incorrect = (new_preds != targets)
    
    # Negative flips: samples where old is correct and new is incorrect.
    neg_flip_mask = old_correct & new_incorrect

    # Compute the fraction of negative flips over the whole batch.
    nfr = np.mean(neg_flip_mask)
    
    return nfr


def compute_relative_negative_flip_rate(new_logits, old_logits, targets):
    """
    Computes the relative negative flip rate.

    Relative NFR is defined as:
        relative_NFR = NFR / ((1 - ER_old) * ER_new)
    where:
        ER_old = error rate of the old model
        ER_new = error rate of the new model

    This denominator is the expected negative flip rate if the new model's errors were
    independent of the old model's predictions (only considering samples the old model got right).

    Args:
        new_logits (ndarray): New model logits of shape (N, num_classes).
        old_logits (ndarray): Old model logits of shape (N, num_classes).
        targets (ndarray): Ground-truth labels of shape (N,).

    Returns:
        float: The relative negative flip rate.
    """
    new_preds = np.argmax(new_logits, axis=1)
    old_preds = np.argmax(old_logits, axis=1)
    
    # Overall error rates.
    er_old = np.mean(old_preds != targets)  # error rate of old model
    er_new = np.mean(new_preds != targets)  # error rate of new model
    
    # Negative flips: old correct and new incorrect.
    old_correct = (old_preds == targets)
    neg_flip_mask = old_correct & (new_preds != targets)
    nfr = np.mean(neg_flip_mask)
    
    # Expected negative flip rate on samples where old was correct.
    # Note that the fraction of samples where the old model is correct is (1 - ER_old).
    expected_nfr = (1.0 - er_old) * er_new

    # Avoid division by zero.
    if expected_nfr.item() == 0:
        relative_nfr = 0.0
    else:
        relative_nfr = nfr / expected_nfr

    return relative_nfr.item()