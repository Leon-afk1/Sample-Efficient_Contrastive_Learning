#!/usr/bin/env python3
"""
Contrastive learning loss function implementations:
- SimCLR (NT-Xent Loss)
- SupCon (Supervised Contrastive Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss used in SimCLR.

    For each (anchor, positive) pair, all other instances in the batch
    are treated as negatives.
    """
    
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, features):
        """
        Args:
            features: Tensor of shape (2*batch_size, projection_dim).
                     First batch_size rows are anchors,
                     next batch_size rows are positives.
        Returns:
            loss: Scalar tensor
        """
        batch_size = features.shape[0] // 2

        features = F.normalize(features, p=2, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # For anchor i, its positive is at position i + batch_size (and vice-versa)
        mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=features.device)
        for i in range(batch_size):
            mask[i, i + batch_size] = True
            mask[i + batch_size, i] = True

        identity_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)

        similarity_matrix = similarity_matrix / self.temperature

        # Exclude self-similarity
        similarity_matrix = similarity_matrix.masked_fill(identity_mask, -float('inf'))

        labels = torch.arange(2 * batch_size, device=features.device)
        labels = torch.where(labels < batch_size, labels + batch_size, labels - batch_size)

        loss = self.criterion(similarity_matrix, labels)
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Extension of SimCLR that uses class labels to identify all positives:
    all instances sharing the same label (excluding the anchor) are treated as positives.
    """
    
    def __init__(self, temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: Tensor of shape (2*batch_size, projection_dim).
                     First batch_size rows are anchors,
                     next batch_size rows are their augmentations.
            labels: Tensor of shape (batch_size,) - class labels for anchors.
        Returns:
            loss: Scalar tensor
        """
        batch_size = features.shape[0] // 2
        device = features.device

        # Duplicate labels for anchors and their augmentations
        labels = labels.contiguous().view(-1, 1)
        labels = torch.cat([labels, labels], dim=0)  # (2*batch_size, 1)

        features = F.normalize(features, p=2, dim=1)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        identity_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)

        # Positive mask: same class label, excluding self
        labels_equal = torch.eq(labels, labels.T).float()
        positive_mask = labels_equal.masked_fill(identity_mask, 0)

        num_positives = positive_mask.sum(dim=1)
        valid_anchors = num_positives > 0

        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        similarity_matrix_masked = similarity_matrix.masked_fill(identity_mask, -1e9)

        # Log-sum-exp trick for numerical stability
        max_sim = similarity_matrix_masked.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(similarity_matrix_masked - max_sim)
        log_sum_exp = max_sim + torch.log(exp_sim.sum(dim=1, keepdim=True))

        log_prob = similarity_matrix_masked - log_sum_exp

        # Average log-probability over positives for each anchor
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / num_positives.clamp(min=1e-6)
        loss = -mean_log_prob_pos[valid_anchors].mean()

        return loss


class TripletMarginLossWrapper(nn.Module):
    """Wrapper around nn.TripletMarginLoss to unify the interface."""
    
    def __init__(self, margin=1.0, p=2):
        super(TripletMarginLossWrapper, self).__init__()
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p)
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: Tensors of shape (batch_size, projection_dim)
        Returns:
            loss: Scalar tensor
        """
        return self.criterion(anchor, positive, negative)


def get_contrastive_loss(loss_type, **kwargs):
    """
    Factory function to instantiate the appropriate contrastive loss.

    Args:
        loss_type: 'triplet', 'simclr', or 'supcon'
        **kwargs: Loss-specific arguments (margin, temperature, etc.)
    Returns:
        Loss module
    """
    if loss_type == 'triplet':
        margin = kwargs.get('margin', 1.0)
        p = kwargs.get('p', 2)
        return TripletMarginLossWrapper(margin=margin, p=p)
    
    elif loss_type == 'simclr':
        temperature = kwargs.get('temperature', 0.5)
        return NTXentLoss(temperature=temperature)
    
    elif loss_type == 'supcon':
        temperature = kwargs.get('temperature', 0.5)
        return SupConLoss(temperature=temperature)
    
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Options: 'triplet', 'simclr', 'supcon'")


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Testing contrastive loss functions")
    print("="*80)

    batch_size = 8
    projection_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features_anchor = F.normalize(torch.randn(batch_size, projection_dim), p=2, dim=1).to(device)
    features_positive = F.normalize(torch.randn(batch_size, projection_dim), p=2, dim=1).to(device)
    features_negative = F.normalize(torch.randn(batch_size, projection_dim), p=2, dim=1).to(device)

    # 2 samples per class to ensure valid triplets
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).to(device)

    print("\n1. Triplet Loss:")
    triplet_loss = get_contrastive_loss('triplet', margin=1.0).to(device)
    loss = triplet_loss(features_anchor, features_positive, features_negative)
    print(f"   Loss: {loss.item():.4f}")

    print("\n2. SimCLR (NT-Xent Loss):")
    simclr_loss = get_contrastive_loss('simclr', temperature=0.5).to(device)
    features_combined = torch.cat([features_anchor, features_positive], dim=0)
    loss = simclr_loss(features_combined)
    print(f"   Loss: {loss.item():.4f}")

    print("\n3. SupCon Loss:")
    supcon_loss = get_contrastive_loss('supcon', temperature=0.5).to(device)
    loss = supcon_loss(features_combined, labels)
    print(f"   Loss: {loss.item():.4f}")

    print("\n" + "="*80)
    print("Tests completes!")
