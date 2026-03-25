"""
Contrastive Learning training with multiple loss functions.
Phase 1: Contrastive pre-training (Triplet, SimCLR, SupCon)
Phase 2: Supervised fine-tuning for classification
Strategies: Stratified, LOSO, LOGO
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import des loss functions contrastives
from contrastive_losses import get_contrastive_loss

# For embedding visualizations
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("WARNING: t-SNE not available")

import random

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
DATA_FRACTION = 1.0  # Fraction of training data to use (1.0 = 100%)

# Fixer les seeds (reproductibilité complète)
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("="*80)
print("CONTRASTIVE LEARNING - MULTI-LOSS SUPPORT")
print("FULL CROSS-VALIDATION ON ALL STRATEGIES")
print("="*80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*80)

# ============================================================================
# 1. DATASET CLASSES
# ============================================================================

class TripletDataset(Dataset):
    """Dataset for Triplet Loss training."""

    def __init__(self, X, y, triplet_indices, augment=False,
                 shift_prob=0.0, shift_max_frac=0.1):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.anchors = triplet_indices['anchors']
        self.positives = triplet_indices['positives']
        self.negatives = triplet_indices['negatives']
        self.augment = augment
        self.shift_prob = shift_prob
        self.shift_max_frac = shift_max_frac
    
    def __len__(self):
        return len(self.anchors)
    
    def augment_sample(self, x):
        """Applies random augmentations (jitter, scaling, and optional temporal shifting)."""
        if np.random.random() < 0.5:
            # Jitter
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        if np.random.random() < 0.5:
            # Scaling
            scale = np.random.uniform(0.8, 1.2)
            x = x * scale

        if self.shift_prob > 0.0 and np.random.random() < self.shift_prob:
            # Temporal shifting (circular shift along time axis)
            seq_len = x.shape[0]
            max_shift = max(1, int(seq_len * self.shift_max_frac))
            shift = np.random.randint(-max_shift, max_shift + 1)
            x = torch.roll(x, shifts=shift, dims=0)

        return x

    def __getitem__(self, idx):
        anchor_idx = self.anchors[idx]
        positive_idx = self.positives[idx]
        negative_idx = self.negatives[idx]

        anchor = self.X[anchor_idx].clone()
        positive = self.X[positive_idx].clone()
        negative = self.X[negative_idx].clone()

        if self.augment:
            anchor = self.augment_sample(anchor)
            positive = self.augment_sample(positive)
            negative = self.augment_sample(negative)

        return anchor, positive, negative


class PairDataset(Dataset):
    """Dataset for SimCLR and SupCon (anchor-positive pairs with augmentation)."""
    
    def __init__(self, X, y, augment=True, shift_prob=0.0, shift_max_frac=0.1):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.shift_prob = shift_prob
        self.shift_max_frac = shift_max_frac
    
    def __len__(self):
        return len(self.X)
    
    def augment_sample(self, x):
        """Applies random augmentations (jitter, scaling, and optional temporal shifting)."""
        if np.random.random() < 0.5:
            # Jitter
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        if np.random.random() < 0.5:
            # Scaling
            scale = np.random.uniform(0.8, 1.2)
            x = x * scale

        if self.shift_prob > 0.0 and np.random.random() < self.shift_prob:
            # Temporal shifting (circular shift along time axis)
            seq_len = x.shape[0]
            max_shift = max(1, int(seq_len * self.shift_max_frac))
            shift = np.random.randint(-max_shift, max_shift + 1)
            x = torch.roll(x, shifts=shift, dims=0)

        return x

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        label = self.y[idx]

        # Create two augmented views of the same sample
        if self.augment:
            x1 = self.augment_sample(x)
            x2 = self.augment_sample(x)
        else:
            x1 = x
            x2 = x.clone()

        return x1, x2, label


class StandardDataset(Dataset):
    """Standard dataset for classification."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AugmentedDataset(Dataset):
    """
    Dataset returning a single augmented (sample, label) pair per index.
    Used for online semi-hard triplet mining, where triplets are selected
    dynamically from the batch embeddings rather than pre-generated.
    """

    def __init__(self, X, y, augment=True, shift_prob=0.0, shift_max_frac=0.1):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.shift_prob = shift_prob
        self.shift_max_frac = shift_max_frac

    def __len__(self):
        return len(self.X)

    def augment_sample(self, x):
        """Applies random augmentations (jitter, scaling, optional temporal shift)."""
        if np.random.random() < 0.5:
            # Additive Gaussian jitter
            x = x + torch.randn_like(x) * 0.05

        if np.random.random() < 0.5:
            # Random amplitude scaling
            x = x * np.random.uniform(0.8, 1.2)

        if self.shift_prob > 0.0 and np.random.random() < self.shift_prob:
            # Circular temporal shift
            seq_len = x.shape[0]
            max_shift = max(1, int(seq_len * self.shift_max_frac))
            shift = np.random.randint(-max_shift, max_shift + 1)
            x = torch.roll(x, shifts=shift, dims=0)

        return x

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x = self.augment_sample(x)
        return x, self.y[idx]


# ============================================================================
# 2. MODELS
# ============================================================================

class SDCBlock(nn.Module):
    """Spatial Dilated Convolution Block."""
    
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(SDCBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, padding=dilation_rate, dilation=dilation_rate
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SDCNetEncoder(nn.Module):
    """SDCNet encoder backbone (without classification head)."""
    
    def __init__(self, input_channels=6, base_filters=64):
        super(SDCNetEncoder, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv1d(input_channels, base_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        # SDC blocks with increasing dilation rates
        self.sdc1 = SDCBlock(base_filters, base_filters, dilation_rate=1)
        self.sdc2 = SDCBlock(base_filters, base_filters*2, dilation_rate=2)
        self.sdc3 = SDCBlock(base_filters*2, base_filters*2, dilation_rate=4)
        self.sdc4 = SDCBlock(base_filters*2, base_filters*4, dilation_rate=8)
        self.sdc5 = SDCBlock(base_filters*4, base_filters*4, dilation_rate=16)
        
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.embedding_dim = base_filters * 4
    
    def forward(self, x):
        # x: (batch, timesteps, features) -> (batch, features, timesteps)
        x = x.transpose(1, 2)

        # Initial layer
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        # SDC blocks
        x = self.sdc1(x)
        x = self.pool(x)
        x = self.sdc2(x)
        x = self.sdc3(x)
        x = self.pool(x)
        x = self.sdc4(x)
        x = self.sdc5(x)

        x = self.dropout(x)

        # Global pooling -> flat embedding
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x


class ContrastiveModel(nn.Module):
    """Full model for contrastive pre-training (encoder + projection head)."""

    def __init__(self, input_channels=6, projection_dim=256):
        super(ContrastiveModel, self).__init__()

        self.encoder = SDCNetEncoder(input_channels=input_channels)

        # Projection head maps embeddings to the contrastive space
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projection_head(embedding)
        return F.normalize(projection, p=2, dim=1)

    def get_embedding(self, x):
        """Returns the encoder embedding (without projection head)."""
        return self.encoder(x)


class ClassificationModel(nn.Module):
    """Classifier for fine-tuning: frozen or trainable encoder + linear head."""

    def __init__(self, encoder, num_classes):
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.embedding_dim, num_classes)
    
    def forward(self, x):
        embedding = self.encoder(x)
        return self.classifier(embedding)


# ============================================================================
# 3. TRIPLET GENERATION
# ============================================================================

def generate_triplets_for_subset(y_subset, n_triplets):
    """
    Generates triplets for a data subset.
    """
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(y_subset):
        label_to_indices[label].append(idx)
    
    labels_list = list(label_to_indices.keys())
    
    # Only use classes with at least 2 samples
    valid_labels = [label for label in labels_list
                   if len(label_to_indices[label]) >= 2]

    if len(valid_labels) < 2:
        print(f"WARNING: Not enough classes with 2+ samples!")
        print(f"  Total classes: {len(labels_list)}")
        print(f"  Valid classes (2+ samples): {len(valid_labels)}")
        return None
    
    anchors = []
    positives = []
    negatives = []
    
    for _ in range(n_triplets):
        anchor_label = np.random.choice(valid_labels)

        anchor_idx, positive_idx = np.random.choice(
            label_to_indices[anchor_label], size=2, replace=False
        )

        negative_labels = [l for l in valid_labels if l != anchor_label]
        if len(negative_labels) == 0:
            negative_labels = [l for l in labels_list if l != anchor_label]
        
        negative_label = np.random.choice(negative_labels)
        negative_idx = np.random.choice(label_to_indices[negative_label])
        
        anchors.append(anchor_idx)
        positives.append(positive_idx)
        negatives.append(negative_idx)
    
    return {
        'anchors': np.array(anchors),
        'positives': np.array(positives),
        'negatives': np.array(negatives)
    }


# ============================================================================
# 4. SEMI-HARD TRIPLET MINING
# ============================================================================

def mine_semihard_triplets(embeddings, labels, margin=1.0):
    """
    Online semi-hard triplet mining within a batch.

    For each anchor-positive pair (i, j) with the same class:
      - Candidate negatives are all samples with a different class.
      - Semi-hard negatives satisfy: d(a,p) < d(a,n) < d(a,p) + margin
        (further than the positive but still within the loss margin).
      - If no semi-hard negative exists, falls back to the hardest negative
        (the one with the smallest distance to the anchor).

    Args:
        embeddings: (N, D) float tensor of L2-normalised projections (detached).
        labels:     (N,)  long tensor of class labels.
        margin:     float, triplet margin (should match the loss function).

    Returns:
        Tuple (anchors, positives, negatives) of 1-D LongTensors holding
        batch indices, or None if the batch contains no valid triplets.
    """
    n = embeddings.size(0)
    device = embeddings.device

    # Pairwise L2 distances — stable even for normalised vectors
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N, N, D)
    dists = diff.pow(2).sum(2).clamp(min=0).sqrt()            # (N, N)

    # Positive / negative masks
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)   # (N, N) bool
    eye = torch.eye(n, dtype=torch.bool, device=device)
    is_pos = same_class & ~eye   # same class, different sample
    is_neg = ~same_class         # different class

    anchors_list, positives_list, negatives_list = [], [], []

    for i in range(n):
        pos_indices = is_pos[i].nonzero(as_tuple=False).view(-1)
        neg_indices = is_neg[i].nonzero(as_tuple=False).view(-1)

        if pos_indices.numel() == 0 or neg_indices.numel() == 0:
            continue

        for j in pos_indices:
            d_ap = dists[i, j]
            d_an = dists[i, neg_indices]  # distances to all negatives

            # Semi-hard: d(a,p) < d(a,n) < d(a,p) + margin
            semihard_mask = (d_an > d_ap) & (d_an < d_ap + margin)
            candidates = neg_indices[semihard_mask]

            if candidates.numel() > 0:
                # Pick a random semi-hard negative
                pick = torch.randint(candidates.numel(), (1,)).item()
                neg = candidates[pick].item()
            else:
                # Fallback: hardest negative (smallest d(a,n))
                neg = neg_indices[d_an.argmin()].item()

            anchors_list.append(i)
            positives_list.append(j.item())
            negatives_list.append(neg)

    if not anchors_list:
        return None

    return (
        torch.tensor(anchors_list,   dtype=torch.long, device=device),
        torch.tensor(positives_list, dtype=torch.long, device=device),
        torch.tensor(negatives_list, dtype=torch.long, device=device),
    )


# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================

def train_contrastive_epoch(model, dataloader, optimizer, criterion, device, loss_type='triplet'):
    """Trains one epoch with the specified contrastive loss."""
    model.train()
    total_loss = 0
    
    for batch_data in dataloader:
        if loss_type == 'triplet':
            # Format: (anchor, positive, negative)
            anchor, positive, negative = batch_data
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            anchor_proj = model(anchor)
            positive_proj = model(positive)
            negative_proj = model(negative)
            
            # Triplet Loss
            loss = criterion(anchor_proj, positive_proj, negative_proj)
        
        else:  # simclr or supcon
            # Format: (x1, x2, label)
            x1, x2, labels = batch_data
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            proj1 = model(x1)
            proj2 = model(x2)

            # Stack both views along a new dimension
            features = torch.cat([proj1, proj2], dim=0)

            if loss_type == 'simclr':
                loss = criterion(features)
            elif loss_type == 'supcon':
                loss = criterion(features, labels)
            else:
                raise ValueError(f"Unknown loss type: '{loss_type}'")
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_contrastive_epoch_semihard(model, dataloader, optimizer, criterion, device, config):
    """
    Trains one contrastive epoch using online semi-hard triplet mining.

    Pipeline per batch:
      1. Forward pass — compute L2-normalised projections for all samples.
      2. Mine semi-hard triplets from the (detached) projections.
      3. Compute triplet loss on the mined (anchor, positive, negative) tuples.
      4. Backprop through the same projection graph (no second forward pass).

    Batches that yield no valid semi-hard triplets are silently skipped.
    """
    model.train()
    total_loss = 0.0
    n_batches_with_triplets = 0
    margin = config.get('triplet_margin', 1.0)

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Single forward pass — keep gradient graph
        projections = model(X_batch)  # (B, D), L2-normalised

        # Mine triplets from detached projections (no gradient needed for mining)
        triplet_indices = mine_semihard_triplets(
            projections.detach(), y_batch, margin=margin
        )

        if triplet_indices is None:
            continue

        a_idx, p_idx, n_idx = triplet_indices

        if a_idx.numel() == 0:
            continue

        # Triplet loss reuses the same projection graph
        loss = criterion(projections[a_idx], projections[p_idx], projections[n_idx])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches_with_triplets += 1

    # Guard against batches where no valid triplets existed
    if n_batches_with_triplets == 0:
        return 0.0

    return total_loss / n_batches_with_triplets


def train_classification_epoch(model, dataloader, optimizer, criterion, device):
    """Trains one epoch for classification fine-tuning."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate_classifier(model, dataloader, device):
    """Evaluates the classifier on a dataloader. Returns loss, accuracy, F1, predictions, labels."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def extract_embeddings(model, dataloader, device):
    """Extracts encoder embeddings for all samples in a dataloader."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            embeddings = model.encoder(X_batch)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.vstack(all_embeddings), np.array(all_labels)


# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

def plot_embeddings(embeddings, labels, idx_to_label, method='umap', save_path=None):
    """
    Visualizes embeddings using UMAP or t-SNE.
    """
    print(f"\nGenerating {method.upper()} visualization...")
    
    if method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
        embedding_2d = reducer.fit_transform(embeddings)
    elif method == 'tsne' and TSNE_AVAILABLE:
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=30)
        embedding_2d = reducer.fit_transform(embeddings)
    else:
        print(f"WARNING: {method.upper()} not available, skipping visualization")
        return
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[colors[idx]], label=idx_to_label[label],
                   alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    plt.title(f'Embedding Space Visualization - {method.upper()}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    plt.close()


def visualize_embeddings(model, X_test, y_test, idx_to_label, results_dir, fold_name="",
                         tsne_dir=None, method_name=""):
    """
    Generates UMAP and t-SNE visualizations of the embedding space.
    t-SNE images are saved to tsne_dir/{method_name}_{fold_name}.png when tsne_dir is set,
    otherwise fall back to results_dir.
    """
    print("\n" + "="*80)
    print("EMBEDDING SPACE VISUALIZATION")
    print("="*80)

    test_dataset = StandardDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    embeddings, labels = extract_embeddings(model, test_loader, DEVICE)
    print(f"Embeddings extracted: {embeddings.shape}")

    # Determine t-SNE output directory
    _tsne_dir = tsne_dir if tsne_dir else results_dir
    os.makedirs(_tsne_dir, exist_ok=True)
    prefix = f"{method_name}_" if method_name else ""

    # UMAP (saved alongside results, not in tsne_dir)
    if UMAP_AVAILABLE:
        umap_path = os.path.join(results_dir, f'embedding_umap{fold_name}.png')
        plot_embeddings(embeddings, labels, idx_to_label, method='umap', save_path=umap_path)

    # t-SNE
    if TSNE_AVAILABLE:
        tsne_suffix = fold_name.lstrip('_') if fold_name else 'Stratified'
        tsne_path = os.path.join(_tsne_dir, f'{prefix}{tsne_suffix}.png')
        plot_embeddings(embeddings, labels, idx_to_label, method='tsne', save_path=tsne_path)


# ============================================================================
# 6. PHASE 1: CONTRASTIVE PRE-TRAINING
# ============================================================================

def pretrain_contrastive(X_train, y_train, triplet_indices, config, fold_name=""):
    """
    Phase 1: Contrastive pre-training.

    Supports:
      - loss_type: triplet | simclr | supcon
      - mining_strategy: random (offline, pre-generated) | semihard (online, per-batch)

    When mining_strategy == 'semihard', triplet_indices is ignored and triplets
    are mined online from each batch using mine_semihard_triplets().
    """
    print("\n" + "="*80)
    print(f"PHASE 1: CONTRASTIVE PRE-TRAINING ({config['loss_type'].upper()}) {fold_name}")
    print("="*80)

    loss_type = config['loss_type']
    mining_strategy = config.get('mining_strategy', 'random')
    shift_prob = config.get('shift_prob', 0.0)
    shift_max_frac = config.get('shift_max_frac', 0.1)

    # Build the appropriate dataset depending on loss type and mining strategy
    if loss_type == 'triplet' and mining_strategy == 'semihard':
        # Online semi-hard mining: dataset yields (augmented_sample, label) pairs
        dataset = AugmentedDataset(X_train, y_train, augment=True,
                                   shift_prob=shift_prob, shift_max_frac=shift_max_frac)
    elif loss_type == 'triplet':
        # Random offline mining: dataset contains pre-generated (anchor, pos, neg) triplets
        dataset = TripletDataset(X_train, y_train, triplet_indices, augment=True,
                                 shift_prob=shift_prob, shift_max_frac=shift_max_frac)
    else:
        # SimCLR / SupCon: on-the-fly augmented pairs
        dataset = PairDataset(X_train, y_train, augment=True,
                              shift_prob=shift_prob, shift_max_frac=shift_max_frac)

    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=0)

    print(f"\nSamples: {len(dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Loss function: {loss_type}")
    print(f"Mining strategy: {mining_strategy}")

    # Model
    model = ContrastiveModel(
        input_channels=X_train.shape[2],
        projection_dim=config['projection_dim']
    ).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    # Loss function
    if loss_type == 'triplet':
        criterion = get_contrastive_loss('triplet', margin=config['triplet_margin'], p=2)
    elif loss_type == 'simclr':
        criterion = get_contrastive_loss('simclr', temperature=config.get('temperature', 0.5))
    elif loss_type == 'supcon':
        criterion = get_contrastive_loss('supcon', temperature=config.get('temperature', 0.5))
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'")

    criterion = criterion.to(DEVICE)

    print(f"\nPre-training ({config['pretrain_epochs']} epochs)...")
    start_time = time.time()

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['pretrain_epochs']):
        if loss_type == 'triplet' and mining_strategy == 'semihard':
            # Online semi-hard mining path
            epoch_loss = train_contrastive_epoch_semihard(
                model, dataloader, optimizer, criterion, DEVICE, config
            )
        else:
            # Random offline mining or SimCLR/SupCon
            epoch_loss = train_contrastive_epoch(
                model, dataloader, optimizer, criterion, DEVICE, loss_type=loss_type
            )

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{config['pretrain_epochs']}] - Loss: {epoch_loss:.4f}")

        # Early stopping on training loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['pretrain_patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    duration = time.time() - start_time
    print(f"\nPre-training completed in {duration:.2f}s")
    print(f"Best loss: {best_loss:.4f}")

    return model


# ============================================================================
# 7. PHASE 2: CLASSIFICATION FINE-TUNING
# ============================================================================

def finetune_classification(pretrained_model, X_train, y_train, X_val, y_val,
                           X_test, y_test, num_classes, config, fold_name=""):
    """
    Phase 2: Supervised fine-tuning for classification.
    """
    print("\n" + "="*80)
    print(f"PHASE 2: FINE-TUNING CLASSIFICATION {fold_name}")
    print("="*80)
    
    # Build classification model
    classifier = ClassificationModel(pretrained_model.encoder, num_classes).to(DEVICE)
    
    # Datasets
    train_dataset = StandardDataset(X_train, y_train)
    val_dataset = StandardDataset(X_val, y_val)
    test_dataset = StandardDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0)

    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config['finetune_lr'],
                                weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nFine-tuning ({config['finetune_epochs']} epochs)...")
    start_time = time.time()
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['finetune_epochs']):
        # Train
        train_loss, train_acc = train_classification_epoch(
            classifier, train_loader, optimizer, criterion, DEVICE
        )
        
        # Validation
        val_loss, val_acc, val_f1, _, _ = evaluate_classifier(classifier, val_loader, DEVICE)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{config['finetune_epochs']}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = classifier.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['finetune_patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    duration = time.time() - start_time
    print(f"\nFine-tuning completed in {duration:.2f}s")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    
    # Load the best model weights
    if best_model_state:
        classifier.load_state_dict(best_model_state)
    
    # Evaluation finale sur test
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate_classifier(
        classifier, test_loader, DEVICE
    )
    
    print(f"\nTest results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    
    return classifier, test_acc, test_f1, test_preds, test_labels


# ============================================================================
# 8. REPORT GENERATION
# ============================================================================

def generate_reports(test_preds, test_labels, idx_to_label, results_dir,
                    strategy, fold_name=""):
    """Generates classification reports and confusion matrices.
    Classification reports go to results_dir/classification_report/.
    Confusion matrices remain in results_dir/.
    """
    # Classification reports in dedicated subfolder
    report_dir = os.path.join(results_dir, 'classification_report')
    os.makedirs(report_dir, exist_ok=True)

    report = classification_report(test_labels, test_preds,
                                   target_names=[idx_to_label[i] for i in range(len(idx_to_label))],
                                   digits=2)
    report_path = os.path.join(report_dir,
                               f'classification_report_Contrastive_{strategy}{fold_name}.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Confusion matrix stays in results_dir
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[idx_to_label[i] for i in range(len(idx_to_label))],
               yticklabels=[idx_to_label[i] for i in range(len(idx_to_label))])
    plt.title(f'Confusion Matrix - Contrastive {strategy}{fold_name}', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = os.path.join(results_dir,
                           f'confusion_matrix_Contrastive_{strategy}{fold_name}.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 9. TRAINING STRATEGIES
# ============================================================================

def run_stratified_split(X, y, participants, idx_to_label, config):
    """
    Strategy 1: Stratified split 70/10/20
    """
    data_fraction = config.get('data_fraction', 1.0)

    print("\n" + "="*80)
    print(f"STRATEGY 1: STRATIFIED SPLIT (70/10/20) - {data_fraction*100:.0f}% of data")
    print("="*80)
    
    # Train/temp split
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=0.125, stratify=y_train_temp, random_state=SEED
    )
    
    # Reduce training data if needed
    if data_fraction < 1.0:
        n_train_reduced = int(len(X_train) * data_fraction)
        print(f"\nReducing training data: {len(X_train)} -> {n_train_reduced} ({data_fraction*100:.0f}%)")

        # Stratified sub-sampling
        from sklearn.model_selection import train_test_split as split_subset
        X_train, _, y_train, _ = split_subset(
            X_train, y_train, train_size=data_fraction, stratify=y_train, random_state=SEED
        )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Pre-generate triplets only for random mining (semihard mines online per batch)
    if config.get('mining_strategy', 'random') == 'semihard':
        triplet_indices = None
        print("Mining strategy: semi-hard (online, no pre-generation needed)")
    else:
        n_triplets = len(X_train) * 5
        triplet_indices = generate_triplets_for_subset(y_train, n_triplets)
        print(f"Triplets generated: {len(triplet_indices['anchors'])}")

    # Phase 1: Pre-training
    contrastive_model = pretrain_contrastive(X_train, y_train, triplet_indices, config)

    # Phase 2: Fine-tuning
    classifier, test_acc, test_f1, test_preds, test_labels = finetune_classification(
        contrastive_model, X_train, y_train, X_val, y_val, X_test, y_test,
        num_classes=len(idx_to_label), config=config
    )

    generate_reports(test_preds, test_labels, idx_to_label,
                    config['results_dir'], 'Stratified')

    visualize_embeddings(classifier, X_test, y_test, idx_to_label,
                        config['results_dir'], fold_name="_Stratified",
                        tsne_dir=config.get('tsne_dir'),
                        method_name=config.get('method_name', ''))
    
    result = {
        'Strategy': 'Stratified',
        'Model': 'Contrastive_SDCNet',
        'Fold': 'single',
        'Test_Acc': test_acc,
        'Test_F1': test_f1
    }
    
    return [result]


def run_loso_cross_validation(X, y, participants, idx_to_label, config):
    """
    Strategy 2: Leave-One-Subject-Out Cross-Validation
    """
    data_fraction = config.get('data_fraction', 1.0)

    print("\n" + "="*80)
    print(f"STRATEGY 2: LOSO (LEAVE-ONE-SUBJECT-OUT) CROSS-VALIDATION - {data_fraction*100:.0f}% of data")
    print("="*80)

    unique_participants = sorted(np.unique(participants))
    print(f"Participants: {len(unique_participants)}")
    print(f"Participant list: {unique_participants}")
    
    results = []
    
    for fold_idx, test_participant in enumerate(unique_participants):
        print(f"\n--- FOLD {fold_idx+1}/{len(unique_participants)} ---")
        print(f"Participant TEST: {test_participant}")
        
        # Masques
        mask_test = participants == test_participant
        mask_train_temp = ~mask_test
        
        X_train_temp = X[mask_train_temp]
        y_train_temp = y[mask_train_temp]
        X_test = X[mask_test]
        y_test = y[mask_test]
        
        if len(X_test) == 0:
            print("WARNING: No test samples for this fold, skipping")
            continue
        
        # Split train/val (80/20 du train_temp)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp, y_train_temp, test_size=0.2, stratify=y_train_temp, random_state=SEED
        )
        
# Reduce training data if needed
        if data_fraction < 1.0:
            n_train_reduced = int(len(X_train) * data_fraction)
            print(f"  Reducing train: {len(X_train)} -> {n_train_reduced}")

            from sklearn.model_selection import train_test_split as split_subset
            X_train, _, y_train, _ = split_subset(
                X_train, y_train, train_size=data_fraction, stratify=y_train, random_state=SEED + fold_idx
            )

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Pre-generate triplets only for random mining
        if config.get('mining_strategy', 'random') == 'semihard':
            triplet_indices = None
        else:
            n_triplets = len(X_train) * 3
            triplet_indices = generate_triplets_for_subset(y_train, n_triplets)
            if triplet_indices is None:
                continue

        # Phase 1: Pre-training
        contrastive_model = pretrain_contrastive(X_train, y_train, triplet_indices,
                                                config, fold_name=f"(Fold {fold_idx+1})")

        # Phase 2: Fine-tuning
        classifier, test_acc, test_f1, test_preds, test_labels = finetune_classification(
            contrastive_model, X_train, y_train, X_val, y_val, X_test, y_test,
            num_classes=len(idx_to_label), config=config, fold_name=f"(Fold {fold_idx+1})"
        )

        generate_reports(test_preds, test_labels, idx_to_label,
                        config['results_dir'], 'LOSO', fold_name=f"_fold{fold_idx+1}")
        
        result = {
            'Strategy': 'LOSO',
            'Model': 'Contrastive_SDCNet',
            'Fold': fold_idx+1,
            'Test_Acc': test_acc,
            'Test_F1': test_f1,
            'Test_Participant': test_participant
        }
        results.append(result)
    
    # Visualization on the last fold
    if results:
        visualize_embeddings(classifier, X_test, y_test, idx_to_label,
                            config['results_dir'], fold_name="_LOSO_last_fold",
                            tsne_dir=config.get('tsne_dir'),
                            method_name=config.get('method_name', ''))
    
    # Resultats agreges
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "="*80)
        print("LOSO AGGREGATED RESULTS")
        print("="*80)
        print(f"\nContrastive_SDCNet:")
        print(f"  Accuracy: {df_results['Test_Acc'].mean():.4f} +/- {df_results['Test_Acc'].std():.4f}")
        print(f"  F1-Score: {df_results['Test_F1'].mean():.4f} +/- {df_results['Test_F1'].std():.4f}")
    
    return results


def run_logo_cross_validation(X, y, participants, idx_to_label, config, n_folds=5):
    """
    Strategy 3: Leave-One-Group-Out Cross-Validation
    """
    data_fraction = config.get('data_fraction', 1.0)

    print("\n" + "="*80)
    print(f"STRATEGY 3: LOGO (LEAVE-ONE-GROUP-OUT) {n_folds}-FOLD CV - {data_fraction*100:.0f}% of data")
    print("="*80)

    unique_participants = sorted(np.unique(participants))
    n_participants = len(unique_participants)
    print(f"Participants: {n_participants}")
    
    # Define test groups (3 participants per fold)
    np.random.seed(SEED)
    shuffled_participants = np.random.permutation(unique_participants)
    
    results = []
    
    for fold_idx in range(n_folds):
        print(f"\n--- FOLD {fold_idx+1}/{n_folds} ---")
        
        # Select 3 participants for test, 2 for validation
        start_test = (fold_idx * 3) % n_participants
        test_participants = [shuffled_participants[(start_test + i) % n_participants] 
                           for i in range(3)]
        
        start_val = (start_test + 3) % n_participants
        val_participants = [shuffled_participants[(start_val + i) % n_participants] 
                          for i in range(2)]
        
        train_participants = [p for p in unique_participants 
                            if p not in test_participants and p not in val_participants]
        
        print(f"Participants TEST: {test_participants}")
        print(f"Participants VAL: {val_participants}")
        
        # Create split masks
        mask_train = np.isin(participants, train_participants)
        mask_val = np.isin(participants, val_participants)
        mask_test = np.isin(participants, test_participants)
        
        X_train = X[mask_train]
        y_train = y[mask_train]
        X_val = X[mask_val]
        y_val = y[mask_val]
        X_test = X[mask_test]
        y_test = y[mask_test]
        if data_fraction < 1.0:
            n_train_reduced = int(len(X_train) * data_fraction)
            print(f"Reducing train: {len(X_train)} -> {n_train_reduced}")

            from sklearn.model_selection import train_test_split as split_subset
            X_train, _, y_train, _ = split_subset(
                X_train, y_train, train_size=data_fraction, stratify=y_train, random_state=SEED + fold_idx
            )

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print("WARNING: One split is empty, skipping this fold")
            continue

        # Pre-generate triplets only for random mining
        if config.get('mining_strategy', 'random') == 'semihard':
            triplet_indices = None
        else:
            n_triplets = len(X_train) * 3
            triplet_indices = generate_triplets_for_subset(y_train, n_triplets)
            if triplet_indices is None:
                continue

        # Phase 1: Pre-training
        contrastive_model = pretrain_contrastive(X_train, y_train, triplet_indices,
                                                config, fold_name=f"(Fold {fold_idx+1})")

        # Phase 2: Fine-tuning
        classifier, test_acc, test_f1, test_preds, test_labels = finetune_classification(
            contrastive_model, X_train, y_train, X_val, y_val, X_test, y_test,
            num_classes=len(idx_to_label), config=config, fold_name=f"(Fold {fold_idx+1})"
        )

        generate_reports(test_preds, test_labels, idx_to_label,
                        config['results_dir'], 'LOGO', fold_name=f"_fold{fold_idx+1}")
        
        result = {
            'Strategy': 'LOGO',
            'Model': 'Contrastive_SDCNet',
            'Fold': fold_idx+1,
            'Test_Acc': test_acc,
            'Test_F1': test_f1,
            'Test_Participants': str(test_participants)
        }
        results.append(result)
    
    # Visualization on the last fold
    if results:
        visualize_embeddings(classifier, X_test, y_test, idx_to_label,
                            config['results_dir'], fold_name="_LOGO_last_fold",
                            tsne_dir=config.get('tsne_dir'),
                            method_name=config.get('method_name', ''))
    
    # Resultats agreges
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "="*80)
        print("LOGO AGGREGATED RESULTS")
        print("="*80)
        print(f"\nContrastive_SDCNet:")
        print(f"  Accuracy: {df_results['Test_Acc'].mean():.4f} +/- {df_results['Test_Acc'].std():.4f}")
        print(f"  F1-Score: {df_results['Test_F1'].mean():.4f} +/- {df_results['Test_F1'].std():.4f}")
    
    return results


# ============================================================================
# 10. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Contrastive Learning with Multiple Loss Functions')
    parser.add_argument('--data-dir', type=str,
                       default=os.environ.get('HAR_CONTRASTIVE_DATA_DIR', 'contrastive_data'),
                       help='Directory containing preprocessed data (default: HAR_CONTRASTIVE_DATA_DIR or contrastive_data)')
    parser.add_argument('--results-base-dir', type=str,
                       default=os.environ.get('HAR_RESULTS_BASE_DIR', '.'),
                       help='Root directory for results_contrastive_* outputs (default: HAR_RESULTS_BASE_DIR or .)')
    parser.add_argument('--checkpoints-base-dir', type=str,
                       default=os.environ.get('HAR_CHECKPOINTS_BASE_DIR', '.'),
                       help='Root directory for checkpoints_contrastive_* (default: HAR_CHECKPOINTS_BASE_DIR or .)')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                       help='Fraction of training data to use (default: 1.0 = 100%%)')
    parser.add_argument('--loss-type', type=str, default='triplet',
                       choices=['triplet', 'simclr', 'supcon'],
                       help='Contrastive loss type: triplet, simclr, supcon (default: triplet)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for SimCLR/SupCon (default: 0.5)')
    parser.add_argument('--pretrain-epochs', type=int,
                       default=int(os.environ.get('HAR_PRETRAIN_EPOCHS', 50)),
                       help='Number of contrastive pre-training epochs (default: 50)')
    parser.add_argument('--pretrain-patience', type=int,
                       default=int(os.environ.get('HAR_PRETRAIN_PATIENCE', 10)),
                       help='Early stopping patience for pre-training (default: 10)')
    parser.add_argument('--finetune-epochs', type=int,
                       default=int(os.environ.get('HAR_FINETUNE_EPOCHS', 100)),
                       help='Number of fine-tuning epochs (default: 100)')
    parser.add_argument('--finetune-patience', type=int,
                       default=int(os.environ.get('HAR_FINETUNE_PATIENCE', 15)),
                       help='Early stopping patience for fine-tuning (default: 15)')
    parser.add_argument('--shift-prob', type=float,
                       default=float(os.environ.get('HAR_SHIFT_PROB', '0.0')),
                       help='Probability of applying temporal shifting (0.0 = disabled, default: 0.0)')
    parser.add_argument('--shift-max-frac', type=float,
                       default=float(os.environ.get('HAR_SHIFT_MAX_FRAC', '0.1')),
                       help='Max shift as a fraction of sequence length (default: 0.1 = 10%%)')
    parser.add_argument('--mining-strategy', type=str,
                       default=os.environ.get('HAR_MINING_STRATEGY', 'random'),
                       choices=['random', 'semihard'],
                       help='Triplet mining strategy: random (offline pre-generation) or '
                            'semihard (online per-batch, default: random)')
    parser.add_argument('--strategies', type=str, default='all',
                       help='Strategies to run: all, stratified, loso, logo, or comma-separated list')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Explicit results directory (overrides auto-naming from --results-base-dir)')
    parser.add_argument('--method-name', type=str, default=None,
                       help='Method label used in CSV filename and t-SNE naming (e.g. random_shift)')
    parser.add_argument('--tsne-dir', type=str, default=None,
                       help='Directory for t-SNE images (defaults to results_dir if not set)')
    parser.add_argument('--checkpoints-dir', type=str, default=None,
                       help='Explicit checkpoints directory (overrides auto-naming from --checkpoints-base-dir)')
    args = parser.parse_args()

    if args.data_fraction <= 0.0 or args.data_fraction > 1.0:
        raise ValueError("--data-fraction must be in the range (0, 1]")
    if args.pretrain_epochs <= 0 or args.finetune_epochs <= 0:
        raise ValueError("--pretrain-epochs and --finetune-epochs must be > 0")
    if args.pretrain_patience <= 0 or args.finetune_patience <= 0:
        raise ValueError("--pretrain-patience and --finetune-patience must be > 0")

    # Parse strategies
    valid_strategies = ['stratified', 'loso', 'logo']
    if args.strategies.lower() == 'all':
        strategies_to_run = valid_strategies
    else:
        requested = [s.strip().lower() for s in args.strategies.split(',') if s.strip()]
        if not requested:
            raise ValueError("No valid strategy provided via --strategies")

        invalid = [s for s in requested if s not in valid_strategies]
        if invalid:
            raise ValueError(
                f"Invalid strategies: {invalid}. Allowed values: {valid_strategies}"
            )

        # Keep canonical execution order
        strategies_to_run = [s for s in valid_strategies if s in requested]
    
    # Build directory suffix — each combination of options gets its own directory
    if args.data_fraction == 1.0:
        dir_suffix = f'{args.loss_type}'
    else:
        dir_suffix = f'{args.loss_type}_{int(args.data_fraction*100)}pct'
    # Append mining strategy suffix (only non-default value changes the name)
    if args.mining_strategy == 'semihard':
        dir_suffix += '_semihard'
    # Append shift suffix to avoid overwriting results without shifting
    if args.shift_prob > 0.0:
        dir_suffix += '_shift'
    
    # Resolve explicit-or-auto paths
    resolved_results_dir = args.results_dir if args.results_dir else os.path.join(args.results_base_dir, f'results_contrastive_{dir_suffix}')
    resolved_checkpoint_dir = args.checkpoints_dir if args.checkpoints_dir else os.path.join(args.checkpoints_base_dir, f'checkpoints_contrastive_{dir_suffix}')
    resolved_method_name = args.method_name if args.method_name else dir_suffix
    resolved_tsne_dir = args.tsne_dir  # None means: fall back to results_dir inside visualize_embeddings

    # Configuration
    config = {
        'data_dir': args.data_dir,
        'results_dir': resolved_results_dir,
        'checkpoint_dir': resolved_checkpoint_dir,
        'method_name': resolved_method_name,
        'tsne_dir': resolved_tsne_dir,
        
        # Data
        'data_fraction': args.data_fraction,
        
        # Loss type
        'loss_type': args.loss_type,
        'temperature': args.temperature,  # Pour SimCLR/SupCon
        
        # Contrastive learning
        'projection_dim': 256,
        'triplet_margin': 1.0,
        'mining_strategy': args.mining_strategy,
        'shift_prob': args.shift_prob,
        'shift_max_frac': args.shift_max_frac,
        'pretrain_epochs': args.pretrain_epochs,
        'pretrain_patience': args.pretrain_patience,
        'lr': 0.001,
        
        # Fine-tuning
        'finetune_epochs': args.finetune_epochs,
        'finetune_patience': args.finetune_patience,
        'finetune_lr': 0.0001,
        'batch_size': 32,
        'weight_decay': 1e-4,
    }
    
    # Create output directories
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    print(f"Configuration: {args.data_fraction*100:.0f}% of training data")
    print(f"Loss function: {args.loss_type.upper()}")
    print(f"Mining strategy: {args.mining_strategy.upper()}")
    print(f"Strategies: {', '.join(strategies_to_run)}")
    if args.shift_prob > 0.0:
        print(f"Temporal shifting: ENABLED (prob={args.shift_prob}, max_frac={args.shift_max_frac})")
    else:
        print("Temporal shifting: DISABLED (use --shift-prob to enable)")
    print(f"Pretrain epochs/patience: {args.pretrain_epochs}/{args.pretrain_patience}")
    print(f"Finetune epochs/patience: {args.finetune_epochs}/{args.finetune_patience}")
    print(f"Data dir: {config['data_dir']}")
    print(f"Results dir: {config['results_dir']}")
    print(f"Checkpoints dir: {config['checkpoint_dir']}")
    if args.loss_type in ['simclr', 'supcon']:
        print(f"Temperature: {args.temperature}")
    
    with open(os.path.join(config['data_dir'], 'preprocessed_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    participants = data['participants']
    idx_to_label = data['idx_to_label']
    
    print(f"Data loaded:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Number of classes: {len(idx_to_label)}")
    print(f"  Number of participants: {len(np.unique(participants))}")
    
    all_results = []

    if 'stratified' in strategies_to_run:
        results_stratified = run_stratified_split(X, y, participants, idx_to_label, config)
        all_results.extend(results_stratified)

    if 'loso' in strategies_to_run:
        results_loso = run_loso_cross_validation(X, y, participants, idx_to_label, config)
        all_results.extend(results_loso)

    if 'logo' in strategies_to_run:
        results_logo = run_logo_cross_validation(X, y, participants, idx_to_label, config, n_folds=5)
        all_results.extend(results_logo)
    
    df_all_results = pd.DataFrame(all_results)
    results_path = os.path.join(config['results_dir'], f"all_results_{config.get('method_name', 'contrastive')}.csv")
    df_all_results.to_csv(results_path, index=False)
    
    print("\n" + "="*80)
    print("CONTRASTIVE TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved: {results_path}")
    print(f"Visualizations available in: {config['results_dir']}/")
    print(f"\nTotal training runs: {len(all_results)}")


if __name__ == '__main__':
    main()
