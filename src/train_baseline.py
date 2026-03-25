#!/usr/bin/env python3
"""
Baseline classification training for IMU/PPG sensor data.
Trains all models on all split strategies with full cross-validation.
"""

import os
import argparse
import glob
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Visualisation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster/server use
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_csv_files(data_path):
    """Loads all CSV files from a directory."""
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    dataframes = {}
    for file in csv_files:
        file_name = os.path.basename(file)
        try:
            df = pd.read_csv(file)
            dataframes[file_name] = df
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    print(f"{len(dataframes)} files loaded successfully")
    return dataframes


def create_segments(dataframes, window_size=405):
    """
    Segments data into 3-second windows (405 samples)
    starting from annotation points (TAG column).
    """
    X = []
    y = []
    groups = []
    
    feature_cols = ['ACC_X', 'ACC_Y', 'ACC_Z', 'PPG_I', 'PPG_II', 'PPG_III']
    
    for filename, df in dataframes.items():
        participant_id = filename.split('_')[1]
        tagged_indices = df[df['TAG'].notna()].index
        
        for start_idx in tagged_indices:
            end_idx = start_idx + window_size
            
            if end_idx <= len(df):
                segment = df.iloc[start_idx:end_idx][feature_cols].values
                label = df.iloc[start_idx]['TAG']
                
                if not np.isnan(segment).any():
                    X.append(segment)
                    y.append(label)
                    groups.append(participant_id)
    
    return np.array(X), np.array(y), np.array(groups)


def normalize_data(X_raw):
    """Normalizes data with StandardScaler (Z-Score)."""
    N, T, F = X_raw.shape
    X_flat = X_raw.reshape(N * T, F)

    scaler = StandardScaler()
    X_norm_flat = scaler.fit_transform(X_flat)
    X_final = X_norm_flat.reshape(N, T, F)

    print(f"Mean after normalization: {np.mean(X_final):.4f}")
    print(f"Std after normalization:  {np.std(X_final):.4f}")

    return X_final, scaler


def encode_labels(y_raw):
    """Encodes string labels to integers."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    print(f"Classes: {label_encoder.classes_}")
    return y_encoded, label_encoder


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_jitter(x, sigma=0.03):
    """Adds Gaussian noise."""
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + noise


def augment_scaling(x, sigma=0.1):
    """Applies random amplitude scaling."""
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[2]))
    return x * factor[:, np.newaxis, :]


def augment_permutation(x, max_segments=5, seg_mode="equal"):
    """Splits and randomly shuffles temporal segments."""
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            
            splits = list(splits)
            np.random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


class AugmentedTimeSeriesDataset(Dataset):
    """PyTorch dataset with on-the-fly data augmentation."""
    def __init__(self, X, y, augment=True, aug_prob=0.8):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.aug_prob = aug_prob
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].numpy()
        y = self.y[idx]
        
        if self.augment and np.random.rand() < self.aug_prob:
            augmentations = []
            
            if np.random.rand() < 0.5:
                augmentations.append(lambda t: augment_jitter(t, sigma=0.03))
            if np.random.rand() < 0.5:
                augmentations.append(lambda t: augment_scaling(t, sigma=0.1))
            if np.random.rand() < 0.3:
                augmentations.append(lambda t: augment_permutation(t, max_segments=4))
            
            x = x[np.newaxis, :, :]
            for aug_fn in augmentations:
                x = aug_fn(x)
            x = x[0]
        
        return torch.FloatTensor(x), y


# ============================================================================
# MODELS
# ============================================================================

class CNN1D_SanityCheck(nn.Module):
    """Standard 1D CNN baseline."""
    def __init__(self, input_channels=6, num_classes=8, seq_length=405):
        super(CNN1D_SanityCheck, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


class SDCBlock(nn.Module):
    """Spatial Dilated Convolution Block with residual connection."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rate=1, dropout=0.3):
        super(SDCBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation_rate // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=padding
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        if self.use_residual:
            out = out + residual
        else:
            residual = self.residual_conv(residual)
            out = out + residual
            
        return out


class SDCNet(nn.Module):
    """SDC-Net: state-of-the-art backbone with Spatial Dilated Convolutions."""
    def __init__(self, input_channels=6, num_classes=8, seq_length=405):
        super(SDCNet, self).__init__()
        
        self.input_conv = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=3)
        self.input_bn = nn.BatchNorm1d(64)
        
        self.sdc_block1 = SDCBlock(64, 64, kernel_size=3, dilation_rate=1, dropout=0.2)
        self.sdc_block2 = SDCBlock(64, 128, kernel_size=3, dilation_rate=2, dropout=0.2)
        self.sdc_block3 = SDCBlock(128, 128, kernel_size=3, dilation_rate=4, dropout=0.3)
        self.sdc_block4 = SDCBlock(128, 256, kernel_size=3, dilation_rate=8, dropout=0.3)
        self.sdc_block5 = SDCBlock(256, 256, kernel_size=3, dilation_rate=16, dropout=0.4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.gelu(x)
        
        x = self.sdc_block1(x)
        x = self.sdc_block2(x)
        x = self.sdc_block3(x)
        x = self.sdc_block4(x)
        x = self.sdc_block5(x)
        
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


class DNN(nn.Module):
    """Deep Neural Network - fully connected architecture."""
    def __init__(self, input_channels=6, num_classes=8, seq_length=405):
        super(DNN, self).__init__()

        # Flatten input: (batch, seq_length, channels) -> (batch, seq_length * channels)
        self.input_size = seq_length * input_channels
        self.fc1 = nn.Linear(self.input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.4)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc5 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Flatten: (batch, seq_length, channels) -> (batch, seq_length * channels)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.fc5(x)

        return x


class LSTMNet(nn.Module):
    """Bidirectional stacked LSTM for time series."""
    def __init__(self, input_channels=6, num_classes=8, seq_length=405):
        super(LSTMNet, self).__init__()

        # Stacked bidirectional LSTM
        self.lstm1 = nn.LSTM(
            input_size=input_channels,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.lstm2 = nn.LSTM(
            input_size=256,  # 128 * 2 (bidirectional)
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc1 = nn.Linear(128, 64)  # 64 * 2 (bidirectional)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take the last temporal output

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerNet(nn.Module):
    """Transformer encoder for time series classification."""
    def __init__(self, input_channels=6, num_classes=8, seq_length=405, d_model=64, nhead=4, num_layers=3):
        super(TransformerNet, self).__init__()

        # Project input features to d_model dimension
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.3,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.fc1 = nn.Linear(d_model, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.input_projection(x)  # (batch, seq_length, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, seq_length, d_model)

        # Global average pooling over the temporal dimension
        x = x.transpose(1, 2)  # (batch, d_model, seq_length)
        x = self.global_avg_pool(x)  # (batch, d_model, 1)
        x = x.squeeze(-1)  # (batch, d_model)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                       batch_size=32, augment_train=True, aug_prob=0.8):
    """Creates PyTorch DataLoaders with optional data augmentation."""
    train_dataset = AugmentedTimeSeriesDataset(X_train, y_train, 
                                               augment=augment_train, 
                                               aug_prob=aug_prob)
    val_dataset = AugmentedTimeSeriesDataset(X_val, y_val, augment=False)
    test_dataset = AugmentedTimeSeriesDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device):
    """Evaluates the model on a dataset."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001,
                weight_decay=1e-4, device='cpu', model_name='Model',
                save_dir='checkpoints'):
    """Full model training with early stopping."""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_best.pth")
            torch.save(best_model_state, model_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    
    elapsed_time = time.time() - start_time
    print(f"  Training completed in {elapsed_time:.2f}s")
    print(f"  Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")

    return model, history


def plot_learning_curves(history, model_name, save_path):
    """Plots training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# SPLIT STRATEGIES
# ============================================================================

def run_stratified_split(X, y, num_classes, label_encoder, device, config):
    """Runs a standard stratified split (70/10/20)."""
    data_fraction = config.get('data_fraction', 1.0)
    print("\n" + "="*80)
    print(f"STRATEGY 1: STRATIFIED SPLIT - {data_fraction*100:.0f}% of data")
    print("="*80)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=42, 
        stratify=y_train_val
    )

    if data_fraction < 1.0:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=data_fraction, stratify=y_train, random_state=SEED
        )
        print(f"Training data reduced to {len(X_train)} samples ({data_fraction*100:.0f}%)")
    
    print(f"Train: {len(X_train)} ({len(X_train)/len(X):.1%})")
    print(f"Val:   {len(X_val)} ({len(X_val)/len(X):.1%})")
    print(f"Test:  {len(X_test)} ({len(X_test)/len(X):.1%})")

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config['batch_size']
    )

    results = []

    all_models = [
        ('DNN', DNN),
        ('LSTM', LSTMNet),
        ('Transformer', TransformerNet),
        ('SDCNet', SDCNet)
    ]
    models_to_train = [m for m in all_models if not config.get('sdcnet_only') or m[0] == 'SDCNet']

    for model_name, model_class in models_to_train:
        print(f"\n--- Training {model_name} ---")
        
        model = model_class(input_channels=6, num_classes=num_classes, seq_length=405)
        
        model_trained, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            device=device,
            model_name=f"{model_name}_Stratified",
            save_dir=config['checkpoints_dir']
        )
        
        test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
            model_trained, test_loader, nn.CrossEntropyLoss(), device
        )
        
        print(f"\nTest results - {model_name}:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")

        plot_learning_curves(
            history, f"{model_name} (Stratified)",
            os.path.join(config['results_dir'], f"learning_curves_{model_name}_Stratified.png")
        )
        
        plot_confusion_matrix(
            test_labels, test_preds, label_encoder.classes_,
            f"{model_name} (Stratified)", 
            os.path.join(config['results_dir'], f"confusion_matrix_{model_name}_Stratified.png")
        )
        
        report = classification_report(test_labels, test_preds, 
                                      target_names=label_encoder.classes_)
        report_dir = os.path.join(config['results_dir'], 'classification_report')
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir,
                              f"classification_report_{model_name}_Stratified.txt"), 'w') as f:
            f.write(report)
        
        results.append({
            'Strategy': 'Stratified',
            'Model': model_name,
            'Fold': 'single',
            'Test_Acc': test_acc,
            'Test_F1': test_f1,
            'Test_Loss': test_loss
        })
    
    return results


def run_loso_cross_validation(X, y, groups, num_classes, label_encoder, device, config):
    """Runs a full Leave-One-Subject-Out cross-validation."""
    data_fraction = config.get('data_fraction', 1.0)
    print("\n" + "="*80)
    print(f"STRATEGY 2: LOSO (LEAVE-ONE-SUBJECT-OUT) CROSS-VALIDATION - {data_fraction*100:.0f}% of data")
    print("="*80)

    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)
    print(f"Participants: {n_participants}")
    print(f"Cross-validation folds: {n_participants}")
    
    results = []
    
    for fold_idx, test_participant in enumerate(unique_participants):
        print(f"\n--- FOLD {fold_idx+1}/{n_participants} ---")
        print(f"Participant TEST: {test_participant}")
        
        # Validation participant: next one in the cycle
        val_participant = unique_participants[(fold_idx + 1) % n_participants]
        print(f"Participant VAL: {val_participant}")
        
        # Create split masks
        mask_test = groups == test_participant
        mask_val = groups == val_participant
        mask_train = ~(mask_test | mask_val)
        
        X_train = X[mask_train]
        y_train = y[mask_train]
        X_val = X[mask_val]
        y_val = y[mask_val]
        X_test = X[mask_test]
        y_test = y[mask_test]

        if data_fraction < 1.0:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=data_fraction, stratify=y_train, random_state=SEED + fold_idx
            )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=config['batch_size']
        )

        all_models = [
            ('DNN', DNN),
            ('LSTM', LSTMNet),
            ('Transformer', TransformerNet),
            ('SDCNet', SDCNet)
        ]
        models_to_train = [m for m in all_models if not config.get('sdcnet_only') or m[0] == 'SDCNet']

        for model_name, model_class in models_to_train:
            print(f"\n  Training {model_name} (Fold {fold_idx+1})...")
            
            model = model_class(input_channels=6, num_classes=num_classes, seq_length=405)
            
            model_trained, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config['num_epochs'],
                lr=config['lr'],
                weight_decay=config['weight_decay'],
                device=device,
                model_name=f"{model_name}_LOSO_fold{fold_idx+1}",
                save_dir=config['checkpoints_dir']
            )
            
            test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
                model_trained, test_loader, nn.CrossEntropyLoss(), device
            )
            
            print(f"  Test results - {model_name} (Fold {fold_idx+1}):")
            print(f"    Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
            
            plot_learning_curves(
                history, f"{model_name} LOSO Fold{fold_idx+1}", 
                os.path.join(config['results_dir'], 
                           f"learning_curves_{model_name}_LOSO_fold{fold_idx+1}.png")
            )
            
            plot_confusion_matrix(
                test_labels, test_preds, label_encoder.classes_,
                f"{model_name} LOSO Fold{fold_idx+1}", 
                os.path.join(config['results_dir'], 
                           f"confusion_matrix_{model_name}_LOSO_fold{fold_idx+1}.png")
            )
            
            report = classification_report(test_labels, test_preds, 
                                          target_names=label_encoder.classes_)
            report_dir = os.path.join(config['results_dir'], 'classification_report')
            os.makedirs(report_dir, exist_ok=True)
            with open(os.path.join(report_dir,
                                  f"classification_report_{model_name}_LOSO_fold{fold_idx+1}.txt"), 'w') as f:
                f.write(report)
            
            results.append({
                'Strategy': 'LOSO',
                'Model': model_name,
                'Fold': fold_idx+1,
                'Test_Participant': test_participant,
                'Test_Acc': test_acc,
                'Test_F1': test_f1,
                'Test_Loss': test_loss
            })
    
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("LOSO AGGREGATED RESULTS")
    print("="*80)
    for model_name in ['CNN1D', 'SDCNet', 'DNN', 'LSTM', 'Transformer']:
        model_results = df_results[df_results['Model'] == model_name]
        mean_acc = model_results['Test_Acc'].mean()
        std_acc = model_results['Test_Acc'].std()
        mean_f1 = model_results['Test_F1'].mean()
        std_f1 = model_results['Test_F1'].std()

        print(f"\n{model_name}:")
        print(f"  Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  F1-Score: {mean_f1:.4f} +/- {std_f1:.4f}")

    df_results.to_csv(os.path.join(config['results_dir'], 'LOSO_cross_validation_results.csv'),
                     index=False)

    return results


def run_logo_cross_validation(X, y, groups, num_classes, label_encoder, device, config):
    """Runs a Leave-One-Group-Out cross-validation with groups of 3 test participants."""
    data_fraction = config.get('data_fraction', 1.0)
    n_folds = 5
    print("\n" + "="*80)
    print(f"STRATEGY 3: LOGO (LEAVE-ONE-GROUP-OUT) {n_folds}-FOLD CV - {data_fraction*100:.0f}% of data")
    print("="*80)

    unique_participants = sorted(np.unique(groups))
    n_participants = len(unique_participants)

    if n_participants < 5:
        print(f"WARNING: Not enough participants ({n_participants}) for LOGO")
        return []

    # Same non-overlapping seed-42 permutation as train_contrastive_model.py
    np.random.seed(SEED)
    shuffled_participants = np.random.permutation(unique_participants)

    print(f"Participants: {n_participants}")
    print(f"Running {n_folds} folds")

    results = []

    for fold_idx in range(n_folds):
        print(f"\n--- FOLD {fold_idx+1}/{n_folds} ---")

        start_test = (fold_idx * 3) % n_participants
        test_participants = [shuffled_participants[(start_test + i) % n_participants]
                             for i in range(3)]

        start_val = (start_test + 3) % n_participants
        val_participants = [shuffled_participants[(start_val + i) % n_participants]
                            for i in range(2)]

        train_participants = [p for p in unique_participants
                              if p not in test_participants and p not in val_participants]

        print(f"Participants TEST: {test_participants}")
        print(f"Participants VAL:  {val_participants}")

        # Create split masks
        mask_test = np.isin(groups, test_participants)
        mask_val = np.isin(groups, val_participants)
        mask_train = np.isin(groups, train_participants)

        X_train = X[mask_train]
        y_train = y[mask_train]
        X_val = X[mask_val]
        y_val = y[mask_val]
        X_test = X[mask_test]
        y_test = y[mask_test]

        if data_fraction < 1.0:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=data_fraction, stratify=y_train, random_state=SEED + fold_idx
            )

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print("WARNING: One of the splits is empty, skipping fold")
            continue

        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=config['batch_size']
        )

        all_models = [
            ('DNN', DNN),
            ('LSTM', LSTMNet),
            ('Transformer', TransformerNet),
            ('SDCNet', SDCNet)
        ]
        models_to_train = [m for m in all_models if not config.get('sdcnet_only') or m[0] == 'SDCNet']

        for model_name, model_class in models_to_train:
            print(f"\n  Training {model_name} (Fold {fold_idx+1})...")
            
            model = model_class(input_channels=6, num_classes=num_classes, seq_length=405)
            
            model_trained, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config['num_epochs'],
                lr=config['lr'],
                weight_decay=config['weight_decay'],
                device=device,
                model_name=f"{model_name}_LOGO_fold{fold_idx+1}",
                save_dir=config['checkpoints_dir']
            )
            
            test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
                model_trained, test_loader, nn.CrossEntropyLoss(), device
            )
            
            print(f"  Test results - {model_name} (Fold {fold_idx+1}):")
            print(f"    Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
            
            plot_learning_curves(
                history, f"{model_name} LOGO Fold{fold_idx+1}", 
                os.path.join(config['results_dir'], 
                           f"learning_curves_{model_name}_LOGO_fold{fold_idx+1}.png")
            )
            
            plot_confusion_matrix(
                test_labels, test_preds, label_encoder.classes_,
                f"{model_name} LOGO Fold{fold_idx+1}", 
                os.path.join(config['results_dir'], 
                           f"confusion_matrix_{model_name}_LOGO_fold{fold_idx+1}.png")
            )
            
            report = classification_report(test_labels, test_preds, 
                                          target_names=label_encoder.classes_)
            report_dir = os.path.join(config['results_dir'], 'classification_report')
            os.makedirs(report_dir, exist_ok=True)
            with open(os.path.join(report_dir,
                                  f"classification_report_{model_name}_LOGO_fold{fold_idx+1}.txt"), 'w') as f:
                f.write(report)
            
            results.append({
                'Strategy': 'LOGO',
                'Model': model_name,
                'Fold': fold_idx+1,
                'Test_Participants': str(test_participants),
                'Test_Acc': test_acc,
                'Test_F1': test_f1,
                'Test_Loss': test_loss
            })
    
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "="*80)
        print("LOGO AGGREGATED RESULTS")
        print("="*80)
        for model_name in ['CNN1D', 'SDCNet', 'DNN', 'LSTM', 'Transformer']:
            model_results = df_results[df_results['Model'] == model_name]
            if len(model_results) > 0:
                mean_acc = model_results['Test_Acc'].mean()
                std_acc = model_results['Test_Acc'].std()
                mean_f1 = model_results['Test_F1'].mean()
                std_f1 = model_results['Test_F1'].std()

                print(f"\n{model_name}:")
                print(f"  Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
                print(f"  F1-Score: {mean_f1:.4f} +/- {std_f1:.4f}")

        df_results.to_csv(os.path.join(config['results_dir'], 'LOGO_cross_validation_results.csv'),
                         index=False)

    return results


# ============================================================================
# COMPARATIVE SUMMARY
# ============================================================================

def generate_comparative_summary(df_all_results, results_dir):
    """Generates a full comparative summary across all models and split strategies."""
    print("\n" + "="*80)
    print("GENERATING COMPARATIVE SUMMARY")
    print("="*80)
    
    # 1. Comparative table by strategy and model
    summary_data = []
    
    for strategy in ['Stratified', 'LOSO', 'LOGO']:
        strategy_results = df_all_results[df_all_results['Strategy'] == strategy]
        
        if len(strategy_results) == 0:
            continue
        
        for model in ['CNN1D', 'SDCNet', 'DNN', 'LSTM', 'Transformer']:
            model_results = strategy_results[strategy_results['Model'] == model]
            
            if len(model_results) == 0:
                continue
            
            mean_acc = model_results['Test_Acc'].mean()
            std_acc = model_results['Test_Acc'].std()
            mean_f1 = model_results['Test_F1'].mean()
            std_f1 = model_results['Test_F1'].std()
            mean_loss = model_results['Test_Loss'].mean()
            n_folds = len(model_results)
            
            summary_data.append({
                'Strategy': strategy,
                'Model': model,
                'N_Folds': n_folds,
                'Mean_Accuracy': mean_acc,
                'Std_Accuracy': std_acc,
                'Mean_F1': mean_f1,
                'Std_F1': std_f1,
                'Mean_Loss': mean_loss
            })
    
    df_summary = pd.DataFrame(summary_data)

    summary_path = os.path.join(results_dir, 'comparative_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\nComparative table saved: {summary_path}")

    print("\n" + "="*80)
    print("COMPARATIVE TABLE - ALL MODELS AND STRATEGIES")
    print("="*80)
    print(df_summary.to_string(index=False))
    
    # 2. Graphiques comparatifs
    
    # Chart 1: Accuracy by model and strategy
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot accuracy
    strategies = df_summary['Strategy'].unique()
    models = ['CNN1D', 'SDCNet', 'DNN', 'LSTM', 'Transformer']
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, strategy in enumerate(strategies):
        strategy_data = df_summary[df_summary['Strategy'] == strategy]
        means = []
        stds = []
        
        for model in models:
            model_data = strategy_data[strategy_data['Model'] == model]
            if len(model_data) > 0:
                means.append(model_data['Mean_Accuracy'].values[0])
                stds.append(model_data['Std_Accuracy'].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        axes[0].bar(x + i*width, means, width, label=strategy, yerr=stds, capsize=5)
    
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Model and Strategy')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot F1-Score
    for i, strategy in enumerate(strategies):
        strategy_data = df_summary[df_summary['Strategy'] == strategy]
        means = []
        stds = []

        for model in models:
            model_data = strategy_data[strategy_data['Model'] == model]
            if len(model_data) > 0:
                means.append(model_data['Mean_F1'].values[0])
                stds.append(model_data['Std_F1'].values[0])
            else:
                means.append(0)
                stds.append(0)

        axes[1].bar(x + i*width, means, width, label=strategy, yerr=stds, capsize=5)

    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('F1-Score by Model and Strategy')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(results_dir, 'models_comparison_by_strategy.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {comparison_plot_path}")

    # Heatmap of performance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    pivot_acc = df_summary.pivot(index='Model', columns='Strategy', values='Mean_Accuracy')
    sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Accuracy'})
    axes[0].set_title('Accuracy Heatmap')
    axes[0].set_xlabel('Strategy')
    axes[0].set_ylabel('Model')

    pivot_f1 = df_summary.pivot(index='Model', columns='Strategy', values='Mean_F1')
    sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'F1-Score'})
    axes[1].set_title('F1-Score Heatmap')
    axes[1].set_xlabel('Strategy')
    axes[1].set_ylabel('Model')
    
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, 'performance_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance heatmap saved: {heatmap_path}")

    print("\n" + "="*80)
    print("BEST MODELS BY STRATEGY")
    print("="*80)
    
    for strategy in strategies:
        strategy_data = df_summary[df_summary['Strategy'] == strategy]
        if len(strategy_data) > 0:
            best_acc = strategy_data.loc[strategy_data['Mean_Accuracy'].idxmax()]
            best_f1 = strategy_data.loc[strategy_data['Mean_F1'].idxmax()]

            print(f"\n{strategy}:")
            print(f"  Best Accuracy: {best_acc['Model']} ({best_acc['Mean_Accuracy']:.4f} +/- {best_acc['Std_Accuracy']:.4f})")
            print(f"  Best F1-Score: {best_f1['Model']} ({best_f1['Mean_F1']:.4f} +/- {best_f1['Std_F1']:.4f})")

    print("\n" + "="*80)
    print("GLOBAL RANKING (average across all strategies)")
    print("="*80)
    
    global_ranking = []
    for model in models:
        model_data = df_summary[df_summary['Model'] == model]
        if len(model_data) > 0:
            global_mean_acc = model_data['Mean_Accuracy'].mean()
            global_mean_f1 = model_data['Mean_F1'].mean()
            
            global_ranking.append({
                'Model': model,
                'Global_Mean_Accuracy': global_mean_acc,
                'Global_Mean_F1': global_mean_f1
            })
    
    df_global_ranking = pd.DataFrame(global_ranking)
    df_global_ranking = df_global_ranking.sort_values('Global_Mean_Accuracy', ascending=False)

    print("\nRanking by Accuracy:")
    for idx, row in df_global_ranking.iterrows():
        print(f"  {row['Model']}: {row['Global_Mean_Accuracy']:.4f}")

    df_global_ranking_f1 = df_global_ranking.sort_values('Global_Mean_F1', ascending=False)
    print("\nRanking by F1-Score:")
    for idx, row in df_global_ranking_f1.iterrows():
        print(f"  {row['Model']}: {row['Global_Mean_F1']:.4f}")

    ranking_path = os.path.join(results_dir, 'global_ranking.csv')
    df_global_ranking.to_csv(ranking_path, index=False)
    print(f"\nGlobal ranking saved: {ranking_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train baseline models (DNN, LSTM, Transformer) on Stratified/LOSO/LOGO splits'
    )
    parser.add_argument('--data-dir', type=str,
                        default=os.environ.get('HAR_DATA_DIR', 'Data Malwear/brut'),
                        help='Raw CSV directory (default: HAR_DATA_DIR or Data Malwear/brut)')
    parser.add_argument('--results-dir', type=str,
                        default=os.environ.get('HAR_BASELINE_RESULTS_DIR', 'results_baseline'),
                        help='Output results directory (default: HAR_BASELINE_RESULTS_DIR or results_baseline)')
    parser.add_argument('--checkpoints-dir', type=str,
                        default=os.environ.get('HAR_BASELINE_CHECKPOINTS_DIR', 'checkpoints_baseline'),
                        help='Output checkpoints directory (default: HAR_BASELINE_CHECKPOINTS_DIR or checkpoints_baseline)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                        help='Fraction of training data to use (default: 1.0 = 100%%)')
    parser.add_argument('--sdcnet-only', action='store_true',
                        help='Train only SDCNet (skip DNN, LSTM, Transformer)')
    args = parser.parse_args()

    print("="*80)
    print("BASELINE TRAINING - FULL CROSS-VALIDATION")
    print("="*80)

    config = {
        'data_path': args.data_dir,
        'results_dir': args.results_dir,
        'checkpoints_dir': args.checkpoints_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'data_fraction': args.data_fraction,
        'sdcnet_only': args.sdcnet_only,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Data dir:        {config['data_path']}")
    print(f"Results dir:     {config['results_dir']}")
    print(f"Checkpoints dir: {config['checkpoints_dir']}")
    
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['checkpoints_dir'], exist_ok=True)

    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    dataframes = load_csv_files(config['data_path'])

    if not dataframes:
        print("ERROR: No files loaded!")
        return

    print("\n" + "="*80)
    print("SEGMENTATION AND NORMALIZATION")
    print("="*80)

    X_raw, y_raw, groups = create_segments(dataframes)
    print(f"\nData dimensions:")
    print(f"  X: {X_raw.shape}")
    print(f"  y: {y_raw.shape}")
    print(f"  Participants: {len(np.unique(groups))}")

    X_norm, scaler = normalize_data(X_raw)
    y_encoded, label_encoder = encode_labels(y_raw)

    num_classes = len(label_encoder.classes_)
    print(f"\nNumber of classes: {num_classes}")
    
    all_results = []

    # 1. Stratified Split
    results_stratified = run_stratified_split(
        X_norm, y_encoded, num_classes, label_encoder, device, config
    )
    all_results.extend(results_stratified)

    # 2. LOSO Cross-Validation
    results_loso = run_loso_cross_validation(
        X_norm, y_encoded, groups, num_classes, label_encoder, device, config
    )
    all_results.extend(results_loso)

    # 3. LOGO Cross-Validation
    results_logo = run_logo_cross_validation(
        X_norm, y_encoded, groups, num_classes, label_encoder, device, config
    )
    all_results.extend(results_logo)

    df_all_results = pd.DataFrame(all_results)
    df_all_results.to_csv(os.path.join(config['results_dir'], 'all_results_baseline.csv'),
                          index=False)
    print(f"\nAll results saved to: {os.path.join(config['results_dir'], 'all_results_baseline.csv')}")

    generate_comparative_summary(df_all_results, config['results_dir'])

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {config['results_dir']}/")
    print(f"Models saved in:  {config['checkpoints_dir']}/")
    print(f"\nTotal training runs: {len(all_results)}")


if __name__ == "__main__":
    main()
