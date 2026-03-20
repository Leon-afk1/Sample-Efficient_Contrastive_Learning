#!/usr/bin/env python3
"""
Data preparation for Contrastive Learning.
Creates triplets (anchor, positive, negative) for training.
"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from sklearn.model_selection import train_test_split

# Configuration
DEFAULT_DATA_DIR = os.environ.get('HAR_DATA_DIR', 'Data Malwear/brut')
DEFAULT_OUTPUT_DIR = os.environ.get('HAR_CONTRASTIVE_DATA_DIR', 'contrastive_data')
SEQUENCE_LENGTH = 405  # 3 seconds at 135 Hz
SEED = 42

np.random.seed(SEED)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_csv_files(data_dir):
    """Loads all annotated CSV files from a directory into a dictionary."""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_annotated.csv')]
    print(f"Found {len(csv_files)} CSV files")

    dataframes = {}

    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(filepath, low_memory=False)
            dataframes[csv_file] = df
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    print(f"{len(dataframes)} files loaded successfully")
    return dataframes


def create_segments(dataframes, window_size=405):
    """
    Segments data into fixed-size windows starting from annotation points (TAG column).
    """
    X = []
    y = []
    groups = []
    files = []
    
    feature_cols = ['ACC_X', 'ACC_Y', 'ACC_Z', 'PPG_I', 'PPG_II', 'PPG_III']
    
    for filename, df in dataframes.items():
        participant_id = filename.split('_')[1]
        # Find annotation start indices
        tagged_indices = df[df['TAG'].notna()].index
        
        for start_idx in tagged_indices:
            end_idx = start_idx + window_size
            
            if end_idx <= len(df):
                segment = df.iloc[start_idx:end_idx][feature_cols].values
                label = df.iloc[start_idx]['TAG']
                
                # Skip segments with missing values
                if not np.isnan(segment).any():
                    X.append(segment)
                    y.append(label)
                    groups.append(participant_id)
                    files.append(filename)
    
    return np.array(X), np.array(y), np.array(groups), np.array(files)


def normalize_data(X):
    """Z-score normalization per channel."""
    X_normalized = np.zeros_like(X)

    for i in range(X.shape[2]):  # Per channel
        mean = X[:, :, i].mean()
        std = X[:, :, i].std()
        X_normalized[:, :, i] = (X[:, :, i] - mean) / (std + 1e-8)
    
    return X_normalized


def encode_labels(y):
    """Encodes string labels to integer indices."""
    unique_labels = sorted(np.unique(y))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[label] for label in y])
    
    return y_encoded, label_to_idx


# ============================================================================
# 2. TRIPLET INDEX GENERATION
# ============================================================================

class TripletIndexGenerator:
    """Triplet index generator for contrastive training."""
    
    def __init__(self, labels, strategy='batch_hard'):
        """
        Args:
            labels: array of sample labels
            strategy: 'random' or 'batch_hard'
        """
        self.labels = labels
        self.strategy = strategy
        self.n_samples = len(labels)

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.labels_list = list(self.label_to_indices.keys())

        print(f"\nTriplet generator initialized:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Classes: {len(self.labels_list)}")
        for label in self.labels_list:
            print(f"    Class {label}: {len(self.label_to_indices[label])} samples")
    
    def generate_random_triplets(self, n_triplets):
        """
        Generates n_triplets random triplets.
        Returns: (anchor_indices, positive_indices, negative_indices)
        """
        # Only use classes with at least 2 samples (anchor != positive)
        valid_labels = [label for label in self.labels_list
                       if len(self.label_to_indices[label]) >= 2]
        
        if len(valid_labels) < 2:
            raise ValueError(f"Not enough classes with 2+ samples. "
                           f"Valid classes: {len(valid_labels)}, Total: {len(self.labels_list)}")
        
        anchors = []
        positives = []
        negatives = []
        
        for _ in range(n_triplets):
            anchor_label = np.random.choice(valid_labels)

            anchor_idx, positive_idx = np.random.choice(
                self.label_to_indices[anchor_label], size=2, replace=False
            )

            assert anchor_idx != positive_idx

            negative_labels = [l for l in valid_labels if l != anchor_label]
            if len(negative_labels) == 0:
                negative_labels = [l for l in self.labels_list if l != anchor_label]
            
            negative_label = np.random.choice(negative_labels)
            negative_idx = np.random.choice(self.label_to_indices[negative_label])
            
            anchors.append(anchor_idx)
            positives.append(positive_idx)
            negatives.append(negative_idx)
        
        return np.array(anchors), np.array(positives), np.array(negatives)
    
    def save_triplets(self, filepath, n_triplets):
        """Generates and saves triplets to disk."""
        anchors, positives, negatives = self.generate_random_triplets(n_triplets)
        
        triplets = {
            'anchors': anchors,
            'positives': positives,
            'negatives': negatives,
            'strategy': self.strategy
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(triplets, f)
        
        print(f"\n{n_triplets} triplets saved: {filepath}")
        return triplets


# ============================================================================
# 3. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare preprocessed data and triplet indices for contrastive training"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help='Raw data directory containing *_annotated.csv files '
             f'(default: {DEFAULT_DATA_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for preprocessed artifacts '
             f'(default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=SEQUENCE_LENGTH,
        help=f'Window length for segmentation (default: {SEQUENCE_LENGTH})'
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    sequence_length = args.sequence_length

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    print(f"Data dir:        {data_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Sequence length: {sequence_length}")

    dataframes = load_csv_files(data_dir)

    if not dataframes:
        print("ERROR: No files loaded!")
        return

    all_participants = set()
    all_tags = set()
    for filename in dataframes.keys():
        participant_id = filename.split('_')[1]
        all_participants.add(participant_id)
    for df in dataframes.values():
        tags = df['TAG'].dropna().unique()
        all_tags.update(tags)

    print(f"Participants: {sorted(all_participants)}")
    print(f"Labels (TAG): {sorted(all_tags)}")

    print("\n" + "="*80)
    print("SEGMENTATION")
    print("="*80)

    X, y, participants, files = create_segments(dataframes, window_size=sequence_length)
    print(f"Segments: {X.shape}  (samples={X.shape[0]}, timesteps={X.shape[1]}, features={X.shape[2]})")

    print("\n" + "="*80)
    print("NORMALIZATION")
    print("="*80)

    X_normalized = normalize_data(X)
    print(f"  Mean: {X_normalized.mean():.6f}")
    print(f"  Std:  {X_normalized.std():.6f}")

    y_encoded, label_to_idx = encode_labels(y)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    print(f"\nLabel encoding:")
    for label, idx in label_to_idx.items():
        print(f"  {label} -> {idx}")

    unique, counts = np.unique(y_encoded, return_counts=True)
    print(f"\nClass distribution:")
    for label_idx, count in zip(unique, counts):
        print(f"  {idx_to_label[label_idx]}: {count} samples")

    print("\n" + "="*80)
    print("SAVING PREPROCESSED DATA")
    print("="*80)

    data_dict = {
        'X': X_normalized,
        'y': y_encoded,
        'participants': participants,
        'files': files,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'sequence_length': sequence_length
    }

    data_path = os.path.join(output_dir, 'preprocessed_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Preprocessed data saved: {data_path}")

    print("\n" + "="*80)
    print("GENERATING TRIPLETS")
    print("="*80)

    triplet_gen = TripletIndexGenerator(y_encoded, strategy='random')

    # Generate enough triplets to cover multiple training epochs
    n_triplets_train = len(X_normalized) * 10

    triplets_path = os.path.join(output_dir, 'triplets_train.pkl')
    triplet_gen.save_triplets(triplets_path, n_triplets_train)

    metadata = {
        'n_samples': len(X_normalized),
        'n_classes': len(label_to_idx),
        'n_participants': len(np.unique(participants)),
        'sequence_length': sequence_length,
        'n_features': X_normalized.shape[2],
        'class_distribution': dict(zip([idx_to_label[i] for i in unique], counts.tolist()))
    }

    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved: {metadata_path}")

    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\nFiles created in {output_dir}/:")
    print(f"  - preprocessed_data.pkl ({X_normalized.nbytes / 1e6:.2f} MB)")
    print(f"  - triplets_train.pkl")
    print(f"  - metadata.pkl")


if __name__ == '__main__':
    main()
