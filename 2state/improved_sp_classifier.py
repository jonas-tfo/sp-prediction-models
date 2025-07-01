# Improved Signal Peptide Classifier with Efficient Encoding

import torch
import numpy as np
import jax.numpy as jnp
import jax
from flax import linen as nn
import optax
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple, Dict, Any
import gc

class ProteinDataset(Dataset):
    """Custom dataset for protein sequences"""
    def __init__(self, sequences: List[str], labels: List[List[int]], max_length: int = 512):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Truncate if too long
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
            label = label[:self.max_length]
            
        return {
            'sequence': sequence,
            'label': label,
            'length': len(sequence)
        }

def collate_batch(batch):
    """Custom collate function for batching"""
    sequences = [item['sequence'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    # Add spaces between amino acids for BERT
    spaced_sequences = [" ".join(seq) for seq in sequences]
    
    return {
        'sequences': spaced_sequences,
        'labels': labels,
        'lengths': lengths
    }

class EfficientProteinEncoder:
    """Efficient protein encoder using batched processing"""
    
    def __init__(self, model_name: str = "Rostlab/prot_bert", batch_size: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()  # Set to evaluation mode
        self.batch_size = batch_size
    
    def encode_batch(self, sequences: List[str]) -> torch.Tensor:
        """Encode a batch of sequences efficiently"""
        # Tokenize batch
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder(**tokens)
            # Get embeddings and remove [CLS] and [SEP] tokens
            embeddings = outputs.last_hidden_state[:, 1:-1]  # [batch, seq_len-2, hidden_dim]
        
        return embeddings
    
    def encode_dataset(self, dataset: ProteinDataset) -> Tuple[np.ndarray, np.ndarray]:
        """Encode entire dataset efficiently"""
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_batch
        )
        
        all_embeddings = []
        all_labels = []
        
        print(f"Encoding {len(dataset)} sequences in batches of {self.batch_size}...")
        
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i+1}/{len(dataloader)}")
            
            # Encode sequences
            embeddings = self.encode_batch(batch['sequences'])
            
            # Process each sequence in the batch
            for j, (emb, labels, length) in enumerate(zip(embeddings, batch['labels'], batch['lengths'])):
                # Truncate embedding to actual sequence length
                emb_truncated = emb[:length].cpu().numpy()
                labels_array = np.array(labels[:length])
                
                all_embeddings.append(emb_truncated)
                all_labels.append(labels_array)
            
            # Clear GPU memory
            del embeddings
            torch.cuda.empty_cache()
        
        return all_embeddings, all_labels

def load_and_prep_data_efficient(data_path: str, max_length: int = 512) -> Tuple[Any, Any, Any, Any]:
    """Efficiently load and prepare data"""
    print("Loading data...")
    
    records = []
    with open(data_path, "r") as f:
        current_record = None
        for line in f:
            if line.startswith(">"):
                if current_record is not None:
                    if current_record["sequence"] and current_record["label"]:
                        # Filter by length
                        if len(current_record["sequence"]) <= max_length:
                            records.append(current_record)
                
                uniprot_ac, kingdom, type_ = line[1:].strip().split("|")
                current_record = {
                    "uniprot_ac": uniprot_ac, 
                    "kingdom": kingdom, 
                    "type": type_, 
                    "sequence": None, 
                    "label": None
                }
            else:
                if current_record["sequence"] is None:
                    current_record["sequence"] = line.strip()
                elif current_record["label"] is None:
                    current_record["label"] = line.strip()
        
        # Don't forget the last record
        if current_record and current_record["sequence"] and current_record["label"]:
            if len(current_record["sequence"]) <= max_length:
                records.append(current_record)
    
    print(f"Loaded {len(records)} valid records")
    
    # Create DataFrame and clean data
    df = pd.DataFrame(records)
    df = df[~df["sequence"].str.contains("P")]  # Remove sequences with 'P'
    
    # Balance classes
    df_majority = df[df["type"] == "NO_SP"]
    df_minority = df[df["type"] != "NO_SP"]
    
    # Sample down majority class instead of upsampling minority (more efficient)
    n_samples = min(len(df_majority), len(df_minority) * 2)  # 2:1 ratio
    df_majority_sampled = df_majority.sample(n=n_samples, random_state=42)
    df_balanced = pd.concat([df_majority_sampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced dataset size: {len(df_balanced)}")
    print("Class distribution:")
    print(df_balanced["type"].value_counts())
    
    # Encode labels
    label_map = {'S': 1, 'T': 1, 'L': 1, 'I': 0, 'M': 0, 'O': 0}
    df_balanced["encoded_label"] = df_balanced["label"].apply(
        lambda x: [label_map[c] for c in x if c in label_map]
    )
    
    # Filter out sequences with no valid labels
    df_balanced = df_balanced[df_balanced["encoded_label"].map(len) > 0]
    
    sequences = df_balanced["sequence"].tolist()
    labels = df_balanced["encoded_label"].tolist()
    
    # Split data first, then encode
    train_seqs, test_seqs, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training sequences: {len(train_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")
    
    return train_seqs, test_seqs, train_labels, test_labels

def pad_sequences(embeddings: List[np.ndarray], labels: List[np.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pad sequences to same length for JAX"""
    max_len = max(len(emb) for emb in embeddings)
    hidden_dim = embeddings[0].shape[1]
    
    # Pad embeddings
    padded_embeddings = np.zeros((len(embeddings), max_len, hidden_dim))
    padded_labels = np.zeros((len(labels), max_len))
    
    for i, (emb, lbl) in enumerate(zip(embeddings, labels)):
        padded_embeddings[i, :len(emb)] = emb
        padded_labels[i, :len(lbl)] = lbl
    
    return jnp.array(padded_embeddings), jnp.array(padded_labels)

class PerResidueClassifier(nn.Module):
    """Improved classifier with dropout and better architecture"""
    hidden_size: int = 256
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_size // 2)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(1)(x)
        x = x.squeeze(-1)
        return x

def create_train_state(model, rng, sample_input, learning_rate=1e-3):
    """Create training state"""
    params = model.init(rng, sample_input, training=False)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    return params, opt_state, optimizer

@jax.jit
def train_step(params, opt_state, optimizer, model, batch_x, batch_y):
    """Single training step"""
    def loss_fn(params):
        logits = model.apply(params, batch_x, training=True, rngs={'dropout': jax.random.PRNGKey(0)})
        loss = optax.sigmoid_binary_cross_entropy(logits, batch_y).mean()
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

@jax.jit
def eval_step(params, model, batch_x, batch_y):
    """Single evaluation step"""
    logits = model.apply(params, batch_x, training=False)
    loss = optax.sigmoid_binary_cross_entropy(logits, batch_y).mean()
    preds = (jax.nn.sigmoid(logits) > 0.5).astype(int)
    accuracy = jnp.mean(preds == batch_y)
    return loss, accuracy

def train_model():
    """Main training function"""
    # Set up paths (adjust as needed)
    FASTA_PATH = "/content/drive/MyDrive/PBLRost/data/complete_set_unpartitioned.fasta"
    
    # Load and prepare data
    train_seqs, test_seqs, train_labels, test_labels = load_and_prep_data_efficient(FASTA_PATH)
    
    # Create datasets
    train_dataset = ProteinDataset(train_seqs, train_labels)
    test_dataset = ProteinDataset(test_seqs, test_labels)
    
    # Encode sequences efficiently
    encoder = EfficientProteinEncoder(batch_size=8)  # Smaller batch size for encoding
    
    print("Encoding training data...")
    train_embeddings, train_labels_encoded = encoder.encode_dataset(train_dataset)
    
    print("Encoding test data...")
    test_embeddings, test_labels_encoded = encoder.encode_dataset(test_dataset)
    
    # Clear encoder to free GPU memory
    del encoder
    torch.cuda.empty_cache()
    gc.collect()
    
    # Pad sequences
    print("Padding sequences...")
    X_train, Y_train = pad_sequences(train_embeddings, train_labels_encoded)
    X_test, Y_test = pad_sequences(test_embeddings, test_labels_encoded)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize model
    model = PerResidueClassifier()
    rng = jax.random.PRNGKey(42)
    
    # Create training state
    params, opt_state, optimizer = create_train_state(model, rng, X_train[:1])
    
    # Training parameters
    batch_size = 16
    num_epochs = 10
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        # Shuffle training data
        perm = jax.random.permutation(rng, X_train.shape[0])
        X_train_shuffled = X_train[perm]
        Y_train_shuffled = Y_train[perm]
        
        # Training
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            batch_x = X_train_shuffled[i:i + batch_size]
            batch_y = Y_train_shuffled[i:i + batch_size]
            
            params, opt_state, loss = train_step(params, opt_state, optimizer, model, batch_x, batch_y)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluation
        eval_loss, eval_acc = eval_step(params, model, X_test, Y_test)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_loss:.4f}, "
              f"Test Loss: {eval_loss:.4f}, "
              f"Test Acc: {eval_acc:.4f}")
    
    return params, model, X_test, Y_test

# Example usage:
if __name__ == "__main__":
    # Mount Google Drive if in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except:
        pass
    
    # Train the model
    params, model, X_test, Y_test = train_model()
    print("Training completed!")