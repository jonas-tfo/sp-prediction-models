"""
make this be callable in the bash script, act as a final layer on top of the homology reduction before training
"""

from sklearn.utils import resample
import sys
import pandas as pd


fasta_path = sys.argv[1]


records = []

with open(fasta_path, "r") as f:
    current_record = None
    for line in f:
        if line.startswith(">"):
            if current_record is not None:
                if current_record["sequence"] is not None and current_record["label"] is not None:
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

    # Add last record
    if current_record is not None:
        if current_record["sequence"] is not None and current_record["label"] is not None:
            records.append(current_record)

print(f"Total records loaded: {len(records)}")

# Convert to DataFrame
df_raw = pd.DataFrame(records)

# Filter out sequences with 'P' in labels (if needed)
df = df_raw[~df_raw["label"].str.contains("P")]

# Map signal peptide types to binary classification
df["has_signal_peptide"] = df["type"].map({
    "NO_SP": 0,
    "LIPO": 1,
    "SP": 1,
    "TAT": 1,
    "TATLIPO": 1
})

# Balance the dataset at sequence level first
df_majority = df[df["has_signal_peptide"] == 0]
df_minority = df[df["has_signal_peptide"] == 1]

if not df_minority.empty and not df_majority.empty:

    n_samples = min(len(df_majority), 5000) # Limit samples to 5000 to prevent high ram usage
    df_majority_sampled = resample(
        df_majority,
        replace=False, # sample without replacement
        n_samples=n_samples,
        random_state=42
    )
    df_balanced = pd.concat([df_majority_sampled, df_minority]) # Include all minority samples
else:
    df_balanced = df.copy()


# Convert residue-level labels to binary
label_map = {'S': 1, 'T': 1, 'L': 1, 'I': 0, 'M': 0, 'O': 0}

# Create sliding windows for all sequences
all_windows = []
all_labels = []
all_seq_ids = []

for idx, row in df_balanced.iterrows():
    sequence = row["sequence"]
    label_string = row["label"]

    # Convert label string to binary array
    residue_labels = [label_map.get(c, 0) for c in label_string]

    # Skip sequences where label length doesn't match sequence length
    if len(residue_labels) != len(sequence):
        print("A sequence length is not equal to the label length")
        continue

    # Create sliding windows for this sequence
    windows, window_labels, positions = create_sliding_windows(
        sequence, residue_labels, WINDOW_SIZE, STRIDE
    )

    all_windows.extend(windows)
    all_labels.extend(window_labels)
    all_seq_ids.extend([idx] * len(windows))

print(f"Total windows created: {len(all_windows)}")
print(f"Signal peptide windows: {sum(all_labels)}")
print(f"Non-signal peptide windows: {len(all_labels) - sum(all_labels)}")
