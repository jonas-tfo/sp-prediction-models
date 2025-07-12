from sklearn.utils import resample
import subprocess
import pandas as pd

SCRIPT_PATH = "/Users/jonas/Desktop/Uni/PBL/sp-prediction-models/data/data_processing/mmseqprocessing.sh"


def run_bash_script(script_path, argument1): 
    """
    Runs a bash script and passes arguments to it.

    Args:
        script_path: The path to the bash script.
        argument1: The first argument to pass to the script.
        argument2: The second argument to pass to the script. # Add more as required

    Returns:
        A subprocess.CompletedProcess object containing the results of the execution.
    """
    try:
        # Correctly construct the command list:
        command = ['bash', script_path, argument1] # Add more arguments as needed
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print("Bash script output:")
        print(result.stdout)

        print("Bash script error output (if any):")
        print(result.stderr)

        print("Return code:", result.returncode)

        return result

    except subprocess.CalledProcessError as e:
        print(f"Error running bash script: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Standard error:\n{e.stderr}")
        return None  # Or raise the exception, depending on your error handling strategy
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def create_sliding_windows(sequence, labels, window_size, stride=1):
    """
    Creates windows with a given size (the number of windows being determined by the stride) from given sequences and their corresponding labels
    """
    windows = []
    window_labels = []
    positions = []

    # Pad sequence for edge cases
    pad_size = window_size // 2 # so starts classification after padding, at first real encoding
    padded_seq = 'X' * pad_size + sequence + 'X' * pad_size
    padded_labels = [0] * pad_size + labels + [0] * pad_size

    # Create sliding windows
    for i in range(0, len(sequence), stride):
        start_idx = i
        end_idx = i + window_size

        if end_idx <= len(padded_seq):
            window_seq = padded_seq[start_idx:end_idx]
            # Label for the center position of the window
            center_idx = start_idx + pad_size # residue to predict
            if center_idx < len(padded_labels):
                center_label = padded_labels[center_idx]

                windows.append(window_seq)
                window_labels.append(center_label)
                positions.append(i)  # Original position in sequence

    return windows, window_labels, positions



def load_and_preprocess_data_window(fasta_path, windowSize, stride):
    """
    Load FASTA data -> preprocess (homology reduction using mmseqs and task specific processing) -> use for sliding window approach
    params: path to the fasta file
    returns: windows, labels, ids as lists
    """
    
    run_bash_script("mmseqprocessing.sh", fasta_path)

    fasta_path = "./ml_filtered_results/ml_filtered_non_redundant.fasta"

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
            sequence, residue_labels, windowSize, stride
        )

        all_windows.extend(windows)
        all_labels.extend(window_labels)
        all_seq_ids.extend([idx] * len(windows))

    print(f"Total windows created: {len(all_windows)}")
    print(f"Signal peptide windows: {sum(all_labels)}")
    print(f"Non-signal peptide windows: {len(all_labels) - sum(all_labels)}")

    return all_windows, all_labels, all_seq_ids