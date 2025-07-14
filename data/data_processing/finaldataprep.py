from sklearn.utils import resample
import subprocess
import pandas as pd
import os


def run_bash_script(script_path, fastaPath): 
    """
    Runs a bash script and passes fasta path to it, to be used with redudancy reduction
    """
    try:
        command = ['bash', script_path, fastaPath] 
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
        return None  
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def load_and_preprocess_data(fastaPath: str, outputPath: str, redundancyRemovalVersion: str = "short") -> str:
    """
    Load FASTA data -> homology reduction using mmseqs -> task specific processing
    """

    scriptDir = os.path.dirname(os.path.abspath(__file__))

    if redundancyRemovalVersion == "short":
        script = os.path.join(scriptDir, "redundancy_removal_short.sh")
    elif redundancyRemovalVersion == "long":
        script = os.path.join(scriptDir, "redundancy_removal_long.sh")
    else:
        print("Invalid redundancy removal version, terminating")
        return
    
    run_bash_script(script, fastaPath)

    scriptDir = os.path.dirname(os.path.abspath(__file__))
    dataDir = os.path.abspath(os.path.join(scriptDir, "..")) # get parent dir, which is where folder with target fasta is
    processedFasta = os.path.join(dataDir, "redundancy_removal_results", "filtered_non_redundant.fasta")

    records = []

    with open(processedFasta, "r") as f:
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

    # Save to CSV
    df.to_csv(outputPath, index=False)
    print(f"Processed data saved to {outputPath}")

    

if __name__ == "__main__":
    load_and_preprocess_data("/Users/jonas/Desktop/Uni/PBL/sp-prediction-models/data/raw/complete_set_unpartitioned.fasta", \
                              "/Users/jonas/Desktop/Uni/PBL/sp-prediction-models/data/processed/processed_data.csv")