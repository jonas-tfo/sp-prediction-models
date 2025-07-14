#!/bin/bash

# MMseqs2 Homology Removal Script with Label Preservation
# Usage: ./remove_homology.sh input.fasta [similarity_threshold] [coverage_threshold]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_fasta> [similarity_threshold] [coverage_threshold]"
    echo "Example: $0 sequences.fasta 0.3 0.8"
    exit 1
fi

INPUT_FASTA="$1"
INPUT_DIR=$(dirname "$(realpath "$INPUT_FASTA")")
SIMILARITY_THRESHOLD="${2:-0.3}"
COVERAGE_THRESHOLD="${3:-0.8}"
OUTPUT_PREFIX="filtered"

if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: Input file '$INPUT_FASTA' not found!"
    exit 1
fi

if ! command -v mmseqs &> /dev/null; then
    echo "Error: MMseqs2 is not installed or not in PATH"
    exit 1
fi

echo "Processing: $INPUT_FASTA"
echo "Similarity threshold: $SIMILARITY_THRESHOLD"
echo "Coverage threshold: $COVERAGE_THRESHOLD"

# Check if this is a 3-line format (header, sequence, labels)
FIRST_THREE_LINES=$(head -3 "$INPUT_FASTA")
HAS_LABELS=false

if echo "$FIRST_THREE_LINES" | grep -q "^>.*" && echo "$FIRST_THREE_LINES" | tail -1 | grep -q "^[IOMSLT]*$"; then
    echo "Converting 3-line format to standard FASTA and preserving labels..."
    HAS_LABELS=true
    
    TEMP_FASTA="${INPUT_FASTA}.temp.fasta"
    LABELS_FILE="${INPUT_FASTA}.labels.txt"
    
    # Extract sequences and create label mapping
    awk '
    BEGIN { seq_count = 0 }
    /^>/ { 
        if (seq_count > 0) print ""
        header = $0
        print header > "'"$TEMP_FASTA"'"
        seq_count++
        next 
    }
    seq_count > 0 && NR % 3 == 2 { 
        print $0 > "'"$TEMP_FASTA"'"
        sequence = $0
        getline labels
        # Store mapping: header -> labels
        print header "\t" labels > "'"$LABELS_FILE"'"
    }
    ' "$INPUT_FASTA"
    
    INPUT_PROCESSED="$TEMP_FASTA"
else
    INPUT_PROCESSED="$INPUT_FASTA"
fi

# Count sequences
ORIGINAL_COUNT=$(grep -c "^>" "$INPUT_PROCESSED")
echo "Original sequences: $ORIGINAL_COUNT"

# Create output directory
cd $INPUT_DIR
cd ..
mkdir -p redundancy_removal_results
cd redundancy_removal_results

# Get absolute path
INPUT_FULL_PATH=$(realpath "$INPUT_PROCESSED")

# Create MMseqs2 database
echo "Creating database..."
mmseqs createdb "$INPUT_FULL_PATH" "${OUTPUT_PREFIX}_db"

# Cluster sequences
echo "Clustering sequences..."
mmseqs cluster "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_clu" tmp \
    --min-seq-id "$SIMILARITY_THRESHOLD" \
    -c "$COVERAGE_THRESHOLD" \
    --cluster-mode 0

# Extract representative sequences
echo "Extracting representatives..."
mmseqs result2repseq "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_clu" "${OUTPUT_PREFIX}_rep"
mmseqs result2flat "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_rep" "${OUTPUT_PREFIX}_representatives.fasta"

# Create final output with labels if they existed (3 line fasta format)
if [ "$HAS_LABELS" = true ]; then
    echo "Adding labels back to representatives..."
    LABELS_FULL_PATH=$(realpath "${INPUT_FASTA}.labels.txt") # stores id line, tab, label
    
    # Create the final output with 3-line format
    awk '
    BEGIN {
        # Read labels into associative array
        while ((getline line < "'"$LABELS_FULL_PATH"'") > 0) {
            split(line, parts, "\t")
            labels[parts[1]] = parts[2]
        }
        close("'"$LABELS_FULL_PATH"'")
    }
    /^>/ {
        header = $0
        getline sequence
        print header
        print sequence
        if (header in labels) {
            print labels[header]
        } else {
            print "Unknown"
        }
    }
    ' "${OUTPUT_PREFIX}_representatives.fasta" > "${OUTPUT_PREFIX}_non_redundant.fasta" # contains the ids and aa sequences (no labels)
else
    # Just copy the representatives file
    cp "${OUTPUT_PREFIX}_representatives.fasta" "${OUTPUT_PREFIX}_non_redundant.fasta"
fi

# Count final sequences
FINAL_COUNT=$(grep -c "^>" "${OUTPUT_PREFIX}_non_redundant.fasta")
REMOVED_COUNT=$((ORIGINAL_COUNT - FINAL_COUNT))

echo ""
echo "Results:"
echo "  Original: $ORIGINAL_COUNT sequences"
echo "  Final: $FINAL_COUNT sequences"  
echo "  Removed: $REMOVED_COUNT sequences"
echo "  Output: redundancy_removal_results/${OUTPUT_PREFIX}_non_redundant.fasta"

# Clean up temporary files
cd ..
if [ -f "${INPUT_FASTA}.temp.fasta" ]; then
    rm -f "${INPUT_FASTA}.temp.fasta"
fi
if [ -f "${INPUT_FASTA}.labels.txt" ]; then
    rm -f "${INPUT_FASTA}.labels.txt"
fi