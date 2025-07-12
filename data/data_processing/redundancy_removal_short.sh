#!/bin/bash

# Simple MMseqs2 Homology Removal Script
# Usage: ./remove_homology.sh input.fasta [similarity_threshold] [coverage_threshold]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_fasta> [similarity_threshold] [coverage_threshold]"
    echo "Example: $0 sequences.fasta 0.3 0.8"
    exit 1
fi

INPUT_FASTA="$1"
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
if echo "$FIRST_THREE_LINES" | grep -q "^>.*" && echo "$FIRST_THREE_LINES" | tail -1 | grep -q "^[IOMSLT]*$"; then
    echo "Converting 3-line format to standard FASTA..."
    
    TEMP_FASTA="${INPUT_FASTA}.temp.fasta"
    awk '
    BEGIN { seq_count = 0 }
    /^>/ { 
        if (seq_count > 0) print ""
        print $0; 
        seq_count++; 
        next 
    }
    seq_count > 0 && NR % 3 == 2 { 
        print $0
    }
    ' "$INPUT_FASTA" > "$TEMP_FASTA"
    
    INPUT_PROCESSED="$TEMP_FASTA"
else
    INPUT_PROCESSED="$INPUT_FASTA"
fi

# Count sequences
ORIGINAL_COUNT=$(grep -c "^>" "$INPUT_PROCESSED")
echo "Original sequences: $ORIGINAL_COUNT"

# Create output directory
mkdir -p results
cd results

# Get absolute path
INPUT_FULL_PATH=$(realpath "../$INPUT_PROCESSED")

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
mmseqs result2flat "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_rep" "${OUTPUT_PREFIX}_non_redundant.fasta"

# Count final sequences
FINAL_COUNT=$(grep -c "^>" "${OUTPUT_PREFIX}_non_redundant.fasta")
REMOVED_COUNT=$((ORIGINAL_COUNT - FINAL_COUNT))

echo ""
echo "Results:"
echo "  Original: $ORIGINAL_COUNT sequences"
echo "  Final: $FINAL_COUNT sequences"  
echo "  Removed: $REMOVED_COUNT sequences"
echo "  Output: results/${OUTPUT_PREFIX}_non_redundant.fasta"

# Clean up
cd ..
if [ -f "${INPUT_FASTA}.temp.fasta" ]; then
    rm -f "${INPUT_FASTA}.temp.fasta"
fi

echo "Done!"