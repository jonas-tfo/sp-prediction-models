#!/bin/bash

# MMseqs2 Homology Removal Script for Machine Learning Training
# Removes sequences with high similarity to create non-redundant dataset
# Usage: ./remove_homology.sh input.fasta [similarity_threshold] [coverage_threshold]

set -e  # Exit on any error

# Input validation
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_fasta> [similarity_threshold] [coverage_threshold]"
    echo "Example: $0 sequences.fasta 0.3 0.8"
    echo ""
    echo "Default thresholds for ML training:"
    echo "  - Similarity: 0.3 (30% - removes high homology)"
    echo "  - Coverage: 0.8 (80% - ensures substantial overlap)"
    echo ""
    echo "Common ML thresholds:"
    echo "  - Conservative (strict): 0.25 similarity, 0.9 coverage"
    echo "  - Moderate: 0.3 similarity, 0.8 coverage"
    echo "  - Permissive: 0.4 similarity, 0.7 coverage"
    exit 1
fi

INPUT_FASTA="$1"
SIMILARITY_THRESHOLD="${2:-0.3}"  # Default 30% similarity
COVERAGE_THRESHOLD="${3:-0.8}"    # Default 80% coverage -> min 80 percent match between sequences 
OUTPUT_PREFIX="ml_filtered"

# Check if input file exists and validate format
if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: Input file '$INPUT_FASTA' not found!"
    exit 1
fi

# Check if this is a 3-line format (header, sequence, quality/labels)
echo "Checking input file format..."
FIRST_THREE_LINES=$(head -3 "$INPUT_FASTA")
if echo "$FIRST_THREE_LINES" | grep -q "^>.*" && echo "$FIRST_THREE_LINES" | tail -1 | grep -q "^[IOMSLT]*$"; then
    echo "Detected 3-line format (header, sequence, quality/labels)"
    echo "Label characters detected: $(echo "$FIRST_THREE_LINES" | tail -1 | grep -o . | sort -u | tr '\n' ' ')"
    echo "Converting to standard FASTA format..."
    
    # Create temporary standard FASTA file
    TEMP_FASTA="${INPUT_FASTA}.temp_standard.fasta"
    awk '
    BEGIN { seq_count = 0 }
    /^>/ { 
        if (seq_count > 0) print ""
        print $0; 
        seq_count++; 
        next 
    }
    seq_count > 0 && NR % 3 == 2 { 
        # This is the amino acid sequence line (every 2nd line after header)
        print $0
    }
    # Skip the 3rd line (labels: I, O, M, S, L, T)
    ' "$INPUT_FASTA" > "$TEMP_FASTA"
    
    INPUT_FASTA_PROCESSED="$TEMP_FASTA"
    echo "Converted file saved as: $TEMP_FASTA"
    
    # Show a sample of what was detected
    echo "Sample conversion:"
    echo "Original (first 9 lines):"
    head -9 "$INPUT_FASTA" | nl
    echo "Converted (first 6 lines):"
    head -6 "$TEMP_FASTA" | nl
else
    echo "Standard FASTA format detected"
    INPUT_FASTA_PROCESSED="$INPUT_FASTA"
fi

# Check if MMseqs2 is installed
if ! command -v mmseqs &> /dev/null; then
    echo "Error: MMseqs2 is not installed or not in PATH"
    echo "Please install MMseqs2 first: https://github.com/soedinglab/MMseqs2"
    exit 1
fi

echo "=========================================="
echo "MMseqs2 Homology Removal for ML Training"
echo "=========================================="
echo "Input file: $INPUT_FASTA"
echo "Similarity threshold: ${SIMILARITY_THRESHOLD} ($(echo "$SIMILARITY_THRESHOLD * 100" | bc -l | cut -d. -f1)%)"
echo "Coverage threshold: ${COVERAGE_THRESHOLD} ($(echo "$COVERAGE_THRESHOLD * 100" | bc -l | cut -d. -f1)%)"
echo "Output prefix: $OUTPUT_PREFIX"
echo ""

# Create output directory
mkdir -p "${OUTPUT_PREFIX}_results"

# Store the absolute path before changing directories
INPUT_FULL_PATH=$(realpath "$INPUT_FASTA_PROCESSED")

# Count original sequences
ORIGINAL_COUNT=$(grep -c "^>" "$INPUT_FASTA_PROCESSED")
echo "Original sequence count: $ORIGINAL_COUNT"

cd "${OUTPUT_PREFIX}_results"

# Step 1: Create MMseqs2 database from input sequences
echo ""
echo "Step 1: Creating MMseqs2 database..."
mmseqs createdb "$INPUT_FULL_PATH" "${OUTPUT_PREFIX}_db"

# Step 2: Cluster sequences with specified thresholds
echo "Step 2: Clustering sequences to identify homologous groups..."
echo "  - Using ${SIMILARITY_THRESHOLD} similarity threshold"
echo "  - Using ${COVERAGE_THRESHOLD} coverage threshold"

mmseqs cluster "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_clu" tmp \
    --min-seq-id "$SIMILARITY_THRESHOLD" \
    -c "$COVERAGE_THRESHOLD" \
    --cluster-mode 0

# Step 3: Extract representative sequences (one per cluster)
echo "Step 3: Extracting representative sequences..."
mmseqs result2repseq "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_clu" "${OUTPUT_PREFIX}_rep"
mmseqs result2flat "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_rep" "${OUTPUT_PREFIX}_non_redundant.fasta"

# Step 4: Create detailed cluster information
echo "Step 4: Generating cluster analysis..."
mmseqs createtsv "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_clu" "${OUTPUT_PREFIX}_clusters.tsv"

# Step 5: Create removed sequences list
echo "Step 5: Creating list of removed sequences..."
# Get all sequence IDs
mmseqs createtsv "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_all_ids.tsv"
# Get representative IDs
mmseqs createtsv "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_db" "${OUTPUT_PREFIX}_rep" "${OUTPUT_PREFIX}_rep_ids.tsv"

# Find removed sequences (those not selected as representatives)
cut -f1 "${OUTPUT_PREFIX}_all_ids.tsv" | sort > all_ids.tmp
cut -f1 "${OUTPUT_PREFIX}_rep_ids.tsv" | sort > rep_ids.tmp
comm -23 all_ids.tmp rep_ids.tmp > "${OUTPUT_PREFIX}_removed_ids.txt"

# Step 6: Generate comprehensive statistics
echo "Step 6: Generating statistics..."

# Check MMseqs2 version for compatibility
MMSEQS_VERSION=$(mmseqs version 2>/dev/null || echo "unknown")
echo "MMseqs2 version: $MMSEQS_VERSION"

FINAL_COUNT=$(grep -c "^>" "${OUTPUT_PREFIX}_non_redundant.fasta")
REMOVED_COUNT=$((ORIGINAL_COUNT - FINAL_COUNT))
REDUCTION_PERCENT=$(echo "scale=2; ($REMOVED_COUNT * 100) / $ORIGINAL_COUNT" | bc -l)
RETENTION_PERCENT=$(echo "scale=2; ($FINAL_COUNT * 100) / $ORIGINAL_COUNT" | bc -l)

# Cluster size analysis
cut -f1 "${OUTPUT_PREFIX}_clusters.tsv" | sort | uniq -c | sort -nr > "${OUTPUT_PREFIX}_cluster_sizes.txt"
TOTAL_CLUSTERS=$(cut -f1 "${OUTPUT_PREFIX}_clusters.tsv" | sort -u | wc -l)
SINGLETONS=$(grep -c "^\s*1 " "${OUTPUT_PREFIX}_cluster_sizes.txt" || echo "0")
MAX_CLUSTER_SIZE=$(head -1 "${OUTPUT_PREFIX}_cluster_sizes.txt" | awk '{print $1}')

# Create comprehensive summary report
cat > "${OUTPUT_PREFIX}_ML_summary.txt" << EOF
MMseqs2 Homology Removal Summary for ML Training
================================================
Analysis Date: $(date)
Input File: $INPUT_FASTA
$(if [ "$INPUT_FASTA_PROCESSED" != "$INPUT_FASTA" ]; then echo "Processed File: $INPUT_FASTA_PROCESSED (converted from 3-line format)"; fi)

Filtering Parameters:
- Similarity Threshold: ${SIMILARITY_THRESHOLD} ($(echo "$SIMILARITY_THRESHOLD * 100" | bc -l | cut -d. -f1)%)
- Coverage Threshold: ${COVERAGE_THRESHOLD} ($(echo "$COVERAGE_THRESHOLD * 100" | bc -l | cut -d. -f1)%)

Sequence Statistics:
- Original sequences: $ORIGINAL_COUNT
- Final sequences: $FINAL_COUNT
- Removed sequences: $REMOVED_COUNT
- Reduction: ${REDUCTION_PERCENT}%
- Retention: ${RETENTION_PERCENT}%

Clustering Statistics:
- Total clusters formed: $TOTAL_CLUSTERS
- Singleton clusters: $SINGLETONS
- Largest cluster size: $MAX_CLUSTER_SIZE
- Average cluster size: $(echo "scale=2; $ORIGINAL_COUNT / $TOTAL_CLUSTERS" | bc -l)

Files Generated:
- ${OUTPUT_PREFIX}_non_redundant.fasta: Final filtered sequences for ML training
- ${OUTPUT_PREFIX}_clusters.tsv: Detailed cluster assignments
- ${OUTPUT_PREFIX}_removed_ids.txt: List of sequence IDs that were removed
- ${OUTPUT_PREFIX}_cluster_sizes.txt: Cluster size distribution
- ${OUTPUT_PREFIX}_ML_summary.txt: This summary file

Recommendations for ML Training:
$(if (( $(echo "$REDUCTION_PERCENT > 50" | bc -l) )); then
    echo "- High redundancy detected (${REDUCTION_PERCENT}% reduction)"
    echo "- Dataset is now suitable for ML training with reduced overfitting risk"
else
    echo "- Moderate redundancy detected (${REDUCTION_PERCENT}% reduction)"
    echo "- Consider lowering similarity threshold if more aggressive filtering is needed"
fi)
$(if (( $(echo "$FINAL_COUNT < 1000" | bc -l) )); then
    echo "- WARNING: Small dataset ($FINAL_COUNT sequences) - consider collecting more data"
elif (( $(echo "$FINAL_COUNT < 5000" | bc -l) )); then
    echo "- Moderate dataset size ($FINAL_COUNT sequences) - should be sufficient for many ML tasks"
else
    echo "- Large dataset ($FINAL_COUNT sequences) - excellent for ML training"
fi)
EOF

# Step 7: Create training/validation split suggestions
echo "Step 7: Creating ML training suggestions..."
cat > "${OUTPUT_PREFIX}_ML_training_guide.txt" << EOF
Machine Learning Training Guide
==============================

Dataset Overview:
- Total sequences: $FINAL_COUNT
- Homology filtered at $(echo "$SIMILARITY_THRESHOLD * 100" | bc -l | cut -d. -f1)% similarity

Recommended Dataset Splits:
1. Training (70%): ~$((FINAL_COUNT * 70 / 100)) sequences
2. Validation (15%): ~$((FINAL_COUNT * 15 / 100)) sequences  
3. Test (15%): ~$((FINAL_COUNT * 15 / 100)) sequences

Next Steps:
1. Use ${OUTPUT_PREFIX}_non_redundant.fasta for your ML pipeline
2. Consider additional filtering based on sequence length if needed
3. Shuffle sequences before splitting into train/val/test sets
4. Monitor for potential data leakage in your specific domain

Quality Checks:
- Verify no sequences in test set are similar to training set
- Check sequence length distribution
- Validate that removed sequences don't contain critical examples

Commands to split dataset:
# Shuffle and split (using seqkit or similar tools)
seqkit shuffle ${OUTPUT_PREFIX}_non_redundant.fasta > shuffled.fasta
seqkit split -p 3 shuffled.fasta  # Creates 3 roughly equal parts
EOF

# Clean up temporary files
rm -f all_ids.tmp rep_ids.tmp

echo ""
echo "=========================================="
echo "HOMOLOGY REMOVAL COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Results Summary:"
echo "   Original sequences: $ORIGINAL_COUNT"
echo "   Final sequences: $FINAL_COUNT"
echo "   Removed: $REMOVED_COUNT (${REDUCTION_PERCENT}%)"
echo "   Clusters formed: $TOTAL_CLUSTERS"
echo ""
echo "Key Output Files:"
echo "   ✓ ${OUTPUT_PREFIX}_non_redundant.fasta (USE THIS FOR ML TRAINING)"
echo "   ✓ ${OUTPUT_PREFIX}_ML_summary.txt (Analysis summary)"
echo "   ✓ ${OUTPUT_PREFIX}_ML_training_guide.txt (ML training recommendations)"
echo "   ✓ ${OUTPUT_PREFIX}_removed_ids.txt (List of removed sequences)"
echo ""
echo "Largest homologous groups:"
head -5 "${OUTPUT_PREFIX}_cluster_sizes.txt"
echo ""
echo "Ready for machine learning training!"
echo "   Use the non-redundant FASTA file as input to your neural network pipeline."

# Clean up temporary files if created
if [ -f "${INPUT_FASTA}.temp_standard.fasta" ]; then
    echo ""
    echo "Cleaning up temporary files..."
    rm -f "${INPUT_FASTA}.temp_standard.fasta"
fi