#!/bin/bash
# =============================================================================
# Download Pre-trained Model Checkpoints
# =============================================================================
# Downloads the P-NDQN model and all ablation checkpoints from Zenodo.
# These allow reproducing Table 2 and Table 3 without retraining.
#
# Usage: bash download_models.sh
#
# Zenodo DOI: https://doi.org/10.5281/zenodo.20648833
# =============================================================================

set -e

ZENODO_URL="https://doi.org/10.5281/zenodo.20648833"

echo "Downloading P-NDQN pre-trained models..."

mkdir -p results/models/morl_seed42
mkdir -p results/ablation/no_write_seed42
mkdir -p results/ablation/no_phase_seed42
mkdir -p results/ablation/no_per_seed42
mkdir -p results/ablation/no_noisy_seed42
mkdir -p results/ablation/no_dueling_seed42
mkdir -p results/ablation/no_mask_seed42

# Main model — needed for Table 2
echo "  Downloading main model ..."
curl -L "${ZENODO_URL}/morl_seed42_model_best.pt" -o results/models/morl_seed42/model_best.pt

# Ablation models — needed for Table 3
for variant in no_write no_phase no_per no_noisy no_dueling no_mask; do
    echo "  Downloading ablation: ${variant}..."
    curl -L "${ZENODO_URL}/ablation_${variant}_seed42_model_best.pt" \
         -o results/ablation/${variant}_seed42/model_best.pt
done

echo ""
echo "Download complete. Verify with:"
echo "  python evaluate_ablation.py"
echo "  python comprehensive_evaluation.py --morl-model results/models/morl_seed42/model_best.pt"
