#!/bin/bash
# =============================================================================
# Run Ablation Experiments for P-NDQN
# =============================================================================
# Trains 7 ablation variants, each removing one component from P-NDQN.
# Each run: 10,000 episodes, seed 42, 100 MB memory budget.
# Training uses chbench_htap_balanced.sql (equal OLAP/OLTP mix).
# Models are saved to results/ablation/ and evaluated at 100 MB by
# evaluate_ablation.py to reproduce Table 3 in the paper.
#
# Usage:
#   bash run_ablation.sh
#   nohup bash run_ablation.sh > ablation_log.txt 2>&1 &
#
# Expected runtime: ~4 hours per variant (~28 hours total, sequential).
# =============================================================================

set -e
# Run from the repository root so all relative paths resolve correctly.
cd "$(dirname "$0")/.."

SEED=42
EPISODES=10000
MEMORY_BUDGET=100
WORKLOAD="data/workloads/chbench_htap_balanced.sql"

echo "============================================================"
echo "P-NDQN Ablation Study"
echo "Episodes: $EPISODES, Seed: $SEED, Memory budget: ${MEMORY_BUDGET}MB"
echo "Workload: $WORKLOAD"
echo "Started: $(date)"
echo "============================================================"

# [1/7] No write penalty
echo ""
echo "[1/7] Training: No Write Penalty"
echo "Started: $(date)"
python train_morl_index.py --mode train --episodes $EPISODES --seed $SEED \
    --workload $WORKLOAD \
    --memory-budget $MEMORY_BUDGET \
    --write-weight 0 \
    --output-dir results/ablation/no_write_seed${SEED}/
echo "Finished: $(date)"

# [2/7] No phase adaptation
echo ""
echo "[2/7] Training: No Phase Adaptation"
echo "Started: $(date)"
python train_morl_index.py --mode train --episodes $EPISODES --seed $SEED \
    --workload $WORKLOAD \
    --memory-budget $MEMORY_BUDGET \
    --phase-adaptation 0 \
    --output-dir results/ablation/no_phase_seed${SEED}/
echo "Finished: $(date)"

# [3/7] No PER (uniform replay)
echo ""
echo "[3/7] Training: No PER (uniform replay)"
echo "Started: $(date)"
python train_morl_index.py --mode train --episodes $EPISODES --seed $SEED \
    --workload $WORKLOAD \
    --memory-budget $MEMORY_BUDGET \
    --no-per \
    --output-dir results/ablation/no_per_seed${SEED}/
echo "Finished: $(date)"

# [4/7] No NoisyNet (epsilon-greedy only)
echo ""
echo "[4/7] Training: No NoisyNet (epsilon-greedy)"
echo "Started: $(date)"
python train_morl_index.py --mode train --episodes $EPISODES --seed $SEED \
    --workload $WORKLOAD \
    --memory-budget $MEMORY_BUDGET \
    --no-noisy \
    --output-dir results/ablation/no_noisy_seed${SEED}/
echo "Finished: $(date)"

# [5/7] No Dueling (standard Q-head)
echo ""
echo "[5/7] Training: No Dueling (standard Q-head)"
echo "Started: $(date)"
python train_morl_index.py --mode train --episodes $EPISODES --seed $SEED \
    --workload $WORKLOAD \
    --memory-budget $MEMORY_BUDGET \
    --no-dueling \
    --output-dir results/ablation/no_dueling_seed${SEED}/
echo "Finished: $(date)"

# [6/7] No action masking
echo ""
echo "[6/7] Training: No Action Masking"
echo "Started: $(date)"
python train_morl_index.py --mode train --episodes $EPISODES --seed $SEED \
    --workload $WORKLOAD \
    --memory-budget $MEMORY_BUDGET \
    --no-action-mask \
    --output-dir results/ablation/no_mask_seed${SEED}/
echo "Finished: $(date)"

# [7/7] No memory penalty (supplementary — not in paper Table 3)
echo ""
echo "[7/7] Training: No Memory Penalty (supplementary)"
echo "Started: $(date)"
python train_morl_index.py --mode train --episodes $EPISODES --seed $SEED \
    --workload $WORKLOAD \
    --memory-budget $MEMORY_BUDGET \
    --memory-weight 0 \
    --output-dir results/ablation/no_memory_seed${SEED}/
echo "Finished: $(date)"

echo ""
echo "============================================================"
echo "ALL ABLATION TRAINING COMPLETE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Next: run 'python evaluate_ablation.py' to reproduce Table 3."
