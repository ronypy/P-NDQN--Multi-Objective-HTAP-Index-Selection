# P-NDQN: Prioritized Noisy Dueling DQN for Multi-Objective HTAP Index Selection

This repository contains the code, workloads, and pre-computed results for a submitted paper in the ICDE 2027:

> **Multi-Objective Reinforcement Learning for Index Selection in Hybrid Transactional/Analytical Workloads**

P-NDQN is a reinforcement learning agent that selects database indexes for Hybrid Transactional/Analytical Processing (HTAP) workloads by jointly optimizing three objectives: query latency, write overhead, and memory efficiency.

---

## Repository Structure

```
├── train_morl_index.py          # P-NDQN agent and training loop
├── environment_morl.py          # MORL index-selection environment
├── morl_reward.py               # Multi-objective reward function
├── pg_database.py               # PostgreSQL + HypoPG interface
│
├── baselines/                   # Re-implementations of comparison methods
│   ├── random_baseline.py       
│   ├── drlinda_baseline.py      
│   ├── swirl_baseline.py         
│   ├── smartix_baseline.py      
│   ├── anytime_baseline.py      
│   └── dba_bandits_baseline.py  
│
├── run_experiments.py           # Train all methods across seeds
├── comprehensive_evaluation.py  
├── run_ablation.sh              
├── evaluate_ablation.py         
│
├── data/
│   ├── db_credentials_template.json  
│   └── workloads/
│       ├── chbench_htap_train.sql    
│       ├── chbench_htap_test.sql     
│       └── chbench_htap_balanced.sql 
│
├── results/
│   ├── comprehensive_evaluation.json  
│   └── ablation_results.json          
│
├── download_models.sh           # Fetch pre-trained checkpoints from Zenodo
└── requirements.txt
```

---

## Requirements

**Python packages:**
```bash
pip install -r requirements.txt
```

**System dependencies (not pip-installable):**
- PostgreSQL 14 with the [HypoPG](https://github.com/HypoPG/hypopg) extension (v1.4.0)
- CH-Benchmark database (https://github.com/cmu-db/benchbase/)

Install HypoPG:
```bash
git clone https://github.com/HypoPG/hypopg.git
cd hypopg && make && sudo make install
psql -c "CREATE EXTENSION hypopg;" chbench
```

---

## Setup

1. **Configure database credentials:**
   ```bash
   cp data/db_credentials_template.json data/db_credentials_pg.json
   # Edit db_credentials_pg.json with your PostgreSQL details
   ```
   Alternatively, export environment variables:
   ```bash
   export PG_USER=postgres PG_PASSWORD=secret PG_HOST=localhost PG_PORT=5432 PG_DATABASE=chbench
   ```

2. **Download pre-trained models** (skip if retraining from scratch):
   ```bash
   bash download_models.sh
   ```
   Models are hosted on Zenodo: https://doi.org/10.5281/zenodo.20648833

---

## Quickstart: Reproduce Paper Results

### Table 2 — Comparison against baselines (using pre-trained model)
```bash
python comprehensive_evaluation.py \
    --morl-model results/models/morl_seed42/model_best.pt \
    --n-episodes 15 \
    --output results/comprehensive_evaluation.json
```
Expected output matches the pre-computed values in `results/comprehensive_evaluation.json`.

### Table 3 — Ablation study (using pre-trained ablation models)
```bash
python evaluate_ablation.py
```
Expected output matches `results/ablation_results.json`.

---

## Full Reproduction from Scratch

### Train P-NDQN (three seeds, ~13 hours each)
```bash
python run_experiments.py --episodes 10000 --seeds 42 153 264
```

### Train ablation variants (~4 hours each, ~28 hours total)
```bash
bash run_ablation.sh
```

### Evaluate
```bash
python comprehensive_evaluation.py --morl-model results/models/morl_seed42/model_best.pt
python evaluate_ablation.py
```

---

## Baseline Disclaimer

All baselines in `baselines/` are **our re-implementations** of the methods described in the cited papers, adapted to use the same environment, action space, and evaluation protocol as P-NDQN for a fair comparison. They are not the original authors' code. Please refer to the original papers for the authors' implementations:

| Baseline | Reference |
|----------|-----------|
| DRLinda | Sadri et al., SIGMOD 2020 |
| SWIRL | Kossmann et al., VLDB 2022 |
| SmartIX | Licks et al., Applied Intelligence 2020 |
| Anytime | Chaudhuri & Narasayya, Microsoft Research 2020 |
| DBA Bandits | Perera et al., VLDB 2021 |

---

## Hardware

All experiments were run on: Intel Xeon E5-2680 v4 (28 cores), 64 GB RAM, Ubuntu 22.04, PostgreSQL 14.9, HypoPG 1.4.0.

Training times: ~13 hours per seed (single seed, no parallelism). Inference: ~1.5 ms per index decision.

---
