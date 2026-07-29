# DB-GNN Final Reproducible Package

This folder is the clean final package for the DB-GNN experiment. It keeps only
the final model code, retained final data, final figures, and trained DB-GNN
checkpoints. Intermediate smoke tests, pilot runs, failed branches, ablation
code, and deprecated baselines are intentionally excluded.

## 1. Directory Layout

```text
DB_GNN_final_package/
  README.md
  code/
    config.py
    env.py
    graph_models.py
    db_gnn_evaluate.py
    paper_plot_results.py
    check_results.py
    requirements.txt
    environment.yml
    configs/
      db_gnn_grid_lite.yaml
      db_gnn_scale_lite.yaml
      db_gnn_users100.yaml
    scripts/
      plot_final_results.ps1
      evaluate_db_gnn_from_checkpoints.ps1
  data/
    raw/
      final_db_gnn_main_results.csv
      final_db_gnn_only.csv
    summary/
      final_db_gnn_main_summary.csv
    tables/
      table_overall.csv
      table_users200.csv
      table_db_gnn_by_users.csv
  models/
    checkpoints/
      trained GraphSAGE-DQN and Candidate GraphSAGE checkpoint pairs
  results/
    figures/
      main/
      main_errorbars/
      replot_main/
      replot_main_errorbars/
    model_architecture.png
```

## 2. Final Model

The final method is **DB-GNN**.

DB-GNN uses two graph Q-networks:

- `GraphSAGE-DQN`: ranks valid channel-power actions from the beam-interference graph.
- `Candidate GraphSAGE`: provides a second candidate ranking with a stronger candidate-action bias.

At inference time, DB-GNN:

1. Constructs a beam-interference graph from the current LEO environment state.
2. Applies a structural action mask to remove illegal channel-power actions.
3. Takes the Top-10 valid actions from each graph network.
4. Merges and deduplicates the candidate actions.
5. Scores only this small candidate set with the simulator's one-step utility.
6. Executes the action with the highest one-step utility.

The algorithm id in retained CSV files is:

```text
gnn_topk10_selector_sage_candidate
```

In all final figures and paper-facing descriptions, this method is labeled as:

```text
DB-GNN
```

## 3. Retained Data

### `data/raw/final_db_gnn_main_results.csv`

Main raw result table. One row corresponds to one:

```text
algorithm x user scale x seed x traffic distribution x weight preset
```

Included algorithms:

- `random`
- `qlearning`
- `dqn`
- `greedy_z`
- `gnn_topk10_selector_sage_candidate` = DB-GNN

Coverage:

- Users: `25, 50, 75, 100, 125, 150, 200`
- Seeds: `0, 1, 2`
- Traffic: `uniform, hotspot`
- Weights: `equal`

Total rows: `210`.

### `data/raw/final_db_gnn_only.csv`

DB-GNN rows only. Total rows: `42`.

### `data/summary/final_db_gnn_main_summary.csv`

Mean and standard deviation grouped by:

```text
algo, users, traffic, weight_name
```

### `data/tables/*.csv`

Small paper-ready result tables:

- `table_overall.csv`
- `table_users200.csv`
- `table_db_gnn_by_users.csv`

## 4. Main Results

Overall mean across all retained users, seeds, and traffic distributions:

| Method | Blocking rate | Spectral efficiency | Energy efficiency | Total power |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.3466 | 267.0902 | 0.2368 | 1349.6951 |
| Q-learning | 0.2454 | 315.5307 | 0.1847 | 1896.2406 |
| DQN | 0.2223 | 227.5635 | 0.2933 | 1395.2756 |
| Greedy-Z | 0.0599 | 402.6059 | 2.4116 | 229.1939 |
| DB-GNN | 0.0359 | 408.9071 | 2.4586 | 297.8579 |

At 200 users:

| Method | Blocking rate | Spectral efficiency | Energy efficiency | Total power |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.5233 | 346.2194 | 0.1697 | 2103.7151 |
| Q-learning | 0.4232 | 421.1346 | 0.1553 | 2807.4600 |
| DQN | 0.3938 | 353.6808 | 0.3534 | 1543.2984 |
| Greedy-Z | 0.1800 | 573.8545 | 1.5789 | 364.9590 |
| DB-GNN | 0.1220 | 595.1416 | 1.9902 | 337.3459 |

## 5. How To Check The Data

From the package root:

```powershell
python code\check_results.py data\raw\final_db_gnn_main_results.csv
```

Expected status:

```text
Overall: PASS_WITH_WARNINGS_OR_OK
```

The warning about hotspot not always being harder than uniform is retained
because traffic realization and admission dynamics can produce matched groups
where hotspot blocking is not strictly larger. This is a scenario note, not a
data corruption error.

## 6. How To Regenerate The Final Figures

From the package root:

```powershell
powershell -ExecutionPolicy Bypass -File code\scripts\plot_final_results.ps1
```

Outputs:

```text
results/figures/replot_main/
results/figures/replot_main_errorbars/
```

The original final figures are also retained in:

```text
results/figures/main/
results/figures/main_errorbars/
```

## 7. How To Re-evaluate DB-GNN From Checkpoints

The package includes the trained DB-GNN checkpoint pairs needed to recompute
DB-GNN evaluation rows.

From the package root:

```powershell
powershell -ExecutionPolicy Bypass -File code\scripts\evaluate_db_gnn_from_checkpoints.ps1
```

This writes:

```text
data/raw/db_gnn_recomputed_shards/
data/raw/db_gnn_recomputed.csv
```

This command can take time because DB-GNN evaluates a Top-10 candidate set for
each arriving user.

## 8. What Was Removed

The package intentionally removes:

- smoke-test outputs;
- pilot outputs;
- stress diagnostics;
- deprecated Greedy-SINR results;
- Top-3/Top-5 intermediate result tables and plotting labels;
- gated/load-gated/candidate-penalty experimental branches;
- old logs and temporary timing files;
- source paper extraction artifacts.

This package is therefore focused on the final DB-GNN model and the final
paper-facing result set.

## 9. Scientific Boundary

The results are reproducible under the explicit simulator configuration in this
package. They should be described as DB-GNN results under the retained
paper-like simulation assumptions, not as an exact numerical reproduction of
the original paper, because the original paper does not fully specify all
random traffic-generation, normalization, and implementation details.
