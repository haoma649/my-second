Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path "$PSScriptRoot\..\.."
Push-Location "$Root\code"

python paper_plot_results.py `
  --csv ..\data\raw\final_db_gnn_main_results.csv `
  --out-dir ..\results\figures\replot_main `
  --traffic uniform hotspot `
  --weights equal `
  --algos random qlearning dqn greedy_z gnn_topk10_selector_sage_candidate `
  --split-traffic `
  --add-origin `
  --legend-loc best

python paper_plot_results.py `
  --csv ..\data\raw\final_db_gnn_main_results.csv `
  --out-dir ..\results\figures\replot_main_errorbars `
  --traffic uniform hotspot `
  --weights equal `
  --algos random qlearning dqn greedy_z gnn_topk10_selector_sage_candidate `
  --split-traffic `
  --error-bars `
  --add-origin `
  --legend-loc best

Pop-Location
