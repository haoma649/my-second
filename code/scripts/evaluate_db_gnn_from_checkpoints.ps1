param(
  [int]$EvalEpisodes = 10,
  [int]$TopK = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path "$PSScriptRoot\..\.."
$ShardDir = Join-Path $Root "data\raw\db_gnn_recomputed_shards"
New-Item -ItemType Directory -Force -Path $ShardDir | Out-Null
Push-Location "$Root\code"

function Invoke-DbGnnEval($config, $checkpointDir, $users, $seed, $traffic) {
  $sagePath = "..\models\checkpoints\${checkpointDir}\graph_sage_${traffic}_users${users}_seed${seed}_equal.pt"
  $candidatePath = "..\models\checkpoints\${checkpointDir}\candidate_sage_${traffic}_users${users}_seed${seed}_equal.pt"
  if (!(Test-Path $sagePath) -or !(Test-Path $candidatePath)) {
    throw "Missing DB-GNN checkpoint pair for users=$users seed=$seed traffic=$traffic"
  }
  python db_gnn_evaluate.py `
    --config $config `
    --sage-checkpoint $sagePath `
    --candidate-checkpoint $candidatePath `
    --users $users `
    --seeds $seed `
    --traffic $traffic `
    --weight-preset equal `
    --eval-episodes $EvalEpisodes `
    --top-k $TopK `
    --algo-name db_gnn `
    --out "..\data\raw\db_gnn_recomputed_shards\db_gnn_users${users}_${traffic}_seed${seed}.csv"
}

foreach ($users in 25, 75, 125) {
  foreach ($seed in 0, 1, 2) {
    foreach ($traffic in "uniform", "hotspot") {
      Invoke-DbGnnEval "configs\db_gnn_grid_lite.yaml" "checkpoint_safe_grid_lite_users${users}_seed${seed}" $users $seed $traffic
    }
  }
}

foreach ($users in 50, 150, 200) {
  foreach ($seed in 0, 1, 2) {
    foreach ($traffic in "uniform", "hotspot") {
      Invoke-DbGnnEval "configs\db_gnn_scale_lite.yaml" "checkpoint_safe_p05_users${users}_seed${seed}" $users $seed $traffic
    }
  }
}

foreach ($seed in 0, 1, 2) {
  foreach ($traffic in "uniform", "hotspot") {
    Invoke-DbGnnEval "configs\db_gnn_users100.yaml" "checkpoint_safe_p05_100_seed${seed}" 100 $seed $traffic
  }
}

Pop-Location

$combined = Join-Path $Root "data\raw\db_gnn_recomputed.csv"
Get-ChildItem -Path $ShardDir -Filter "*.csv" |
  Sort-Object Name |
  ForEach-Object -Begin { $first = $true } -Process {
    if ($first) {
      Get-Content $_.FullName
      $first = $false
    } else {
      Get-Content $_.FullName | Select-Object -Skip 1
    }
  } | Set-Content $combined

Write-Host "wrote $combined"
