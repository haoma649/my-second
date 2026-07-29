"""Evaluate the final DB-GNN policy.

DB-GNN uses two graph Q-networks to propose valid channel-power actions:

1. GraphSAGE-DQN ranks actions from the beam-interference graph.
2. Candidate GraphSAGE provides a second candidate ranking.

The final policy takes the Top-K valid actions from both networks, removes
duplicates, and evaluates only this small candidate set with the simulator's
one-step utility. This file is intentionally self-contained and does not expose
earlier experimental variants.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from config import WEIGHT_PRESETS, load_configs
from env import LEOEnv
from graph_models import CandidateGraphSageDQN, GraphDQN, GraphSageDQN


def load_graph_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    model_name = ckpt.get("gnn_model", "graph_sage")
    node_features = int(ckpt["node_features"])
    num_actions = int(ckpt["num_actions"])
    if model_name == "graph_dqn":
        model = GraphDQN(node_features, num_actions)
    elif model_name == "graph_sage":
        model = GraphSageDQN(node_features, num_actions)
    elif model_name == "candidate_sage":
        model = CandidateGraphSageDQN(
            node_features,
            num_actions,
            power_penalty=float(ckpt.get("candidate_power_penalty", 0.5)),
        )
    else:
        raise ValueError(f"Unknown checkpoint gnn_model={model_name}")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def graph_q_values(model, env: LEOEnv, device: str) -> np.ndarray:
    obs = env.get_graph_observation()
    current_beam = env.users[env.current_index].beam if env.current_index < len(env.users) else 0
    with torch.no_grad():
        q = model(
            torch.tensor(obs["node_features"][None], dtype=torch.float32, device=device),
            torch.tensor(obs["edge_index"], dtype=torch.long, device=device),
            torch.tensor(obs["edge_weight"], dtype=torch.float32, device=device),
            torch.tensor([current_beam], dtype=torch.long, device=device),
        ).cpu().numpy()[0]
    return q


def masked_topk(q: np.ndarray, mask: np.ndarray, k: int) -> list[int]:
    scores = q.copy()
    scores[~mask] = -np.inf
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) == 0:
        return [0]
    k = max(1, min(int(k), len(valid)))
    top = valid[np.argpartition(scores[valid], -k)[-k:]]
    top = top[np.argsort(scores[top])[::-1]]
    return [int(a) for a in top]


def one_step_objective(env: LEOEnv, action: int) -> float:
    _, _, info = env.simulate_action(action)
    return float(info.get("objective", env.compute_objective()))


def evaluate_episode(sim_cfg, args, sage, candidate, eval_seed: int) -> dict[str, float]:
    env = LEOEnv(
        sim_cfg,
        num_users=args.users,
        weights=WEIGHT_PRESETS[args.weight_preset],
        seed=eval_seed,
        traffic_distribution=args.traffic,
    )
    env.reset(num_users=args.users, seed=eval_seed)
    done = False
    total_reward = 0.0
    sage_steps = 0
    candidate_steps = 0
    rejected_actions = 0
    invalid_rates = []
    evaluated_action_counts = []
    decision_times_ms = []

    while not done:
        start = time.perf_counter()
        mask = env.get_action_mask(include_qos=args.qos_action_mask)
        invalid_rates.append(1.0 - float(np.sum(mask)) / len(mask))

        sage_q = graph_q_values(sage, env, args.device)
        candidate_q = graph_q_values(candidate, env, args.device)
        sage_actions = masked_topk(sage_q, mask, args.top_k)
        candidate_actions = masked_topk(candidate_q, mask, args.top_k)

        action_pool = []
        for action in [*sage_actions, *candidate_actions, 0]:
            if bool(mask[action]) and action not in action_pool:
                action_pool.append(int(action))

        scored = [(action, one_step_objective(env, action)) for action in action_pool]
        evaluated_action_counts.append(len(scored))
        sage_best = max((score for action, score in scored if action in sage_actions), default=-np.inf)
        action, best_obj = max(scored, key=lambda item: item[1])
        if action in candidate_actions and best_obj > sage_best:
            candidate_steps += 1
        else:
            sage_steps += 1
        if action == 0:
            rejected_actions += 1

        decision_times_ms.append((time.perf_counter() - start) * 1000.0)
        _, reward, done, _ = env.step(action)
        total_reward += reward

    row = env.metrics()
    row.update(
        {
            "episode_reward": total_reward,
            "invalid_action_rate": float(np.mean(invalid_rates)) if invalid_rates else np.nan,
            "candidate_steps": candidate_steps,
            "sage_steps": sage_steps,
            "db_gnn_reject_actions": rejected_actions,
            "db_gnn_evaluated_actions": float(np.mean(evaluated_action_counts)) if evaluated_action_counts else np.nan,
            "db_gnn_decision_time_ms": float(np.mean(decision_times_ms)) if decision_times_ms else np.nan,
        }
    )
    return row


def evaluate_one(sim_cfg, args, seed: int) -> dict[str, float]:
    sage = load_graph_model(args.sage_checkpoint, args.device)
    candidate = load_graph_model(args.candidate_checkpoint, args.device)
    base_seed = seed * 1_000_000 + 910_000
    episode_rows = [
        evaluate_episode(sim_cfg, args, sage, candidate, base_seed + ep)
        for ep in range(args.eval_episodes)
    ]
    row = {}
    for key in episode_rows[0]:
        values = [r[key] for r in episode_rows if isinstance(r.get(key), (int, float, np.floating))]
        if values:
            row[key] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    row.update(
        {
            "algo": args.algo_name,
            "method": args.algo_name,
            "users": args.users,
            "seed": seed,
            "traffic": args.traffic,
            "weight_name": args.weight_preset,
            "weights": args.weight_preset,
            "weight_se": WEIGHT_PRESETS[args.weight_preset][0],
            "weight_ee": WEIGHT_PRESETS[args.weight_preset][1],
            "weight_blocking": WEIGHT_PRESETS[args.weight_preset][2],
            "total_power": row["total_power_w"],
            "epsilon": 0.0,
            "train_steps": 0,
            "loss": np.nan,
            "replay_buffer_size": np.nan,
            "eval_episodes": args.eval_episodes,
            "top_k": args.top_k,
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/db_gnn_grid_lite.yaml")
    parser.add_argument("--sage-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--traffic", choices=["uniform", "poisson", "hotspot"], required=True)
    parser.add_argument("--weight-preset", choices=WEIGHT_PRESETS.keys(), default="equal")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--qos-action-mask", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--algo-name", default="db_gnn")
    parser.add_argument("--out", default="../data/raw/db_gnn_eval.csv")
    args = parser.parse_args()

    sim_cfg, _, _ = load_configs(args.config)
    rows = [evaluate_one(sim_cfg, args, seed) for seed in args.seeds]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    for row in rows:
        print(
            f"seed={row['seed']} traffic={row['traffic']} users={row['users']} "
            f"blocking={row['blocking_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
