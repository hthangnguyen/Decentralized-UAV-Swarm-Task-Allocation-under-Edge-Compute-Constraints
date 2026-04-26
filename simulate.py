"""
simulate.py — Episode runner and metrics collector.

Usage
-----
    python simulate.py                      # default: 3 algorithms x 10 episodes
    python simulate.py --n_uavs 8 --ticks 500 --episodes 20
    python simulate.py --algo local         # single algorithm

Outputs a results table to stdout and saves results/metrics.json.
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Type

import numpy as np

from env import SwarmEnv
from consensus import LocalConsensus, GreedyBaseline, CentralizedOracle


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class EpisodeMetrics:
    algorithm: str
    episode: int
    tasks_completed: int
    tasks_spawned: int
    completion_rate: float          # tasks_completed / tasks_spawned
    mean_completion_time: float     # ticks from spawn to complete
    total_messages: int             # inter-UAV + UAV-base comms
    mean_distance_per_uav: float
    connectivity_ratio: float       # fraction of ticks UAVs had ≥1 neighbor


def run_episode(
    algo,
    n_uavs: int = 5,
    area: float = 100.0,
    comm_range: float = 35.0,
    uav_speed: float = 2.0,
    task_rate: float = 0.10,
    n_ticks: int = 300,
    seed: int = 0,
    episode: int = 0,
) -> EpisodeMetrics:

    env = SwarmEnv(
        n_uavs=n_uavs,
        area_size=area,
        comm_range=comm_range,
        uav_speed=uav_speed,
        task_rate=task_rate,
        seed=seed,
    )
    obs = env.reset(seed=seed)

    total_msgs = 0
    connected_ticks = 0   # ticks where avg UAV has ≥1 neighbor

    for _ in range(n_ticks):
        # Run assignment round
        msgs = algo.assign(obs)
        total_msgs += msgs

        # Step environment
        obs = env.step()

        # Track connectivity
        avg_neighbors = np.mean([len(u.neighbors) for u in env.uavs])
        if avg_neighbors >= 1.0:
            connected_ticks += 1

    all_tasks = env.tasks
    done_tasks = [t for t in all_tasks if t.completed]
    completion_times = [
        t.complete_tick - t.spawn_tick
        for t in done_tasks
        if t.complete_tick is not None
    ]

    return EpisodeMetrics(
        algorithm=algo.name,
        episode=episode,
        tasks_completed=len(done_tasks),
        tasks_spawned=len(all_tasks),
        completion_rate=len(done_tasks) / max(len(all_tasks), 1),
        mean_completion_time=float(np.mean(completion_times)) if completion_times else float("nan"),
        total_messages=total_msgs,
        mean_distance_per_uav=float(np.mean([u.distance_flown for u in env.uavs])),
        connectivity_ratio=connected_ticks / n_ticks,
    )


# ── Multi-episode runner ──────────────────────────────────────────────────────

def run_benchmark(
    algos,
    n_episodes: int = 10,
    n_uavs: int = 5,
    n_ticks: int = 300,
    comm_range: float = 35.0,
    task_rate: float = 0.10,
    base_seed: int = 42,
) -> list[EpisodeMetrics]:

    all_metrics: list[EpisodeMetrics] = []

    for algo in algos:
        print(f"\n  Running {algo.name} ...")
        for ep in range(n_episodes):
            seed = base_seed + ep
            m = run_episode(
                algo,
                n_uavs=n_uavs,
                comm_range=comm_range,
                task_rate=task_rate,
                n_ticks=n_ticks,
                seed=seed,
                episode=ep,
            )
            all_metrics.append(m)
            print(f"    ep {ep+1:2d}  done={m.tasks_completed:3d}/{m.tasks_spawned:3d}"
                  f"  rate={m.completion_rate:.2f}"
                  f"  msgs={m.total_messages:5d}"
                  f"  t_complete={m.mean_completion_time:.1f}")

    return all_metrics


# ── Summary table ─────────────────────────────────────────────────────────────

def summarise(metrics: list[EpisodeMetrics]):
    from collections import defaultdict
    import statistics

    groups: dict[str, list[EpisodeMetrics]] = defaultdict(list)
    for m in metrics:
        groups[m.algorithm].append(m)

    algo_order = [LocalConsensus.name, GreedyBaseline.name, CentralizedOracle.name]
    ordered = [k for k in algo_order if k in groups] + [k for k in groups if k not in algo_order]

    col_w = [34, 12, 14, 12, 10, 12]
    header = ["Algorithm", "Completion%", "Avg time(t)", "Msgs/ep", "Conn%", "Dist/UAV"]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "|" + "|".join(f"{{:<{w}}}" for w in col_w) + "|"

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(sep)
    print(fmt.format(*header))
    print(sep)

    summary_data = {}
    for name in ordered:
        ms = groups[name]
        cr  = statistics.mean(m.completion_rate for m in ms) * 100
        ct  = statistics.mean(m.mean_completion_time for m in ms if not np.isnan(m.mean_completion_time))
        msg = statistics.mean(m.total_messages for m in ms)
        con = statistics.mean(m.connectivity_ratio for m in ms) * 100
        dist= statistics.mean(m.mean_distance_per_uav for m in ms)
        print(fmt.format(
            name[:col_w[0]-1],
            f"{cr:.1f}%",
            f"{ct:.1f}",
            f"{msg:.0f}",
            f"{con:.1f}%",
            f"{dist:.1f}",
        ))
        summary_data[name] = {
            "completion_pct": round(cr, 2),
            "avg_completion_time": round(ct, 2),
            "avg_msgs_per_episode": round(msg, 1),
            "connectivity_pct": round(con, 2),
            "avg_dist_per_uav": round(dist, 2),
        }
    print(sep)

    # Efficiency vs oracle
    oracle_cr = summary_data.get(CentralizedOracle.name, {}).get("completion_pct", 100)
    oracle_msg = summary_data.get(CentralizedOracle.name, {}).get("avg_msgs_per_episode", 1)
    local_cr   = summary_data.get(LocalConsensus.name,    {}).get("completion_pct", 0)
    local_msg  = summary_data.get(LocalConsensus.name,    {}).get("avg_msgs_per_episode", 1)

    if oracle_cr > 0 and oracle_msg > 0:
        perf_ratio = local_cr / oracle_cr * 100
        msg_saving = (1 - local_msg / oracle_msg) * 100
        print(f"\n  LocalConsensus achieves {perf_ratio:.1f}% of Oracle completion rate")
        print(f"  while using {msg_saving:.1f}% fewer messages than the Oracle.")

    return summary_data


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UAV swarm task-allocation benchmark")
    parser.add_argument("--n_uavs",    type=int,   default=5)
    parser.add_argument("--ticks",     type=int,   default=300)
    parser.add_argument("--episodes",  type=int,   default=10)
    parser.add_argument("--comm_range",type=float, default=35.0)
    parser.add_argument("--task_rate", type=float, default=0.10)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--algo",      type=str,   default="all",
                        choices=["all", "local", "greedy", "oracle"])
    args = parser.parse_args()

    algo_map = {
        "local":  LocalConsensus,
        "greedy": GreedyBaseline,
        "oracle": CentralizedOracle,
    }

    if args.algo == "all":
        algos = [LocalConsensus(), GreedyBaseline(), CentralizedOracle()]
    else:
        algos = [algo_map[args.algo]()]

    print(f"\nUAV Swarm Task-Allocation Benchmark")
    print(f"  UAVs={args.n_uavs}  ticks={args.ticks}  episodes={args.episodes}")
    print(f"  comm_range={args.comm_range}  task_rate={args.task_rate}")

    t0 = time.time()
    metrics = run_benchmark(
        algos,
        n_episodes=args.episodes,
        n_uavs=args.n_uavs,
        n_ticks=args.ticks,
        comm_range=args.comm_range,
        task_rate=args.task_rate,
        base_seed=args.seed,
    )
    print(f"\n  Total wall time: {time.time()-t0:.1f}s")

    summary = summarise(metrics)

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    payload = {
        "config": vars(args),
        "summary": summary,
        "episodes": [asdict(m) for m in metrics],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved to results/metrics.json")


if __name__ == "__main__":
    main()
