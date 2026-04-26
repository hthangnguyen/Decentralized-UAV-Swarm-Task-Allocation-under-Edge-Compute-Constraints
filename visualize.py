"""
visualize.py — Plots and animation for the UAV swarm simulation.

Modes
-----
  python visualize.py --mode animate    live matplotlib animation (one episode)
  python visualize.py --mode results    bar/line charts from results/metrics.json
  python visualize.py --mode snapshot   saves a single-frame PNG for README

Run simulate.py first to generate results/metrics.json before using --mode results.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from env import SwarmEnv
from consensus import LocalConsensus, GreedyBaseline, CentralizedOracle
from edge import EGSAssistedConsensus


# ── Colour palette ────────────────────────────────────────────────────────────
C_UAV_FREE  = "#378ADD"
C_UAV_BUSY  = "#1D9E75"
C_TASK_OPEN = "#EF9F27"
C_TASK_ASSI = "#639922"
C_COMM      = "#AFA9EC"
C_EGS       = "#E74C3C"  # Red for Edge Ground Station
C_OFFLOAD   = "#9B59B6"  # Purple for offloaded processing
C_BG        = "#F8F8F6"


# ── Animation ─────────────────────────────────────────────────────────────────

def animate(n_uavs: int = 5, comm_range: float = 35.0, n_ticks: int = 400, seed: int = 0, save_path: str = None):
    algo = EGSAssistedConsensus(area_size=100.0)
    env  = SwarmEnv(n_uavs=n_uavs, comm_range=comm_range, task_rate=0.08, seed=seed)
    obs  = env.reset(seed=seed)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, env.area); ax.set_ylim(0, env.area)
    ax.set_aspect("equal"); ax.set_title("LocalConsensus UAV Swarm", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])

    legend_elements = [
        mpatches.Patch(color=C_UAV_FREE,  label="UAV (free)"),
        mpatches.Patch(color=C_UAV_BUSY,  label="UAV (on task)"),
        mpatches.Patch(color=C_TASK_OPEN, label="Task (open)"),
        mpatches.Patch(color=C_TASK_ASSI, label="Task (assigned)"),
        Line2D([0],[0], color=C_EGS, marker='s', linestyle='None', markersize=10, label="Edge Ground Station"),
        Line2D([0],[0], color=C_COMM, linewidth=0.8, linestyle="--", label="Comm link"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    
    frames = []
    if not save_path:
        plt.ion()
        plt.show()

    print(f"Running simulation ({'saving to ' + save_path if save_path else 'live view'})...")

    for tick in range(n_ticks):
        msgs = algo.assign(obs)
        pre_completed = {t.id for t in env.tasks if t.completed}
        obs = env.step()

        # Release compute for completed tasks
        for t in env.tasks:
            if t.completed and t.id not in pre_completed and t.assigned_to is not None:
                uav = next((u for u in env.uavs if u.id == t.assigned_to), None)
                if uav: algo.on_task_complete(uav, t)

        ax.cla()
        ax.set_facecolor(C_BG)
        ax.set_xlim(0, env.area); ax.set_ylim(0, env.area)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        # Edge Ground Station (EGS)
        ax.scatter(env.area/2, env.area/2, s=200, marker="s", color=C_EGS, zorder=2, edgecolors="white", linewidths=1.5)
        ax.text(env.area/2, env.area/2 + 5, "EGS", fontsize=10, ha="center", fontweight="bold", color=C_EGS)

        # Comm links
        uav_map = {u.id: u for u in env.uavs}
        drawn_pairs = set()
        for u in env.uavs:
            # Show link to EGS
            ax.plot([u.x, env.area/2], [u.y, env.area/2], color=C_COMM, lw=0.5, ls=":", alpha=0.3)
            
            for nid in u.neighbors:
                pair = tuple(sorted((u.id, nid)))
                if pair in drawn_pairs: continue
                drawn_pairs.add(pair)
                v = uav_map[nid]
                ax.plot([u.x, v.x], [u.y, v.y],
                        color=C_COMM, linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

        # Tasks
        for t in env.tasks:
            if t.completed: continue
            col = C_TASK_ASSI if t.assigned_to is not None else C_TASK_OPEN
            edge_col = C_OFFLOAD if t.processed_at == "edge" else "white"
            ax.scatter(t.x, t.y, s=100, marker="^", color=col, zorder=3, edgecolors=edge_col, linewidths=2.0)
            ax.text(t.x, t.y - 4, f"T{t.id}", fontsize=7, ha="center", color="#555")

        # UAVs
        for u in env.uavs:
            col  = C_UAV_BUSY if u.target_task_id is not None else C_UAV_FREE
            ax.scatter(u.x, u.y, s=130, marker="o", color=col, zorder=4, edgecolors="white", linewidths=1.0)
            ax.text(u.x, u.y, str(u.id), fontsize=7, ha="center", va="center",
                    color="white", fontweight="bold", zorder=5)
            # Load label
            ax.text(u.x, u.y + 5, f"L:{u.compute_load:.1f}", fontsize=6, ha="center", color="#333")
            
            comm_circle = plt.Circle((u.x, u.y), env.comm_range,
                                     fill=False, edgecolor=C_COMM, linewidth=0.4, alpha=0.25)
            ax.add_patch(comm_circle)

        done  = len([t for t in env.tasks if t.completed])
        total = len(env.tasks)
        ax.set_title(f"EGS-Assisted Swarm  |  tick {tick+1}/{n_ticks}  |  done {done}/{total}", fontsize=11)
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.7)

        if save_path:
            import io
            from PIL import Image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=80)
            buf.seek(0)
            frames.append(Image.open(buf))
            if tick % 50 == 0:
                print(f"  Frame {tick}/{n_ticks} captured...")
        else:
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.04)

    if save_path:
        print(f"Saving {len(frames)} frames to {save_path}...")
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"Done.")
    else:
        plt.ioff()
        print(f"\nAnimation complete. Completed {done}/{total} tasks.")
        plt.show()


def plot_results(metrics_path: str = "results/metrics.json"):
    with open(metrics_path) as f:
        data = json.load(f)

    summary = data["summary"]
    names   = list(summary.keys())
    short   = {
        "LocalConsensus (proposed)":     "LocalConsensus",
        "GreedyBaseline (no coordination)": "Greedy",
        "CentralizedOracle (upper bound)":  "Oracle",
        "EGS-Assisted Consensus (proposed)": "EGS-Assisted\n(Proposed)",
    }
    labels = [short.get(n, n) for n in names]

    completion = [summary[n]["completion_pct"]      for n in names]
    comp_time  = [summary[n]["avg_completion_time"] for n in names]
    msgs       = [summary[n]["avg_msgs_per_episode"] for n in names]

    # SAGIN/EGS metrics for the assisted algo
    offload_ratio = 0
    egs_corr = 0
    if "EGS-Assisted Consensus (proposed)" in summary:
        s = summary["EGS-Assisted Consensus (proposed)"]
        # we need to re-extract from raw episode data or assume these keys exist in summary
        # actually, let's just use what we have in summary if I updated summarise()
    
    bar_colors = ["#378ADD", "#BA7517", "#1D9E75", "#9B59B6"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), facecolor="white")
    fig.suptitle("UAV Swarm Extended Benchmark Results (SAGIN + EGS)", fontsize=13, fontweight="bold", y=1.01)

    def bar(ax, vals, title, ylabel, ymax=None):
        bars = ax.bar(labels, vals, color=bar_colors[:len(names)], width=0.5, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        if ymax: ax.set_ylim(0, ymax)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + (ymax or max(vals))*0.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.tick_params(axis="y", labelsize=9)

    bar(axes[0], completion,  "Task Completion Rate",      "Completion %",   ymax=105)
    bar(axes[1], comp_time,   "Avg Task Completion Time",  "Ticks",          ymax=max(comp_time)*1.2)
    bar(axes[2], msgs,        "Communication Overhead",    "Messages / ep",  ymax=max(msgs)*1.2)
    
    # 4th plot: Offloading effectiveness
    if "EGS-Assisted Consensus (proposed)" in summary:
        eps = data["episodes"]
        assist_eps = [e for e in eps if e["algorithm"] == "EGS-Assisted Consensus (proposed)"]
        if assist_eps:
            offloads = [e["edge_offloaded"] for e in assist_eps]
            locals_  = [e["local_processed"] for e in assist_eps]
            corrs    = [e["egs_corrections"] for e in assist_eps]
            
            x = np.arange(len(assist_eps))
            axes[3].bar(x, locals_,  label="Local", color=C_UAV_BUSY)
            axes[3].bar(x, offloads, bottom=locals_, label="Edge", color=C_OFFLOAD)
            axes[3].set_title("Assisted: Local vs Edge", fontsize=11)
            axes[3].set_ylabel("Tasks", fontsize=9)
            axes[3].legend(fontsize=8)
            axes[3].spines["top"].set_visible(False)
            axes[3].spines["right"].set_visible(False)
    else:
        axes[3].axis("off")

    plt.tight_layout()
    out = Path("results/benchmark_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved chart to {out}")
    plt.show()


# ── Snapshot (for README) ─────────────────────────────────────────────────────

def snapshot(n_uavs: int = 5, comm_range: float = 35.0, seed: int = 7,
             out_path: str = "results/snapshot.png"):
    algo = EGSAssistedConsensus(area_size=100.0)
    env  = SwarmEnv(n_uavs=n_uavs, comm_range=comm_range, task_rate=0.08, seed=seed)
    obs  = env.reset(seed=seed)

    # Run a few ticks to get an interesting state
    for _ in range(80):
        algo.assign(obs)
        pre_comp = {t.id for t in env.tasks if t.completed}
        obs = env.step()
        for t in env.tasks:
            if t.completed and t.id not in pre_comp and t.assigned_to is not None:
                uav = next((u for u in env.uavs if u.id == t.assigned_to), None)
                if uav: algo.on_task_complete(uav, t)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, env.area); ax.set_ylim(0, env.area)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("EGS-Assisted UAV Swarm (SAGIN + Edge AI)", fontsize=11, pad=10)

    # Edge Ground Station (EGS)
    ax.scatter(env.area/2, env.area/2, s=150, marker="s", color=C_EGS, zorder=2, edgecolors="white", lw=1)

    uav_map = {u.id: u for u in env.uavs}
    drawn = set()
    for u in env.uavs:
        # Link to EGS
        ax.plot([u.x, env.area/2], [u.y, env.area/2], color=C_COMM, lw=0.4, ls=":", alpha=0.3)
        
        for nid in u.neighbors:
            pair = tuple(sorted((u.id, nid)))
            if pair in drawn: continue
            drawn.add(pair)
            v = uav_map[nid]
            ax.plot([u.x,v.x],[u.y,v.y], color=C_COMM, lw=0.8, ls="--", alpha=0.5)

    for t in env.tasks:
        if t.completed: continue
        col = C_TASK_ASSI if t.assigned_to is not None else C_TASK_OPEN
        edge_col = C_OFFLOAD if t.processed_at == "edge" else "white"
        ax.scatter(t.x, t.y, s=100, marker="^", color=col, zorder=3, edgecolors=edge_col, lw=1.5)

    for u in env.uavs:
        col = C_UAV_BUSY if u.target_task_id is not None else C_UAV_FREE
        ax.scatter(u.x, u.y, s=140, color=col, zorder=4, edgecolors="white", lw=1.0)
        ax.text(u.x, u.y, str(u.id), fontsize=7, ha="center", va="center",
                color="white", fontweight="bold", zorder=5)
        ax.add_patch(plt.Circle((u.x,u.y), env.comm_range,
                                fill=False, edgecolor=C_COMM, lw=0.4, alpha=0.3))

    legend_elements = [
        mpatches.Patch(color=C_UAV_FREE,  label="UAV (free)"),
        mpatches.Patch(color=C_UAV_BUSY,  label="UAV (on task)"),
        mpatches.Patch(color=C_TASK_OPEN, label="Task (open)"),
        mpatches.Patch(color=C_TASK_ASSI, label="Task (assigned)"),
        Line2D([0],[0], color=C_EGS, marker='s', linestyle='None', label="EGS"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.8)

    Path(out_path).parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Snapshot saved to {out_path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["animate", "results", "snapshot"], default="snapshot")
    parser.add_argument("--n_uavs",     type=int,   default=5)
    parser.add_argument("--comm_range", type=float, default=35.0)
    parser.add_argument("--ticks",      type=int,   default=200)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--save",       action="store_true", help="Save animation as GIF")
    parser.add_argument("--filename",   type=str,   default="results/simulation.gif")
    args = parser.parse_args()

    if args.mode == "animate":
        save_path = args.filename if args.save else None
        animate(args.n_uavs, args.comm_range, args.ticks, args.seed, save_path=save_path)
    elif args.mode == "results":
        plot_results()
    else:
        snapshot(args.n_uavs, args.comm_range, args.seed)


if __name__ == "__main__":
    main()
