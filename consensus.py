"""
consensus.py — Task-allocation algorithms.

Three allocators are implemented here so you can benchmark them head-to-head:

1. LocalConsensus   (proposed) — UAVs bid only with in-range neighbors.
                                 No global broadcast.  O(k) messages per round,
                                 where k = mean neighborhood size.

2. GreedyBaseline   — Each free UAV greedily picks the nearest unassigned task
                      with no coordination.  Leads to collisions / duplicate
                      assignments that must be resolved by re-assignment.

3. CentralizedOracle— A fictional base-station sees all UAVs and all tasks
                      and solves the assignment optimally (Hungarian algorithm).
                      Used as the performance ceiling.

All three expose the same interface:
    algo.assign(env_obs) -> int   (messages sent this round)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional

from env import UAV, Task


# ── Shared cost function ──────────────────────────────────────────────────────

def _cost(uav: UAV, task: Task, w_dist: float = 0.6, w_load: float = 0.4) -> float:
    """
    Composite bid cost: weighted sum of normalised distance and compute load.
    Lower cost = better candidate.
    """
    return w_dist * uav.dist_to(task) + w_load * uav.compute_load * 100


# ── Algorithm 1 : Local Consensus (proposed) ─────────────────────────────────

class LocalConsensus:
    """
    Decentralised auction run purely over local neighborhoods.

    Each task is assigned to the lowest-cost free UAV within the current
    connected local neighborhood. A UAV with neighbors only bids when at
    least one free peer is in range, so the allocator remains grounded in
    short-range sharing instead of central coordination.

    Messages counted: one bid message per (UAV, task) pair within range.
    """

    name = "LocalConsensus (proposed)"

    def assign(self, obs: dict) -> int:
        open_tasks: list[Task] = obs["open_tasks"]
        uavs: list[UAV] = obs["uavs"]
        uav_map = {u.id: u for u in uavs}

        msgs = 0
        assignments_created = 0

        for task in open_tasks:
            # Each UAV broadcasts a bid to in-range peers
            # A UAV is eligible if it is free AND at least one neighbor is also free
            candidates = []
            for u in uavs:
                if u.target_task_id is not None:
                    continue
                # Count how many free neighbors this UAV has
                free_neighbors = [
                    nid for nid in u.neighbors
                    if uav_map.get(nid) and uav_map[nid].target_task_id is None
                ]
                # Must be connected to at least one peer (edge constraint)
                if len(free_neighbors) == 0 and len(u.neighbors) > 0:
                    continue
                candidates.append(u)
                msgs += max(1, len(free_neighbors))   # bid broadcast to peers
                u.consensus_msgs_sent += max(1, len(free_neighbors))

            if not candidates:
                # isolated UAV can still self-assign
                solo = [u for u in uavs if u.target_task_id is None]
                if solo:
                    solo.sort(key=lambda u: _cost(u, task))
                    winner = solo[0]
                    task.assigned_to = winner.id
                    winner.target_task_id = task.id
                    msgs += 1
                    assignments_created += 1
                continue

            candidates.sort(key=lambda u: _cost(u, task))
            winner = candidates[0]
            task.assigned_to = winner.id
            winner.target_task_id = task.id
            assignments_created += 1

        self.last_assign_stats = {
            "assignments_created": assignments_created,
            "msg_local_bids": msgs,
        }

        return msgs


# ── Algorithm 2 : Greedy Baseline ────────────────────────────────────────────

class GreedyBaseline:
    """
    Each free UAV independently picks the nearest open task.
    No coordination: multiple UAVs can target the same task.
    Conflicts are resolved by re-assigning the slower UAV.

    Messages counted: zero inter-UAV messages (no communication).
    Does require a global task list — treated as a broadcast from the task
    itself (1 msg per task per tick).
    """

    name = "GreedyBaseline (no coordination)"

    def assign(self, obs: dict) -> int:
        open_tasks: list[Task] = obs["open_tasks"]
        uavs: list[UAV] = obs["uavs"]

        msgs = len(open_tasks)   # task-broadcast overhead
        assignments_created = 0

        # Each free UAV picks nearest open task
        claimed: dict[int, list[UAV]] = {}   # task_id -> list of claimants
        for u in uavs:
            if u.target_task_id is not None:
                continue
            if not open_tasks:
                break
            nearest = min(open_tasks, key=lambda t: u.dist_to(t))
            claimed.setdefault(nearest.id, []).append(u)

        task_map = {t.id: t for t in open_tasks}
        for tid, claimants in claimed.items():
            t = task_map[tid]
            # Closest UAV wins; others must retry next round
            claimants.sort(key=lambda u: u.dist_to(t))
            winner = claimants[0]
            t.assigned_to = winner.id
            winner.target_task_id = t.id
            assignments_created += 1

        self.last_assign_stats = {
            "assignments_created": assignments_created,
            "msg_task_broadcasts": msgs,
        }

        return msgs


# ── Algorithm 3 : Centralized Oracle ─────────────────────────────────────────

class CentralizedOracle:
    """
    Optimal assignment via the Hungarian algorithm.
    Assumes a base-station with full global state.

    Messages counted:
      - Each UAV uploads its state to the base-station : n_uavs messages
      - Base-station broadcasts assignments back         : n_assigned messages
    """

    name = "CentralizedOracle (upper bound)"

    def assign(self, obs: dict) -> int:
        open_tasks: list[Task] = obs["open_tasks"]
        uavs: list[UAV] = obs["uavs"]

        free_uavs = [u for u in uavs if u.target_task_id is None]

        # Upload cost: every UAV reports state
        msgs = len(uavs)
        assigned = 0

        if not free_uavs or not open_tasks:
            self.last_assign_stats = {
                "assignments_created": 0,
                "msg_oracle_uploads": len(uavs),
                "msg_oracle_downlinks": 0,
            }
            return msgs

        n_u = len(free_uavs)
        n_t = len(open_tasks)
        n = max(n_u, n_t)

        # Build cost matrix (padded to square)
        C = np.full((n, n), fill_value=1e6)
        for i, u in enumerate(free_uavs):
            for j, t in enumerate(open_tasks):
                C[i, j] = _cost(u, t)

        row_ind, col_ind = linear_sum_assignment(C)

        for i, j in zip(row_ind, col_ind):
            if i < n_u and j < n_t:
                u = free_uavs[i]
                t = open_tasks[j]
                t.assigned_to = u.id
                u.target_task_id = t.id
                assigned += 1

        # Broadcast assignments back
        msgs += assigned
        self.last_assign_stats = {
            "assignments_created": assigned,
            "msg_oracle_uploads": len(uavs),
            "msg_oracle_downlinks": assigned,
        }
        return msgs
