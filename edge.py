"""
edge.py — Edge Ground Station (EGS) and Offloading Model.

This module implements two extensions that create honest, structural
connections to the two lab papers:

────────────────────────────────────────────────────────────────────────────
1.  OffloadingModel   →  SAGIN paper connection
    (Le et al., "Integrated Computation Offloading, UAV Trajectory Control,
     Edge-Cloud and Radio Resource Allocation in SAGIN", IEEE TCC 2024)

    The SAGIN paper's core contribution is joint optimisation of offloading
    decisions, UAV trajectory, and radio resources.  Here we model the
    offloading decision layer:

    - Each task carries a compute_demand (fraction of UAV compute budget).
    - If assigning a task would push the UAV's compute_load over a threshold,
      the task is offloaded to the EGS instead of processed locally.
    - Offloading adds a transmission latency (proportional to distance to
      the EGS) but frees the UAV's compute budget, recovering flight speed
      (via env.UAV.effective_speed — the mobility-compute coupling).
    - This models the hybrid edge–cloud trade-off the SAGIN paper analyses:
      local processing is fast but constrained; offloading is slower but
      removes the compute bottleneck.

────────────────────────────────────────────────────────────────────────────
2.  EdgeGroundStation  →  UAV Swarm paper connection
    (Le et al., "Agentic AI Meets Edge Computing in Autonomous UAV Swarms",
     IEEE IoT Magazine 2026)

    The UAV Swarm paper proposes an EGS running a full-scale LLM (GPT-4.1)
    that validates and corrects the outputs of lightweight onboard agents
    (TinyLLaMA).  Here we implement the same architectural split without
    the LLM layer (using classical optimisation instead), but preserving
    the key mechanism: the EGS receives proposed assignments from onboard
    agents, detects conflicts or overloads, and issues corrections.

    The EGS:
    - Receives a proposed assignment dict from LocalConsensus (onboard agents).
    - Detects two failure modes:
        (a) Compute overload: UAV's load would exceed capacity.
        (b) Coverage gap: open tasks with no candidate UAV in range.
    - Issues corrections:
        (a) Reroutes overloaded tasks to a lower-load UAV (if reachable).
        (b) Dispatches the globally nearest free UAV to uncovered tasks.
    - Counts the extra uplink/downlink messages this validation requires,
      making the communication overhead of EGS-assisted coordination
      explicit and measurable.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from env import UAV, Task, SwarmEnv


# ── 1. Offloading Model ───────────────────────────────────────────────────────

class OffloadingModel:
    """
    Decides whether a UAV processes a task locally or offloads to the EGS.

    Decision rule (mirrors the SAGIN paper's compute-offloading subproblem):
      - Compute the UAV's projected load after taking the task.
      - If projected_load > overload_threshold  →  offload to EGS.
      - Offloading penalty: extra latency_per_unit * dist_to_egs ticks.
      - Local processing: compute_load increases by task.compute_demand.

    Parameters
    ----------
    egs_x, egs_y        : EGS position (default: centre of area)
    overload_threshold  : load fraction above which offloading is triggered
    latency_per_unit    : extra ticks added per unit distance to EGS
    """

    def __init__(
        self,
        egs_x: float = 50.0,
        egs_y: float = 50.0,
        overload_threshold: float = 0.75,
        latency_per_unit: float = 0.05,
    ):
        self.egs_x = egs_x
        self.egs_y = egs_y
        self.overload_threshold = overload_threshold
        self.latency_per_unit = latency_per_unit

    def decide(self, uav: UAV, task: Task) -> str:
        """
        Returns 'local' or 'edge', and updates UAV compute_load + task fields.
        """
        projected_load = uav.compute_load + task.compute_demand

        if projected_load > self.overload_threshold:
            # Offload to EGS
            dist_egs = float(np.hypot(uav.x - self.egs_x, uav.y - self.egs_y))
            task.offload_latency = self.latency_per_unit * dist_egs
            task.processed_at = "edge"
            uav.edge_offloaded += 1
            # Compute load stays low (task runs on EGS, not UAV)
        else:
            # Process locally
            task.processed_at = "local"
            uav.compute_load = min(0.95, projected_load)
            uav.local_processed += 1
            task.offload_latency = 0.0

        return task.processed_at

    def release_compute(self, uav: UAV, task: Task):
        """Call when UAV completes a locally-processed task to free compute."""
        if task.processed_at == "local":
            uav.compute_load = max(0.05, uav.compute_load - task.compute_demand)


# ── 2. Edge Ground Station ────────────────────────────────────────────────────

@dataclass
class CorrectionStats:
    overload_corrections: int = 0
    coverage_corrections: int = 0
    total_extra_msgs: int = 0


class EdgeGroundStation:
    """
    High-capacity ground station that validates and corrects onboard assignments.

    Mirrors the EGS in the UAV Swarm paper:
      - Onboard agents (LocalConsensus) make initial assignments.
      - EGS receives the proposed plan, detects problems, and issues fixes.
      - Two correction types are modelled (hallucination mitigation analog):
          (a) Compute overload  — reassign task to a lower-load UAV.
          (b) Coverage gap      — dispatch nearest free UAV to orphan task.

    The EGS is *not* used as the primary allocator — it is a validator,
    exactly as in the paper where the EGS corrects TinyLLaMA outputs rather
    than replacing them.

    Communication cost:
      Each validation round requires every UAV to upload state to EGS (+N msgs)
      and the EGS to broadcast corrections (+n_corrections msgs).
    """

    def __init__(
        self,
        x: float = 50.0,
        y: float = 50.0,
        overload_threshold: float = 0.75,
        offloading_model: Optional[OffloadingModel] = None,
    ):
        self.x = x
        self.y = y
        self.overload_threshold = overload_threshold
        self.offloading_model = offloading_model or OffloadingModel(x, y)
        self.stats = CorrectionStats()

    def validate_and_correct(self, obs: dict) -> int:
        """
        Inspect current assignments. Detect and fix overload + coverage gaps.
        Returns number of extra messages used in this validation round.
        """
        uavs: list[UAV] = obs["uavs"]
        tasks: list[Task] = obs["tasks"]
        uav_map = {u.id: u for u in uavs}
        task_map = {t.id: t for t in tasks if not t.completed}

        # Every UAV uploads state to EGS
        extra_msgs = len(uavs)

        # ── (a) Fix compute overloads ──────────────────────────────────────
        for u in uavs:
            if u.target_task_id is None:
                continue
            t = task_map.get(u.target_task_id)
            if t is None:
                continue
            
            # If current load exceeds threshold (due to onboard assignment)
            if u.compute_load > self.overload_threshold:
                # Revert local assignment on original UAV
                if t.processed_at == "local":
                    u.compute_load = max(0.05, u.compute_load - t.compute_demand)
                    u.local_processed -= 1

                # Find a lower-load free UAV to hand off to
                candidates = [
                    v for v in uavs
                    if v.id != u.id
                    and v.target_task_id is None
                    and (v.compute_load + t.compute_demand) <= self.overload_threshold
                ]
                if candidates:
                    candidates.sort(key=lambda v: v.dist_to(t))
                    new_uav = candidates[0]
                    # Hand off
                    u.target_task_id = None
                    t.assigned_to = new_uav.id
                    new_uav.target_task_id = t.id
                    # New UAV processes the task (local or edge)
                    self.offloading_model.decide(new_uav, t)
                    self.stats.overload_corrections += 1
                    extra_msgs += 2
                else:
                    # No swap possible; force offload processing to EGS for this UAV
                    self.offloading_model.decide(u, t)
                    extra_msgs += 1

        # ── (b) Fix coverage gaps (orphan tasks) ──────────────────────────
        open_tasks = [
            t for t in tasks
            if not t.completed and t.assigned_to is None
        ]
        for t in open_tasks:
            free_uavs = [u for u in uavs if u.target_task_id is None]
            if not free_uavs:
                break
            # EGS dispatches globally nearest free UAV (full visibility)
            best = min(free_uavs, key=lambda u: u.dist_to(t))
            t.assigned_to = best.id
            best.target_task_id = t.id
            self.offloading_model.decide(best, t)
            self.stats.coverage_corrections += 1
            extra_msgs += 1   # EGS -> dispatched UAV

        self.stats.total_extra_msgs += extra_msgs
        return extra_msgs

    def release_on_completion(self, uav: UAV, task: Task):
        """Notify offloading model when a UAV finishes a task."""
        self.offloading_model.release_compute(uav, task)


# ── Combined: LocalConsensus + EGS validation ────────────────────────────────

class EGSAssistedConsensus:
    """
    Two-stage allocation mirroring the UAV Swarm paper architecture:

      Stage 1 — Onboard (lightweight):
        LocalConsensus runs on each UAV using only local neighborhood info.

      Stage 2 — EGS validation (high-capacity ground node):
        EGS receives proposed plan, detects overloads and coverage gaps,
        and issues targeted corrections.

    This is the proposed method that combines both paper ideas.
    """

    name = "EGS-Assisted Consensus (proposed)"

    def __init__(self, area_size: float = 100.0, overload_threshold: float = 0.75):
        from consensus import LocalConsensus
        self.onboard = LocalConsensus()
        self.egs = EdgeGroundStation(
            x=area_size / 2,
            y=area_size / 2,
            overload_threshold=overload_threshold,
        )

    def assign(self, obs: dict) -> int:
        # Stage 1: onboard lightweight consensus
        msgs = self.onboard.assign(obs)
        # Stage 2: EGS validation and correction
        msgs += self.egs.validate_and_correct(obs)
        return msgs

    def on_task_complete(self, uav: UAV, task: Task):
        self.egs.release_on_completion(uav, task)

    @property
    def correction_stats(self) -> CorrectionStats:
        return self.egs.stats