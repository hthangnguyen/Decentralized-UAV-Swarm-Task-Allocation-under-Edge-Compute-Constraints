"""
env.py — UAV swarm environment.

Models a 2-D airspace with UAVs and tasks.  Each UAV has a position,
velocity, compute-load, battery, and communication range.  Tasks appear
at random positions with a priority level.  The environment steps forward
in discrete ticks and is intentionally framework-free (pure NumPy).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: int
    x: float
    y: float
    priority: float          # 0.0 (low) … 1.0 (high)
    assigned_to: Optional[int] = None
    completed: bool = False
    spawn_tick: int = 0
    complete_tick: Optional[int] = None

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class UAV:
    id: int
    x: float
    y: float
    comm_range: float
    speed: float = 2.0
    compute_load: float = 0.3   # fraction 0-1 of on-board compute used
    battery: float = 1.0        # fraction 0-1

    vx: float = 0.0
    vy: float = 0.0
    target_task_id: Optional[int] = None
    neighbors: list = field(default_factory=list)   # ids of in-range UAVs

    # per-episode counters
    tasks_completed: int = 0
    distance_flown: float = 0.0
    consensus_msgs_sent: int = 0

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def dist_to(self, other) -> float:
        if isinstance(other, Task):
            return float(np.linalg.norm(self.pos - other.pos))
        return float(np.linalg.norm(self.pos - other.pos))


# ── Environment ───────────────────────────────────────────────────────────────

class SwarmEnv:
    """
    Parameters
    ----------
    n_uavs       : number of UAVs
    area_size    : side length of the square airspace (metres / arbitrary units)
    comm_range   : radio communication radius for each UAV
    uav_speed    : movement units per tick
    task_rate    : expected new tasks per tick (Poisson λ)
    arrive_radius: distance at which a UAV is considered to have reached a task
    seed         : RNG seed for reproducibility
    """

    def __init__(
        self,
        n_uavs: int = 5,
        area_size: float = 100.0,
        comm_range: float = 35.0,
        uav_speed: float = 2.0,
        task_rate: float = 0.08,
        arrive_radius: float = 3.0,
        seed: int = 42,
    ):
        self.n_uavs = n_uavs
        self.area = area_size
        self.comm_range = comm_range
        self.uav_speed = uav_speed
        self.task_rate = task_rate
        self.arrive_radius = arrive_radius
        self.rng = np.random.default_rng(seed)

        self.uavs: list[UAV] = []
        self.tasks: list[Task] = []
        self.tick: int = 0
        self._next_task_id: int = 0

        self._init_uavs()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _init_uavs(self):
        self.uavs = []
        for i in range(self.n_uavs):
            self.uavs.append(UAV(
                id=i,
                x=float(self.rng.uniform(5, self.area - 5)),
                y=float(self.rng.uniform(5, self.area - 5)),
                comm_range=self.comm_range,
                speed=self.uav_speed,
                compute_load=float(self.rng.uniform(0.1, 0.5)),
                battery=float(self.rng.uniform(0.7, 1.0)),
            ))

    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.tasks = []
        self.tick = 0
        self._next_task_id = 0
        self._init_uavs()
        # seed a few tasks at start
        for _ in range(max(2, self.n_uavs // 2)):
            self._spawn_task()
        self._update_neighbors()
        return self._obs()

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self) -> dict:
        """Advance one tick: spawn tasks, move UAVs, check arrivals."""
        self.tick += 1

        # maybe spawn new tasks (Poisson)
        n_new = self.rng.poisson(self.task_rate)
        for _ in range(n_new):
            self._spawn_task()

        # move UAVs toward their targets
        self._move_uavs()

        # check arrivals & free completed tasks
        self._check_arrivals()

        # drift compute load a bit
        for u in self.uavs:
            u.compute_load = float(np.clip(
                u.compute_load + self.rng.normal(0, 0.02), 0.05, 0.95))
            u.battery = float(np.clip(u.battery - 0.0005, 0.0, 1.0))

        self._update_neighbors()
        return self._obs()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _spawn_task(self):
        t = Task(
            id=self._next_task_id,
            x=float(self.rng.uniform(5, self.area - 5)),
            y=float(self.rng.uniform(5, self.area - 5)),
            priority=float(self.rng.beta(2, 2)),
            spawn_tick=self.tick,
        )
        self.tasks.append(t)
        self._next_task_id += 1

    def _update_neighbors(self):
        for u in self.uavs:
            u.neighbors = [
                v.id for v in self.uavs
                if v.id != u.id and u.dist_to(v) <= self.comm_range
            ]

    def _move_uavs(self):
        task_map = {t.id: t for t in self.tasks}
        for u in self.uavs:
            if u.target_task_id is not None and u.target_task_id in task_map:
                t = task_map[u.target_task_id]
                delta = t.pos - u.pos
                dist = float(np.linalg.norm(delta))
                if dist > 1e-6:
                    step = min(u.speed, dist)
                    move = delta / dist * step
                    u.x += float(move[0])
                    u.y += float(move[1])
                    u.distance_flown += step
                # clamp to area
                u.x = float(np.clip(u.x, 0, self.area))
                u.y = float(np.clip(u.y, 0, self.area))
            else:
                # random drift when idle
                u.vx += float(self.rng.normal(0, 0.3))
                u.vy += float(self.rng.normal(0, 0.3))
                spd = np.hypot(u.vx, u.vy)
                if spd > u.speed * 0.3:
                    u.vx *= u.speed * 0.3 / spd
                    u.vy *= u.speed * 0.3 / spd
                u.x = float(np.clip(u.x + u.vx, 0, self.area))
                u.y = float(np.clip(u.y + u.vy, 0, self.area))

    def _check_arrivals(self):
        task_map = {t.id: t for t in self.tasks}
        for u in self.uavs:
            if u.target_task_id is None:
                continue
            t = task_map.get(u.target_task_id)
            if t is None:
                u.target_task_id = None
                continue
            if u.dist_to(t) <= self.arrive_radius:
                t.completed = True
                t.complete_tick = self.tick
                u.tasks_completed += 1
                u.target_task_id = None

    # ── Observation ───────────────────────────────────────────────────────────

    def _obs(self) -> dict:
        return {
            "tick": self.tick,
            "uavs": self.uavs,
            "tasks": self.tasks,
            "open_tasks": [t for t in self.tasks if not t.completed and t.assigned_to is None],
            "active_tasks": [t for t in self.tasks if not t.completed and t.assigned_to is not None],
            "done_tasks": [t for t in self.tasks if t.completed],
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    def free_uavs(self) -> list[UAV]:
        return [u for u in self.uavs if u.target_task_id is None]

    def busy_uavs(self) -> list[UAV]:
        return [u for u in self.uavs if u.target_task_id is not None]
