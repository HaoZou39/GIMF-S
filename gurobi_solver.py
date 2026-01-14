# -*- coding: utf-8 -*-
"""
gurobi_solver.py

【一句话用途】
    用 Gurobi 求解 TSP（Traveling Salesman Problem），支持 MTZ 与 Lazy DFJ(subtour cuts) 两种子回路消除方案。

【功能概览 / Features】
    1) solve_tsp_mtz():   MTZ 约束（一次性建模，无回调）
    2) solve_tsp_lazy():  DFJ 子回路 cut（LazyConstraints + callback 动态添加）
    3) solve_tsp_from_matrix():  输入 NxN 距离矩阵直接求解的快捷入口

【输入约定 / Input Contract】
    - 距离矩阵 D 为 shape=(n,n) 的二维数组
    - D[i, j] 表示从 i 到 j 的距离/成本
    - 允许非对称 (asymmetric)；对角线推荐设置为0

【输出约定 / Output Contract】
    - 返回 dict：
        {
            "tour": [0, ..., 0],          # 从 0 出发并回到 0（闭环）
            "objective": float,           # 目标值（总成本/距离）
            "runtime_sec": float,
            "status": int,
            "model": gp.Model,            # 可选：用于调试/查看变量
        }
    - tour 的长度通常为 n+1（首尾同为 0）

【模型说明 / Model】
    - 变量：x[i,j] ∈ {0,1}，表示是否选用有向边 i->j
    - 约束：每个点出度=1、入度=1（形成若干个有向环）
    - 子回路消除：
        a) MTZ：引入顺序变量 u[i]，用线性约束破坏小环
        b) Lazy DFJ：在整数可行解出现时，检测子回路并添加 DFJ cut

【依赖 / Dependencies】
    - gurobipy
    - numpy

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np

import gurobipy as gp
from gurobipy import GRB

# -----------------------------
# Type aliases (for readability)
# -----------------------------
Edge = Tuple[int, int]
DistDict = Dict[Edge, float]

# -----------------------------
# Config dataclass
# -----------------------------
@dataclass(frozen=True)
class GurobiConfig:
    time_limit: Optional[float] = None   # 秒
    mip_gap: Optional[float] = None
    threads: Optional[int] = None
    output_flag: int = 1                 # 0=静默；1=输出日志

# ============================================================
# Helper functions (pure utilities; not tied to a specific model)
# ============================================================
def dist_dict_from_matrix(D: np.ndarray) -> DistDict:
    """
    Convert NxN distance matrix into dict dist[(i,j)]. when i==j, skip

    Raises:
        ValueError if D is not square.
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be square matrix")
    n = D.shape[0]
    dist: DistDict = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist[(i, j)] = float(D[i, j])
    return dist


def _set_common_params(model: gp.Model, cfg: GurobiConfig) -> None:
    """
    Apply common Gurobi parameters.

    Note:
        These parameters affect solve behavior (time limit, gap, threads, logging),
        but do NOT modify the mathematical model.
    """
    model.Params.OutputFlag = cfg.output_flag
    if cfg.time_limit is not None:
        model.Params.TimeLimit = cfg.time_limit
    if cfg.mip_gap is not None:
        model.Params.MIPGap = cfg.mip_gap
    if cfg.threads is not None:
        model.Params.Threads = cfg.threads


def _recover_tour_from_x(n: int, dist: DistDict, x: gp.tupledict) -> List[int]:
    """
    Recover a single tour from binary decision variables x.

    Assumption:
        Out-degree constraints enforce exactly one successor for each node in an incumbent solution.

    Returns:
        tour like [0, ..., 0]. If something is wrong, it may return a partial path.
    """
    succ: Dict[int, int] = {}
    for (i, j) in dist.keys():
        if x[i, j].X > 0.5:
            succ[i] = j

    tour = [0]
    cur = 0
    for _ in range(n + 1):  # 防止异常死循环：最多 n+1 步
        nxt = succ.get(cur)
        if nxt is None:
            break
        tour.append(nxt)
        cur = nxt
        if cur == 0:
            break
    return tour


# =========================
# 1) MTZ formulation
# =========================
def solve_tsp_mtz(n: int, dist: DistDict, cfg: Optional[GurobiConfig] = None):
    """
    Solve directed TSP using MTZ subtour elimination.

    Model:
        - x[i,j] binary (i != j)
        - outdegree=1 and indegree=1
        - u[i] continuous ordering variables (MTZ)

    Pros:
        - No callback required
    Cons:
        - MTZ can be weaker for larger n

    Returns:
        dict with tour/objective/runtime/status/model
    """
    cfg = cfg or GurobiConfig()
    model = gp.Model("tsp_mtz")
    _set_common_params(model, cfg)

    # 1) Variables
    x = model.addVars(dist.keys(), vtype=GRB.BINARY, name="x")

    # MTZ ordering variable:
    # u[i] ~ visit order of node i (breaks subtours)
    u = model.addVars(range(n), vtype=GRB.CONTINUOUS, name="u")
    # Fix u[0]=0 to reduce symmetry (otherwise u can shift by a constant)
    model.addConstr(u[0] == 0, name="u_0_fix")
    # Bound u for non-depot nodes
    for i in range(1, n):
        model.addConstr(u[i] >= 1, name=f"u_lb_{i}")
        model.addConstr(u[i] <= n - 1, name=f"u_ub_{i}")

    # 2) Objective
    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for (i, j) in dist.keys()),
        GRB.MINIMIZE
    )

    # 3) Degree constraints
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(n) if j != i) == 1,
            name=f"outdeg_{i}"
        )

    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n) if i != j) == 1,
            name=f"indeg_{j}"
        )

    # 4) MTZ subtour elimination (apply only to nodes 1...n-1)
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            model.addConstr(
                u[i] - u[j] + n * x[i, j] <= n - 1,
                name=f"mtz_{i}_{j}"
            )
    # 5) Optimize
    model.optimize()
    # 6) Check status / recover solution
    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi status not optimal: {model.status}")
    if model.SolCount == 0:
        raise RuntimeError("No feasible solution found.")

    tour = _recover_tour_from_x(n, dist, x)

    return {
        "tour": tour,
        "objective": float(model.objVal) if model.SolCount > 0 else None,
        "runtime_sec": float(model.Runtime),
        "status": int(model.status),
        "model": model,
    }


# =========================
# 2) Lazy DFJ callback formulation
# =========================
def _find_cycles_from_succ(succ: Dict[int, int], n: int) -> List[List[int]]:
    """
    Given succ mapping (each node has exactly one successor), return disjoint directed cycles.

    Output:
        cycles: list of cycles, each cycle is a list of nodes in that cycle.
    """
    unvisited = set(range(n))
    cycles: List[List[int]] = []
    while unvisited:
        start = unvisited.pop()
        cycle = [start]
        cur = start

        # 用 dict: node -> position 来检测回到某个已见点
        pos = {start: 0}

        while True:
            nxt = succ.get(cur)
            if nxt is None:
                # Defensive: should not happen if outdegree=1
                break

            if nxt in pos:
                # Found a cycle: from pos[nxt] to end
                cycle = cycle[pos[nxt]:]
                break

            pos[nxt] = len(cycle)
            cycle.append(nxt)
            unvisited.discard(nxt)
            cur = nxt

        cycles.append(cycle)
    return cycles


def _subtour_elim_callback_directed(model: gp.Model, where: int):
    """
    Lazy constraint callback (directed DFJ subtour cuts).

    Trigger:
        - Called at GRB.Callback.MIPSOL when an integer feasible solution is found.

    Cut:
        For a subtour S (|S| < n):
            sum_{i in S} sum_{j in S, j!=i} x[i,j] <= |S| - 1

    Meaning:
        Prevent S from forming an independent directed cycle.
    """
    if where == GRB.Callback.MIPSOL:
        x = model._vars
        n = model._n

        vals = model.cbGetSolution(x)

        # Build successor mapping from the incumbent integer solution
        succ: Dict[int, int] = {}
        for (i, j) in x.keys():
            if vals[i, j] > 0.5:
                succ[i] = j
        # Find all cycles and cut those not spanning all nodes
        cycles = _find_cycles_from_succ(succ, n)
        for cycle in cycles:
            if len(cycle) < n:
                expr = gp.quicksum(
                    x[i, j]
                    for i in cycle
                    for j in cycle
                    if i != j and (i, j) in x
                )
                model.cbLazy(expr <= len(cycle) - 1)


def solve_tsp_lazy(n: int, dist: DistDict, cfg: Optional[GurobiConfig] = None):
    """
    Solve directed TSP using LazyConstraints + DFJ subtour elimination cuts.

    Model:
        - x[i,j] binary (i != j)
        - outdegree=1 and indegree=1
        - subtours removed via callback cuts (DFJ)

    Pros:
        - Stronger than MTZ for many cases
    Cons:
        - Requires callback; must set LazyConstraints=1

    Returns:
        dict with tour/objective/runtime/status/model
    """
    cfg = cfg or GurobiConfig()
    model = gp.Model("tsp_lazy_dfj_directed")
    _set_common_params(model, cfg)

    # 1) Variables]
    x = model.addVars(dist.keys(), vtype=GRB.BINARY, name="x")

    # 2) Objective
    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for (i, j) in dist.keys()),
        GRB.MINIMIZE
    )

    # 3) Degree constraints
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(n) if j != i) == 1,
            name=f"outdeg_{i}"
        )


    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n) if i != j) == 1,
            name=f"indeg_{j}"
        )

    # 4) Enable lazy constraints (required for cbLazy)
    model.Params.LazyConstraints = 1

    # Attach data for callback
    model._vars = x
    model._n = n

    # 5) Optimize with callback
    model.optimize(_subtour_elim_callback_directed)

    # 6) Check status / recover solution
    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi status not optimal: {model.status}")
    if model.SolCount == 0:
        raise RuntimeError("No feasible solution found (no incumbent).")

    tour = _recover_tour_from_x(n, dist, x)

    return {
        "tour": tour,
        "objective": float(model.objVal) if model.SolCount > 0 else None,
        "runtime_sec": float(model.Runtime),
        "status": int(model.status),
        "model": model,
    }

# =========================
# 3) Public entry
# =========================
def solve_tsp_from_matrix(
    D: np.ndarray,
    cfg: Optional[GurobiConfig] = None,
    method: str = "lazy",
):
    """
    Public entry: solve TSP from an NxN distance matrix.

    Parameters
    ----------
    D : np.ndarray
        NxN distance matrix.
    cfg : Optional[GurobiConfig]
        Solver parameters.
    method : str
        "lazy" (DFJ callback) or "mtz".

    Returns
    -------
    dict
        Solution dict (see module output contract).
    """
    dist = dist_dict_from_matrix(D)
    n = D.shape[0]
    method = method.lower().strip()
    if method == "lazy":
        return solve_tsp_lazy(n, dist, cfg=cfg)
    if method == "mtz":
        return solve_tsp_mtz(n, dist, cfg=cfg)
    raise ValueError("method must be 'lazy' or 'mtz'")
