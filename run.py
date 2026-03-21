#!/usr/bin/env python3
"""CLI runner for hierarchy simulation — single runs and parameter sweeps."""

import argparse
import os
import sys

import mesa
import numpy as np
import pandas as pd

from model import HierarchyModel

# ------------------------------------------------------------------ defaults
DEFAULTS = dict(
    width=40,
    height=40,
    n_agents=200,
    mu=11.0,
    sigma=3.0,
    escape_cost=0.5,
    granary_capacity=50.0,
    gatekeeper_bias=0.6,
    myth_decay_rate=0.02,
    stress_threshold=0.3,
    daily_need=10.0,
    carry_capacity=15.0,
    steps=2000,
    seed=42,
    out_dir="output",
)


def run_single(args):
    """Run a single simulation and save model-level metrics to CSV."""
    os.makedirs(args.out_dir, exist_ok=True)

    model = HierarchyModel(
        width=args.width,
        height=args.height,
        n_agents=args.n_agents,
        mu=args.mu,
        sigma=args.sigma,
        escape_cost=args.escape_cost,
        granary_capacity=args.granary_capacity,
        gatekeeper_bias=args.gatekeeper_bias,
        myth_decay_rate=args.myth_decay_rate,
        stress_threshold=args.stress_threshold,
        daily_need=args.daily_need,
        carry_capacity=args.carry_capacity,
        seed=args.seed,
    )

    for _ in range(args.steps):
        model.step()
        if len(model._living_agents()) == 0:
            print("Population extinct at step", model._step_count)
            break

    df = model.datacollector.get_model_vars_dataframe()
    csv_path = os.path.join(args.out_dir, "model_metrics.csv")
    df.to_csv(csv_path)
    print(f"Model metrics saved to {csv_path}  ({len(df)} rows)")

    agent_df = model.datacollector.get_agent_vars_dataframe()
    agent_csv = os.path.join(args.out_dir, "agent_metrics.csv")
    agent_df.to_csv(agent_csv)
    print(f"Agent metrics saved to {agent_csv}  ({len(agent_df)} rows)")
    return df


def run_sweep(args):
    """Run a parameter sweep using mesa.batch_run."""
    os.makedirs(args.out_dir, exist_ok=True)

    params = {
        "width": args.width,
        "height": args.height,
        "n_agents": args.n_agents,
        "mu": args.mu,
        "sigma": args.sigma,
        "escape_cost": [0.1, 0.3, 0.5, 0.7, 0.9],
        "granary_capacity": [0.0, 25.0, 50.0, 100.0],
        "gatekeeper_bias": args.gatekeeper_bias,
        "myth_decay_rate": args.myth_decay_rate,
        "stress_threshold": args.stress_threshold,
        "daily_need": args.daily_need,
        "carry_capacity": args.carry_capacity,
        "seed": args.seed,
    }

    print("Starting parameter sweep …")
    results = mesa.batch_run(
        HierarchyModel,
        parameters=params,
        iterations=1,
        max_steps=args.steps,
        data_collection_period=50,
        display_progress=True,
    )

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Sweep results saved to {csv_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------- scenarios

def run_baseline(args):
    """Baseline: low escape cost, no granary → expect flat society."""
    args.escape_cost = 0.1
    args.granary_capacity = 0.0
    args.steps = min(args.steps, 2000)
    print("=== BASELINE: escape_cost=0.1, granary=0 ===")
    return run_single(args)


def run_stress(args):
    """Stress test: spike variance for 100 days mid-run."""
    os.makedirs(args.out_dir, exist_ok=True)
    model = HierarchyModel(
        width=args.width, height=args.height, n_agents=args.n_agents,
        mu=args.mu, sigma=args.sigma, escape_cost=args.escape_cost,
        granary_capacity=args.granary_capacity,
        gatekeeper_bias=args.gatekeeper_bias,
        myth_decay_rate=args.myth_decay_rate,
        stress_threshold=args.stress_threshold,
        daily_need=args.daily_need, carry_capacity=args.carry_capacity,
        seed=args.seed,
    )
    spike_start = args.steps // 4
    spike_end = spike_start + 100

    for t in range(args.steps):
        # Temporarily raise sigma during the spike window
        if t == spike_start:
            print(f"  [tick {t}] Spiking sigma to {args.sigma * 3:.1f}")
            for p in model.patches.values():
                p.sigma = args.sigma * 3
        elif t == spike_end:
            print(f"  [tick {t}] Restoring sigma to {args.sigma:.1f}")
            for p in model.patches.values():
                p.sigma = args.sigma
        model.step()
        if len(model._living_agents()) == 0:
            print("Population extinct at step", model._step_count)
            break

    df = model.datacollector.get_model_vars_dataframe()
    csv_path = os.path.join(args.out_dir, "stress_metrics.csv")
    df.to_csv(csv_path)
    print(f"Stress test metrics saved to {csv_path}")
    return df


def run_hardening(args):
    """Hardening test: high escape cost + large granary → ossified hierarchy."""
    args.escape_cost = 0.9
    args.granary_capacity = 200.0
    args.steps = min(args.steps, 3000)
    print("=== HARDENING: escape_cost=0.9, granary=200 ===")
    return run_single(args)


# ------------------------------------------------------------------- CLI

def build_parser():
    p = argparse.ArgumentParser(
        description="Hierarchy simulation via recursive othering (Mesa)"
    )
    sub = p.add_subparsers(dest="command")

    # -- shared arguments via parent parser --
    parent = argparse.ArgumentParser(add_help=False)
    for k, v in DEFAULTS.items():
        flag = f"--{k}"
        parent.add_argument(flag, type=type(v), default=v)

    sub.add_parser("single", parents=[parent], help="Single simulation run")
    sub.add_parser("sweep", parents=[parent], help="Parameter sweep")
    sub.add_parser("baseline", parents=[parent], help="Baseline scenario (flat)")
    sub.add_parser("stress", parents=[parent], help="Stress-test scenario")
    sub.add_parser("hardening", parents=[parent], help="Hardening scenario")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "single": run_single,
        "sweep": run_sweep,
        "baseline": run_baseline,
        "stress": run_stress,
        "hardening": run_hardening,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
