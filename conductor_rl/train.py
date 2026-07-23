"""Command-line entry point for the paper-aligned Conductor RL scheduler."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from .pipeline_simulator import PipelineSimulator
from .ppo_scheduler import PPOScheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--micro-batches", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--max-memory", type=float, default=4.0)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    simulator = PipelineSimulator(
        num_devices=args.devices,
        num_micro_batches=args.micro_batches,
        micro_batch_size=args.micro_batch_size,
        max_memory=args.max_memory,
        noise_level=args.noise,
    )
    scheduler = PPOScheduler(simulator)
    scheduler.train(num_episodes=args.episodes)

    payload = {
        "paper_configuration": {
            "devices": args.devices,
            "micro_batches": args.micro_batches,
            "micro_batch_size": args.micro_batch_size,
            "max_memory": args.max_memory,
            "noise": args.noise,
            "episodes": args.episodes,
            "seed": args.seed,
        },
        "ppo": {
            "learning_rate": scheduler.lr,
            "clip": scheduler.epsilon,
            "discount_factor": scheduler.gamma,
            "update_epochs": scheduler.epochs,
            "batch_size": scheduler.batch_size,
            "behavioral_cloning_rollouts": scheduler.bc_rollouts,
            "behavioral_cloning_epochs": scheduler.bc_epochs,
            "value_loss_coefficient": 0.5,
            "entropy_coefficient": 0.01,
        },
        "training": scheduler.get_train_metrics(),
        "evaluation": scheduler.evaluate_vs_zbpp(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
