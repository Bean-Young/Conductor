# Conductor RL Scheduler

This repository contains the reinforcement-learning scheduling component of
**Conductor: Dynamically Orchestrating Pipeline Parallelism with
Multi-Granularity Control**.

The release is intentionally limited to the paper's RL component:

- `conductor_rl/ppo_scheduler.py` implements heuristic-bootstrapped PPO.
- `conductor_rl/pipeline_simulator.py` provides the event-driven environment.
- `conductor_rl/zbpp_scheduler.py` provides the ZBPP expert used for behavioral
  cloning and evaluation.
- `conductor_rl/train.py` is a reproducible command-line entry point.

Migration code, Megatron-LM integration code, generated figures, experiment
outputs, virtual environments, caches, and legacy RL prototypes are not part of
this RL-only release.

## Paper-aligned configuration

The defaults exposed by `PPOScheduler` and the command-line entry point match
the revised manuscript:

| Setting | Value |
| --- | ---: |
| Devices | 4 |
| Micro-batches | 8 |
| Micro-batch size | 32 |
| Maximum memory budget | 4.0 |
| Runtime-noise settings | 0%, 10%, 20%, 30% |
| PPO training budget | 500 episodes |
| Learning rate | 3e-4 |
| PPO clipping parameter | 0.2 |
| Discount factor | 0.99 |
| PPO update epochs | 10 |
| Batch size | 64 |
| Optimizer | Adam |
| Value-loss coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Behavioral-cloning rollouts | 64 |
| Behavioral-cloning epochs | 10 |
| Hidden dimension | 128 |

The implementation uses discounted returns directly and does not expose a
separate GAE parameter, consistent with the manuscript.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

The manuscript evaluates noise levels independently. For example:

```bash
python -m conductor_rl.train --noise 0.1 --episodes 500 --seed 0
```

For a quick smoke test:

```bash
python -m conductor_rl.train --noise 0.1 --episodes 1 --seed 0
```

## Scope

This code evaluates the global RL scheduler in an offline, event-driven
simulator. It does not constitute a fully integrated RL-plus-migration
Megatron-LM runtime and should not be interpreted as hardware scalability
evidence.
