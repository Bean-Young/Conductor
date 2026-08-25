# Conductor

This project was created by [***AHU-Team***](https://github.com/ahuteam) for
the paper "**Conductor: Dynamically orchestrating pipeline parallelism with
multi-granularity control**" ([Paper
Link](https://doi.org/10.1016/j.future.2026.108762)).

### [**Project Page**](https://bean-young.github.io/conductor/)

### *Future Generation Computer Systems, 2027*

## *Abstract*

Pipeline parallelism is a fundamental pillar of large-scale model training, yet
its efficiency is frequently constrained by straggler-induced pipeline bubbles.
This issue is exacerbated by static scheduling approaches, including
handcrafted heuristics and Integer Linear Programming, which are inherently
brittle when facing real-world execution-time variance. In this work, we
introduce Conductor, a dynamic two-tiered scheduling framework designed to
virtually eliminate straggler-induced bubbles under realistic stochastic
conditions. The key technical insight is to decouple global, long-horizon
scheduling from local, instantaneous load balancing. At a coarse grain, a
reinforcement learning agent leverages millisecond-scale inference to generate
robust global schedules and adapts to runtime dynamics in scenarios where
traditional static solvers are computationally intractable. At a fine grain,
we introduce a dynamic computation migration mechanism that resolves residual
micro-bubbles by offloading sub-computations, such as attention heads, from
transiently slower workers to faster ones within a single timestep. Evaluated
on large-scale language-model training configurations, our framework
outperforms state-of-the-art static scheduling baselines by 5% to 14% in
throughput and demonstrates superior resilience to injected system noise and
execution variance.

## *Implementation*

The code follows the paper's RL-based global scheduler and evaluates each
scheduling decision in an event-driven simulator that models runtime variation
and straggler behavior.

- `conductor_rl/ppo_scheduler.py` implements the PPO scheduler and its
  heuristic-bootstrapped training procedure.
- `conductor_rl/pipeline_simulator.py` models the pipeline execution process
  and stochastic runtime variation.
- `conductor_rl/zbpp_scheduler.py` implements the ZBPP baseline used for
  behavioral cloning and evaluation.
- `conductor_rl/train.py` trains the scheduler and reports its comparison with
  the ZBPP baseline.

## *Set Up*

Python 3.10 or later is recommended. The paper-aligned implementation requires
PyTorch 2.0 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## *Run*

The manuscript evaluates runtime-noise levels independently. For example, run
the paper-aligned configuration at 10% noise with:

```bash
python -m conductor_rl.train --noise 0.1 --episodes 500 --seed 0
```

For a quick smoke test:

```bash
python -m conductor_rl.train --noise 0.1 --episodes 1 --seed 0
```

The command prints a JSON record containing the configuration, PPO training
metrics, and evaluation against the ZBPP expert.

## *Paper-aligned Configuration*

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
separate GAE parameter, consistent with the paper.

## *References*

- [**Proximal Policy Optimization (PPO)**](https://github.com/openai/baselines/tree/master/baselines/ppo2)
- [**Zero Bubble Pipeline Parallelism (ZBPP)**](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)
- [**Megatron-LM (1F1B)**](https://github.com/NVIDIA/Megatron-LM)

## *Citation*

```bibtex
@article{dong2027conductor,
  title={Conductor: Dynamically orchestrating pipeline parallelism with multi-granularity control},
  author={Dong, Xingbo and Liu, Ziyuan and Yang, Yuezhe and Lai, Yen-Lung and Jin, Zhe},
  journal={Future Generation Computer Systems},
  volume={186},
  pages={108762},
  year={2027},
  doi={10.1016/j.future.2026.108762}
}
```
