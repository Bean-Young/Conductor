import torch.nn as nn

from conductor_rl.pipeline_simulator import PipelineSimulator
from conductor_rl.ppo_scheduler import ActorCritic, PPOScheduler


def test_paper_aligned_defaults():
    simulator = PipelineSimulator(
        num_devices=4,
        num_micro_batches=8,
        micro_batch_size=32,
        max_memory=4.0,
        noise_level=0.1,
    )
    scheduler = PPOScheduler(simulator)

    assert scheduler.lr == 3e-4
    assert scheduler.gamma == 0.99
    assert scheduler.epsilon == 0.2
    assert scheduler.epochs == 10
    assert scheduler.batch_size == 64
    assert scheduler.bc_rollouts == 64
    assert scheduler.bc_epochs == 10


def test_actor_critic_uses_paper_hidden_dimension():
    model = ActorCritic(state_dim=16, action_dim=8)
    shared_linear = [m for m in model.shared_net if isinstance(m, nn.Linear)]
    actor_linear = [m for m in model.actor if isinstance(m, nn.Linear)]
    critic_linear = [m for m in model.critic if isinstance(m, nn.Linear)]

    assert [m.out_features for m in shared_linear] == [128, 128]
    assert actor_linear[0].out_features == 128
    assert critic_linear[0].out_features == 128
