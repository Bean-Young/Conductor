#!/usr/bin/env python3
"""
PPO-based Pipeline Scheduler
基于PPO强化学习的流水线调度算法

核心思想：
1. 在ZBPP调度的基础上，在关键决策点使用PPO进行事件选择
2. 只训练在多个可行事件中选择的能力，而不是整个调度序列
3. 确保所有约束始终被满足
4. 学习比ZBPP启发式更优的调度策略
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
from typing import List, Dict, Any, Optional, Tuple
from .pipeline_simulator import PipelineSimulator, OperationType
from .zbpp_scheduler import ZBPPScheduler

class PipelineSchedulingEnv(gym.Env):
    """
    流水线调度环境 - 包装ZBPP调度器为RL环境
    """
    
    def __init__(self, num_devices: int = 4, num_micro_batches: int = 8, 
                 micro_batch_size: int = 32, max_memory: float = 4.0,
                 noise_level: float = 0.1):
        super().__init__()
        
        # 初始化模拟器和ZBPP调度器
        self.simulator = PipelineSimulator(num_devices, num_micro_batches, 
                                          micro_batch_size, max_memory, noise_level)
        self.zbpp_scheduler = ZBPPScheduler(self.simulator)
        
        # 状态空间设计
        # 状态包括：设备状态 + 微批次进度 + 全局状态 + 候选事件特征
        self.candidate_feature_size = 6  # type one-hot(3) + device + micro-batch + duration
        self.state_size = (
            num_devices * 3 +  # 设备状态 (可用时间, 内存使用, 工作负载)
            num_micro_batches * 3 +  # 微批次进度 (每个微批次的F/B/W完成状态)
            4 +  # 全局状态 (当前时间, 总进度, 气泡时间, 内存使用率)
            (num_devices * num_micro_batches * 3) * self.candidate_feature_size
        )
        
        # 动作空间：在当前可行事件列表中选择一个。
        # 这里使用总事件数上界，真正采样时会按当前可行事件数做mask。
        self.total_event_count = num_devices * num_micro_batches * 3
        self.action_space = spaces.Discrete(self.total_event_count)
        
        # 状态空间
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32
        )
        
        # 重置环境
        self.reset()
    
    def reset(self):
        """重置环境"""
        # 初始化调度状态
        self.scheduled_events = []
        self.current_time = 0.0
        self.device_available_time = {i: 0.0 for i in range(self.simulator.num_devices)}

        # 先生成一条ZBPP基线轨迹，再从同一组事件模板中创建RL环境事件，
        # 保证基线与RL比较时拥有完全相同的事件持续时间。
        self.zbpp_sequence = self._get_zbpp_schedule_sequence()
        self.event_list = [self._clone_event_template(event) for event in self.zbpp_sequence]
        self.current_feasible_events: List[Dict[str, Any]] = []
        self._prepare_next_decision()

        return self._get_state()
    
    def _get_zbpp_schedule_sequence(self) -> List[Dict[str, Any]]:
        """获取ZBPP的调度序列"""
        zbpp_events = self.zbpp_scheduler.schedule()
        return sorted(copy.deepcopy(zbpp_events), key=lambda x: x.get('start_time', 0))

    def _clone_event_template(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """复制一个未调度事件模板，避免污染基线轨迹。"""
        return {
            'type': event['type'],
            'micro_batch_id': event['micro_batch_id'],
            'device_id': event['device_id'],
            'start_time': None,
            'duration': event['duration'],
            'end_time': None,
        }
    
    def _identify_decision_points(self) -> List[Dict[str, Any]]:
        """识别需要PPO决策的关键点"""
        decision_points = []
        
        # 模拟ZBPP调度过程，识别有多个选择的点
        temp_scheduled = []
        temp_device_time = self.device_available_time.copy()
        temp_time = self.current_time
        
        zbpp_sequence = self.zbpp_sequence.copy()
        
        step_count = 0
        while zbpp_sequence and step_count < 1000:  # 防止无限循环
            step_count += 1
            
            # 获取当前可执行事件
            feasible_events = self._get_feasible_events(temp_scheduled, temp_time, temp_device_time)
            
            if len(feasible_events) > 1:
                # 多个选择，这是一个决策点
                decision_point = {
                    'time': temp_time,
                    'feasible_events': feasible_events,
                    'device_available_time': temp_device_time.copy(),
                    'scheduled_events': temp_scheduled.copy()
                }
                decision_points.append(decision_point)
            
            # 推进调度（使用ZBPP选择）
            if zbpp_sequence:
                next_event = zbpp_sequence.pop(0)
                self._schedule_event_simulation(next_event, temp_scheduled, temp_device_time, temp_time)
            else:
                break
        
        # 如果没有找到决策点，创建一些简单的决策点
        if not decision_points:
            # 创建一些基本的决策点
            for i in range(10):  # 创建10个决策点
                decision_point = {
                    'time': i * 10.0,
                    'feasible_events': self.event_list[i*5:(i+1)*5] if i*5 < len(self.event_list) else [],
                    'device_available_time': temp_device_time.copy(),
                    'scheduled_events': temp_scheduled.copy()
                }
                decision_points.append(decision_point)
        
        return decision_points
    
    def _get_feasible_events(self, scheduled_events: List[Dict[str, Any]], 
                           current_time: float, device_available_time: Dict[int, float]) -> List[Dict[str, Any]]:
        """获取可执行事件"""
        feasible_events = []
        
        # 创建已调度事件的标识符集合
        scheduled_keys = set()
        for event in scheduled_events:
            key = (event['type'], event['micro_batch_id'], event['device_id'])
            scheduled_keys.add(key)
        
        for event in self.event_list:
            # 检查是否已调度
            event_key = (event['type'], event['micro_batch_id'], event['device_id'])
            if event_key in scheduled_keys:
                continue
            
            # 检查依赖
            if not self._check_dependencies(event, scheduled_events, current_time):
                continue
            
            # 检查设备可用性
            device_id = event['device_id']
            if device_available_time[device_id] > current_time:
                continue
            
            # 检查内存
            if not self._check_memory_feasible(event, scheduled_events, device_id, current_time):
                continue
            
            feasible_events.append(event)
        
        return feasible_events
    
    def _is_event_finished(
        self,
        target_key: Tuple[OperationType, int, int],
        scheduled_events: List[Dict[str, Any]],
        current_time: float,
    ) -> bool:
        """检查依赖事件是否已经完成。"""
        for scheduled_event in scheduled_events:
            key = (
                scheduled_event['type'],
                scheduled_event['micro_batch_id'],
                scheduled_event['device_id'],
            )
            if key != target_key:
                continue
            end_time = scheduled_event.get('end_time')
            return end_time is not None and end_time <= current_time
        return False

    def _check_dependencies(
        self,
        event: Dict[str, Any],
        scheduled_events: List[Dict[str, Any]],
        current_time: float,
    ) -> bool:
        """检查依赖关系"""
        op_type = event['type']
        mb_id = event['micro_batch_id']
        device_id = event['device_id']

        if op_type == OperationType.FORWARD:
            # 依赖前一个设备
            if device_id > 0:
                prev_key = (OperationType.FORWARD, mb_id, device_id - 1)
                if not self._is_event_finished(prev_key, scheduled_events, current_time):
                    return False
            
            # 依赖前一个微批次
            if mb_id > 0:
                prev_mb_key = (OperationType.FORWARD, mb_id - 1, device_id)
                if not self._is_event_finished(prev_mb_key, scheduled_events, current_time):
                    return False
        
        elif op_type == OperationType.BACKWARD:
            # 依赖后一个设备
            if device_id < self.simulator.num_devices - 1:
                next_key = (OperationType.BACKWARD, mb_id, device_id + 1)
                if not self._is_event_finished(next_key, scheduled_events, current_time):
                    return False
            
            # 依赖最后一个设备的Forward
            last_f_key = (OperationType.FORWARD, mb_id, self.simulator.num_devices - 1)
            if not self._is_event_finished(last_f_key, scheduled_events, current_time):
                return False
        
        elif op_type == OperationType.WEIGHT_UPDATE:
            # 依赖同设备的Backward
            b_key = (OperationType.BACKWARD, mb_id, device_id)
            if not self._is_event_finished(b_key, scheduled_events, current_time):
                return False
        
        return True
    
    def _check_memory_feasible(self, event: Dict[str, Any], scheduled_events: List[Dict[str, Any]], 
                             device_id: int, current_time: float) -> bool:
        """检查内存可行性"""
        current_memory = self._get_current_memory_usage(device_id, scheduled_events, current_time)
        memory_change = self._get_memory_change(event)
        
        return current_memory + memory_change <= self.simulator.max_memory
    
    def _get_current_memory_usage(self, device_id: int, scheduled_events: List[Dict[str, Any]], 
                                current_time: float) -> float:
        """获取当前内存使用"""
        memory_usage = 0.0
        
        for event in scheduled_events:
            if event['device_id'] == device_id and event.get('start_time', 0) <= current_time:
                memory_usage += self._get_memory_change(event)
        
        return memory_usage
    
    def _get_memory_change(self, event: Dict[str, Any]) -> float:
        """获取内存变化"""
        if event['type'] == OperationType.FORWARD:
            return 1.0
        elif event['type'] == OperationType.BACKWARD:
            return -0.5
        elif event['type'] == OperationType.WEIGHT_UPDATE:
            return -0.5
        return 0.0
    
    def _schedule_event_simulation(self, event: Dict[str, Any], scheduled_events: List[Dict[str, Any]], 
                                 device_available_time: Dict[int, float], current_time: float):
        """模拟调度事件"""
        device_id = event['device_id']
        
        # 计算开始时间
        start_time = max(current_time, device_available_time[device_id])
        event['start_time'] = start_time
        event['end_time'] = start_time + event['duration']
        
        # 添加到调度列表
        scheduled_events.append(event)
        
        # 更新设备可用时间
        device_available_time[device_id] = event['end_time']

    def _has_pending_events(self) -> bool:
        """是否还有未调度事件。"""
        return len(self.scheduled_events) < len(self.event_list)

    def _prepare_next_decision(self) -> bool:
        """
        将时间推进到下一个真正需要决策的位置。

        返回:
            bool: True 表示已经终止；False 表示存在可行事件可供决策。
        """
        while True:
            if not self._has_pending_events():
                self.current_feasible_events = []
                return True

            feasible_events = self._get_feasible_events(
                self.scheduled_events, self.current_time, self.device_available_time
            )
            if feasible_events:
                # 使用稳定排序，保证相同状态下动作索引语义固定。
                type_priority = {
                    OperationType.BACKWARD: 0,
                    OperationType.FORWARD: 1,
                    OperationType.WEIGHT_UPDATE: 2,
                }
                self.current_feasible_events = sorted(
                    feasible_events,
                    key=lambda e: (
                        e['device_id'],
                        type_priority[e['type']],
                        e['micro_batch_id'],
                    ),
                )
                return False

            running_end_times = [
                event['end_time']
                for event in self.scheduled_events
                if event.get('end_time') is not None and event['end_time'] > self.current_time
            ]
            if not running_end_times:
                self.current_feasible_events = []
                return True

            self.current_time = min(running_end_times)

    def get_action_candidates(self) -> List[Dict[str, Any]]:
        """返回当前动作候选列表，并在必要时自动推进时间。"""
        self._prepare_next_decision()
        return self.current_feasible_events
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []
        
        # 设备状态 (3个维度 × 设备数)
        for device_id in range(self.simulator.num_devices):
            # 设备可用时间 (归一化)
            avail_time = self.device_available_time.get(device_id, 0.0)
            state.append(min(avail_time / 100.0, 1.0))  # 假设最大时间100s
            
            # 设备内存使用 (归一化)
            memory_usage = self._get_current_memory_usage(device_id, self.scheduled_events, self.current_time)
            state.append(memory_usage / self.simulator.max_memory)
            
            # 设备工作负载 (已调度事件数 / 总事件数)
            device_events = [e for e in self.scheduled_events if e['device_id'] == device_id]
            workload = len(device_events) / (self.simulator.num_micro_batches * 3)  # 每个设备最多 M×3 个事件
            state.append(workload)
        
        # 微批次进度 (3个操作 × 3个状态 × 微批次数)
        for mb_id in range(self.simulator.num_micro_batches):
            for op_type in [OperationType.FORWARD, OperationType.BACKWARD, OperationType.WEIGHT_UPDATE]:
                # 检查该操作是否完成
                completed = any(
                    e['micro_batch_id'] == mb_id and e['type'] == op_type 
                    for e in self.scheduled_events
                )
                state.append(1.0 if completed else 0.0)
        
        # 全局状态
        # 当前时间 (归一化)
        state.append(min(self.current_time / 100.0, 1.0))
        
        # 总进度
        total_progress = len(self.scheduled_events) / len(self.event_list)
        state.append(total_progress)
        
        # 气泡时间估计 (设备空闲时间比例)
        total_device_time = sum(self.device_available_time.values())
        ideal_time = max(self.device_available_time.values()) * self.simulator.num_devices
        bubble_time = max(0, ideal_time - total_device_time) / ideal_time if ideal_time > 0 else 0
        state.append(bubble_time)
        
        # 内存使用率
        avg_memory_usage = np.mean([
            self._get_current_memory_usage(i, self.scheduled_events, self.current_time)
            for i in range(self.simulator.num_devices)
        ])
        memory_utilization = avg_memory_usage / self.simulator.max_memory
        state.append(memory_utilization)

        # 候选动作特征。按当前固定排序后的可行事件列表编码，帮助策略理解每个动作槽位的语义。
        max_duration = max((event['duration'] for event in self.event_list), default=1.0)
        type_encoding = {
            OperationType.FORWARD: [1.0, 0.0, 0.0],
            OperationType.BACKWARD: [0.0, 1.0, 0.0],
            OperationType.WEIGHT_UPDATE: [0.0, 0.0, 1.0],
        }
        for index in range(self.total_event_count):
            if index < len(self.current_feasible_events):
                event = self.current_feasible_events[index]
                state.extend(type_encoding[event['type']])
                state.append(
                    event['device_id'] / max(self.simulator.num_devices - 1, 1)
                )
                state.append(
                    event['micro_batch_id'] / max(self.simulator.num_micro_batches - 1, 1)
                )
                state.append(min(event['duration'] / max(max_duration, 1e-8), 1.0))
            else:
                state.extend([0.0] * self.candidate_feature_size)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        done = self._prepare_next_decision()
        if done:
            reward = self._calculate_final_reward()
            return self._get_state(), reward, True, {}

        feasible_events = self.current_feasible_events

        if action >= len(feasible_events):
            zbpp_choice = self._get_zbpp_choice(feasible_events)
            selected_event = feasible_events[zbpp_choice]
            reward = -0.5
        else:
            selected_event = feasible_events[action]
            reward = self._calculate_step_reward(selected_event, feasible_events)

        self._schedule_selected_event(selected_event)
        self.current_feasible_events = []
        done = self._prepare_next_decision()

        if done:
            reward += self._calculate_final_reward()

        return self._get_state(), reward, done, {}
    
    def _get_zbpp_choice(self, feasible_events: List[Dict[str, Any]]) -> int:
        """获取ZBPP的选择"""
        # 简单的启发式：优先选择Backward，然后是Forward，最后是Weight Update
        priorities = {
            OperationType.BACKWARD: 0,
            OperationType.FORWARD: 1,
            OperationType.WEIGHT_UPDATE: 2
        }
        
        # 按优先级排序
        sorted_events = sorted(feasible_events, key=lambda x: priorities[x['type']])
        return feasible_events.index(sorted_events[0])
    
    def _calculate_step_reward(self, selected_event: Dict[str, Any], 
                             feasible_events: List[Dict[str, Any]]) -> float:
        """计算步骤奖励"""
        reward = 0.0
        
        # 基础奖励
        reward += 0.1  # 成功调度奖励
        
        # 效率奖励：选择执行时间短的事件
        min_duration = min(e['duration'] for e in feasible_events)
        max_duration = max(e['duration'] for e in feasible_events)
        if max_duration > min_duration:
            efficiency = 1.0 - (selected_event['duration'] - min_duration) / (max_duration - min_duration)
            reward += 0.2 * efficiency

        # 轻量的启发式蒸馏：与ZBPP启发式一致时给少量奖励，加快前期收敛。
        if feasible_events[self._get_zbpp_choice(feasible_events)] is selected_event:
            reward += 0.1
        
        return reward
    
    def _calculate_final_reward(self) -> float:
        """计算最终奖励"""
        if not self.scheduled_events:
            return -10.0  # 完全失败

        completion_time = max(event['end_time'] for event in self.scheduled_events)
        zbpp_time = max(event['end_time'] for event in self.zbpp_sequence)

        time_delta_ratio = (zbpp_time - completion_time) / max(zbpp_time, 1.0)
        if time_delta_ratio >= 0:
            time_reward = 100.0 * time_delta_ratio
        else:
            time_reward = 50.0 * time_delta_ratio

        if completion_time <= zbpp_time * 0.99:
            time_reward += 20.0
        elif completion_time <= zbpp_time * 1.01:
            time_reward += 5.0

        completion_ratio = len(self.scheduled_events) / max(len(self.event_list), 1)
        completion_reward = 10.0 * completion_ratio
        if completion_ratio < 1.0:
            completion_reward -= 20.0 * (1.0 - completion_ratio)

        device_busy_time = sum(event['duration'] for event in self.scheduled_events)
        utilization = device_busy_time / max(
            completion_time * self.simulator.num_devices, 1.0
        )
        bubble_penalty = (1.0 - utilization) * 30.0

        device_times = [0.0] * self.simulator.num_devices
        for event in self.scheduled_events:
            device_times[event['device_id']] = max(device_times[event['device_id']], event['end_time'])
        balance_penalty = np.var(device_times) / max(completion_time, 1.0) * 10.0

        return time_reward + completion_reward - bubble_penalty - balance_penalty

    def _schedule_selected_event(self, selected_event: Dict[str, Any]):
        """在当前环境状态上真实执行一次调度。"""
        device_id = selected_event['device_id']

        # 防止因为动作快照失真导致同一个事件被重复加入。
        event_key = (
            selected_event['type'],
            selected_event['micro_batch_id'],
            selected_event['device_id'],
        )
        scheduled_keys = {
            (event['type'], event['micro_batch_id'], event['device_id'])
            for event in self.scheduled_events
        }
        if event_key in scheduled_keys:
            return

        start_time = max(self.current_time, self.device_available_time[device_id])
        selected_event['start_time'] = start_time
        selected_event['end_time'] = start_time + selected_event['duration']

        self.scheduled_events.append(selected_event)
        self.device_available_time[device_id] = selected_event['end_time']

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # 共享特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor网络 (策略)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic网络 (价值函数)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared_net(x)
        action_logits = self.actor(shared_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        state_values = self.critic(shared_features)
        return action_probs, state_values

class PPOScheduler:
    """PPO调度器"""
    
    def __init__(self, simulator: PipelineSimulator, 
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 epochs: int = 10,
                 batch_size: int = 64,
                 bc_rollouts: int = 64,
                 bc_epochs: int = 10):
        
        self.simulator = simulator
        self.env = PipelineSchedulingEnv(
            simulator.num_devices, 
            simulator.num_micro_batches,
            simulator.micro_batch_size,
            simulator.max_memory,
            getattr(simulator, "noise_level", 0.1),
        )
        
        # PPO超参数
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.bc_rollouts = bc_rollouts
        self.bc_epochs = bc_epochs
        
        # 网络初始化
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.policy = ActorCritic(state_dim, action_dim)
        self._init_weights()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 训练记录
        self.rewards_history = []
        self.loss_history = []
        self.last_train_metrics: Dict[str, Any] = {}
        self.last_schedule_metrics: Dict[str, Any] = {}
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.policy.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def _get_masked_action_distribution(
        self, action_probs: torch.Tensor, feasible_count: int
    ) -> Categorical:
        """在当前可行动作子空间上构造分布。"""
        valid_action_probs = action_probs[:, :feasible_count]
        valid_action_probs = valid_action_probs / valid_action_probs.sum(dim=-1, keepdim=True)
        return Categorical(valid_action_probs)

    def _policy_inference(
        self, state: np.ndarray, feasible_count: int, deterministic: bool = False
    ) -> Tuple[int, float, float, float]:
        """
        执行一次策略前向并返回动作、log_prob、value 和耗时。

        Returns:
            action, log_prob, state_value, latency_sec
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        t_start = time.perf_counter()
        with torch.no_grad():
            action_probs, state_value = self.policy(state_tensor)
            dist = self._get_masked_action_distribution(action_probs, feasible_count)
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        latency_sec = time.perf_counter() - t_start

        return action.item(), log_prob.item(), state_value.item(), latency_sec

    def schedule(self, profile_latency: bool = False) -> List[Dict[str, Any]]:
        """使用训练好的策略进行调度，并可选统计单次调用延迟。"""
        state = self.env.reset()
        done = False
        policy_call_latencies = []
        decision_count = 0
        
        while not done:
            feasible_events = self.env.get_action_candidates()
            if not feasible_events:
                break
            action, _, _, latency_sec = self._policy_inference(
                state, len(feasible_events), deterministic=True
            )
            if profile_latency:
                policy_call_latencies.append(latency_sec)
            decision_count += 1
            state, reward, done, _ = self.env.step(action)

        if profile_latency and policy_call_latencies:
            self.last_schedule_metrics = {
                'decision_count': decision_count,
                'avg_policy_call_ms': float(np.mean(policy_call_latencies) * 1000.0),
                'p99_policy_call_ms': float(np.percentile(policy_call_latencies, 99) * 1000.0),
                'single_policy_call_ms': float(policy_call_latencies[0] * 1000.0),
            }
        else:
            self.last_schedule_metrics = {'decision_count': decision_count}

        return self.env.scheduled_events

    def _behavioral_clone(self):
        """使用ZBPP启发式动作进行监督预训练。"""
        if self.bc_rollouts <= 0 or self.bc_epochs <= 0:
            return

        bc_states = []
        bc_actions = []
        bc_valid_action_counts = []

        for _ in range(self.bc_rollouts):
            state = self.env.reset()
            done = False

            while not done:
                feasible_events = self.env.get_action_candidates()
                if not feasible_events:
                    break

                expert_action = self.env._get_zbpp_choice(feasible_events)
                bc_states.append(state)
                bc_actions.append(expert_action)
                bc_valid_action_counts.append(len(feasible_events))
                state, _, done, _ = self.env.step(expert_action)

        if not bc_states:
            return

        states = torch.FloatTensor(np.array(bc_states))
        actions = torch.LongTensor(bc_actions)
        valid_action_counts = torch.LongTensor(bc_valid_action_counts)

        for _ in range(self.bc_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_valid_action_counts = valid_action_counts[batch_indices]

                action_probs, _ = self.policy(batch_states)
                expert_log_probs, _ = self._compute_masked_policy_stats(
                    action_probs, batch_actions, batch_valid_action_counts
                )
                bc_loss = -expert_log_probs.mean()

                self.optimizer.zero_grad()
                bc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
    
    def train(self, num_episodes: int = 1000):
        """训练PPO策略，并记录训练总耗时（用于训练成本分析）"""
        print("开始训练PPO调度器...")
        t_start = time.perf_counter()
        total_env_steps = 0
        total_policy_updates = 0
        policy_call_latencies = []
        best_improvement_ratio = float("-inf")
        best_policy_state = copy.deepcopy(self.policy.state_dict())

        self._behavioral_clone()
        
        for episode in range(num_episodes):
            # 重置环境
            state = self.env.reset()
            done = False
            
            states = []
            actions = []
            valid_action_counts = []
            old_log_probs = []
            rewards = []
            values = []
            masks = []
            
            episode_reward = 0
            
            while not done:
                feasible_events = self.env.get_action_candidates()
                if not feasible_events:
                    break
                action, old_log_prob, state_value, latency_sec = self._policy_inference(
                    state, len(feasible_events), deterministic=False
                )
                policy_call_latencies.append(latency_sec)
                
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储经验
                states.append(state)
                actions.append(action)
                valid_action_counts.append(len(feasible_events))
                old_log_probs.append(old_log_prob)
                rewards.append(reward)
                values.append(state_value)
                masks.append(1 - done)
                
                state = next_state
                episode_reward += reward
                total_env_steps += 1
            
            # 记录奖励
            self.rewards_history.append(episode_reward)
            
            # 每10轮打印一次进度
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                      f"Avg Reward: {np.mean(self.rewards_history[-10:]):.2f}")
            
            # 更新策略
            if len(states) > 0:
                self._update_policy(
                    states, actions, valid_action_counts, old_log_probs, rewards, values, masks
                )
                total_policy_updates += self.epochs * int(np.ceil(len(states) / self.batch_size))

            if (episode + 1) % 25 == 0 or episode == num_episodes - 1:
                eval_metrics = self.evaluate_vs_zbpp()
                eval_improvement_ratio = eval_metrics.get('improvement_ratio', float("-inf"))
                if eval_improvement_ratio > best_improvement_ratio:
                    best_improvement_ratio = eval_improvement_ratio
                    best_policy_state = copy.deepcopy(self.policy.state_dict())

        self.policy.load_state_dict(best_policy_state)
        
        t_end = time.perf_counter()
        total_time_sec = t_end - t_start
        avg_policy_call_ms = (
            float(np.mean(policy_call_latencies) * 1000.0) if policy_call_latencies else 0.0
        )
        p99_policy_call_ms = (
            float(np.percentile(policy_call_latencies, 99) * 1000.0) if policy_call_latencies else 0.0
        )
        single_policy_call_ms = (
            float(policy_call_latencies[0] * 1000.0) if policy_call_latencies else 0.0
        )
        self.last_train_metrics = {
            'num_episodes': num_episodes,
            'total_wall_time_sec': float(total_time_sec),
            'total_env_steps': int(total_env_steps),
            'total_policy_updates': int(total_policy_updates),
            'best_eval_improvement_ratio': float(best_improvement_ratio),
            'avg_policy_call_ms': avg_policy_call_ms,
            'p99_policy_call_ms': p99_policy_call_ms,
            'single_policy_call_ms': single_policy_call_ms,
        }
        print("训练完成!")
        print(f"[TrainCost] num_episodes = {num_episodes}")
        print(f"[TrainCost] total_wall_time_sec = {total_time_sec:.2f}")
        print(f"[TrainCost] total_env_steps = {total_env_steps}")
        print(f"[TrainCost] total_policy_updates = {total_policy_updates}")
        print(f"[TrainCost] best_eval_improvement_ratio = {best_improvement_ratio:.4f}")
        print(f"[TrainCost] avg_policy_call_ms = {avg_policy_call_ms:.4f}")
        print(f"[TrainCost] p99_policy_call_ms = {p99_policy_call_ms:.4f}")
        print(f"[TrainCost] single_policy_call_ms = {single_policy_call_ms:.4f}")
    
    def _update_policy(self, states, actions, valid_action_counts, old_log_probs, rewards, values, masks):
        """更新策略网络"""
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        valid_action_counts = torch.LongTensor(valid_action_counts)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        masks = torch.FloatTensor(masks)
        
        # 计算回报
        returns = self._compute_returns(rewards, masks)
        
        # 标准化回报
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()
        else:
            returns = returns - returns.mean()
        
        # 计算优势函数
        advantages = returns - values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()
        else:
            advantages = advantages - advantages.mean()
        
        # 多轮优化
        for _ in range(self.epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_valid_action_counts = valid_action_counts[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices].detach()
                
                action_probs, state_values = self.policy(batch_states)
                new_log_probs, entropy = self._compute_masked_policy_stats(
                    action_probs, batch_actions, batch_valid_action_counts
                )
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                if state_values.dim() > 1:
                    state_values = state_values.squeeze()
                if batch_returns.dim() > 1:
                    batch_returns = batch_returns.squeeze()
                value_loss = nn.MSELoss()(state_values, batch_returns)
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                self.loss_history.append(loss.item())

    def _compute_masked_policy_stats(self, action_probs, actions, valid_action_counts):
        """在每个样本自己的可行动作子空间内恢复log-prob和熵。"""
        log_probs = []
        entropies = []

        for probs, action, valid_count in zip(action_probs, actions, valid_action_counts):
            valid_count_int = int(valid_count.item())
            valid_probs = probs[:valid_count_int]
            valid_probs = valid_probs / valid_probs.sum()
            dist = Categorical(valid_probs)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

        return torch.stack(log_probs), torch.stack(entropies).mean()
    
    def _compute_returns(self, rewards, masks):
        """计算折扣回报"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * masks[t] * running_return
            returns[t] = running_return
        
        return returns
    
    def evaluate_vs_zbpp(self) -> Dict[str, Any]:
        """与ZBPP进行性能比较"""
        # PPO调度
        ppo_events = self.schedule(profile_latency=True)
        ppo_time = max(event['end_time'] for event in ppo_events) if ppo_events else float('inf')
        
        # ZBPP调度
        zbpp_events = self.env.zbpp_sequence
        zbpp_time = max(event['end_time'] for event in zbpp_events) if zbpp_events else float('inf')
        
        # 计算改进
        improvement = zbpp_time - ppo_time
        improvement_ratio = improvement / zbpp_time if zbpp_time > 0 else 0
        throughput_improvement_ratio = (zbpp_time / ppo_time - 1.0) if ppo_time > 0 else float('-inf')
        
        results = {
            'ppo_completion_time': ppo_time,
            'zbpp_completion_time': zbpp_time,
            'time_improvement': improvement,
            'improvement_ratio': improvement_ratio,
            'throughput_improvement_ratio': throughput_improvement_ratio,
            'ppo_events_scheduled': len(ppo_events),
            'zbpp_events_scheduled': len(zbpp_events),
        }
        results.update(self.last_schedule_metrics)
        return results

    def get_train_metrics(self) -> Dict[str, Any]:
        """返回最近一次训练指标。"""
        return dict(self.last_train_metrics)

    def get_schedule_metrics(self) -> Dict[str, Any]:
        """返回最近一次调度指标。"""
        return dict(self.last_schedule_metrics)

# 使用示例
if __name__ == "__main__":
    # 创建模拟器
    simulator = PipelineSimulator(
        num_devices=4,
        num_micro_batches=8,
        micro_batch_size=32,
        max_memory=4.0
    )
    
    # 创建PPO调度器
    ppo_scheduler = PPOScheduler(simulator)
    
    # ===== 训练成本分析 =====
    num_episodes = 500  # 如需与论文设置一致，可在此修改
    ppo_scheduler.train(num_episodes=num_episodes)
    # 此时终端中会打印：
    # [TrainCost] num_episodes = ...
    # [TrainCost] total_wall_time_sec = ...
    
    # ===== Ablation：ZBPP vs ZBPP+RL 提升 =====
    results = ppo_scheduler.evaluate_vs_zbpp()
    print("\n性能比较结果 (ZBPP vs ZBPP+RL):")
    for key, value in results.items():
        print(f"{key}: {value}")
    if results["zbpp_completion_time"] > 0:
        improve_pct = results["improvement_ratio"] * 100.0
        throughput_improve_pct = results["throughput_improvement_ratio"] * 100.0
        print(f"[Ablation] RL 调度相对 ZBPP 的时间提升: {improve_pct:+.2f}%")
        print(f"[Ablation] RL 调度相对 ZBPP 的吞吐提升: {throughput_improve_pct:+.2f}%")
        print(
            f"[InferenceCost] 单次策略调用耗时: "
            f"{results.get('single_policy_call_ms', 0.0):.4f} ms"
        )
    
