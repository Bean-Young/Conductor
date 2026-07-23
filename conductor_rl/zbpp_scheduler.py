#!/usr/bin/env python3
"""
Zero Bubble Pipeline Parallelism (ZBPP) Scheduler
实现零气泡流水线并行调度算法

ZBPP核心特点：
1. 不区分第一阶段和第二阶段，统一调度
2. B和W被分为两个不同的步骤
3. 通过精心安排执行顺序实现零气泡
4. 支持debug输出可执行事件
"""

import random
from typing import List, Dict, Any, Optional
from .pipeline_simulator import PipelineSimulator, OperationType

class ZBPPScheduler:
    """
    Zero Bubble Pipeline Parallelism调度器
    
    算法原理：
    1. 将Backward操作分解为B(backward)和W(weight update)两个独立阶段
    2. 使用特定的调度模式来最小化pipeline bubble
    3. 通过B和W的分离实现更好的并行性
    4. 统一调度，不区分阶段
    """
    
    def __init__(self, simulator: PipelineSimulator):
        """
        初始化ZBPP调度器
        
        Args:
            simulator: 流水线并行仿真器实例
        """
        self.simulator = simulator
        self.scheduled_events = []
        self.current_time = 0.0
        self.device_available_time = {i: 0.0 for i in range(simulator.num_devices)}
        self.event_list = []
        
        # Debug输出控制
        self.debug_output = False
    
    def schedule(self) -> List[Dict[str, Any]]:
        """
        执行ZBPP调度算法
        
        Returns:
            List[Dict[str, Any]]: 调度后的事件列表
        """
        # 生成事件
        self.simulator.generate_events()
        self.event_list = self.simulator.get_event_list()
        
        
        # 执行ZBPP调度
        self._schedule_zbpp()
        
        
        return self.scheduled_events
    
    def _schedule_zbpp(self):
        """
        执行ZBPP调度算法
        
        ZBPP核心思想：
        1. 不区分阶段，统一调度F、B、W操作
        2. B和W是独立的步骤
        3. 通过精心安排执行顺序实现零气泡
        4. 关键：同一个微批次的不同操作不能重叠
        5. 重新设计：确保微批次内的操作按顺序执行
        """
        # 重新设计ZBPP：按微批次顺序调度，确保每个微批次的F、B、W按顺序执行
        self._schedule_zbpp_correct()
    
    def _schedule_zbpp_correct(self):
        """
        正确的ZBPP调度算法
        
        核心思想：
        1. 微批次内的操作必须按顺序执行：F -> B -> W
        2. 不同微批次的相同操作可以并行执行
        3. 通过精心安排实现零气泡
        4. 关键：确保微批次内的操作不重叠
        """
        # 重新设计：按微批次顺序调度，确保每个微批次的F、B、W按顺序执行
        self._schedule_by_microbatch()
    
    def _schedule_by_microbatch(self):
        """
        按微批次顺序调度ZBPP - 修正版本
        
        核心思想：
        1. 一个设备每个step只能选择一个可执行的任务
        2. 确保微批次内的操作按顺序执行：F -> B -> W
        3. 不同微批次的相同操作可以并行执行
        4. 关键：微批次内的操作不能重叠
        """
        step_count = 0
        
        while True:
            step_count += 1
            
            
            # 获取可执行事件
            feasible_events = self._get_feasible_events()
            
            
            if not feasible_events:
                break
            
            # 获取可用设备
            available_devices = self._get_available_devices()
            
            
            if not available_devices:
                # 没有设备可用，推进到最早完成的事件时间
                running_end_times = [e['end_time'] for e in self.scheduled_events if e['end_time'] > self.current_time]
                if not running_end_times:
                    break
                
                next_time = min(running_end_times)
                self.current_time = next_time
                continue
            
            # 为每个可用设备选择事件 - 确保一个设备只选择一个任务
            selected_events = []
            
            for device_id in available_devices:
                device_candidates = [e for e in feasible_events if e['device_id'] == device_id]
                
                if not device_candidates:
                    continue
                
                # 关键修改：确保微批次内的操作按顺序执行
                chosen = self._select_event_for_device_zbpp(device_id, device_candidates)
                
                if chosen is not None:
                    # 检查总体内存限制
                    if self._check_memory_feasible(chosen):
                        selected_events.append(chosen)
                    else:
                        # 内存不足，跳过这个事件
                        pass
                else:
                    pass  # 设备在候选事件中无可行项，跳过
            
            if not selected_events:
                # 推进到最近完成的正在运行事件时间
                running_end_times = [e['end_time'] for e in self.scheduled_events if e['end_time'] > self.current_time]
                if not running_end_times:
                    break
                
                self.current_time = min(running_end_times)
                continue
            
            # 调度选中的事件
            for event in selected_events:
                self._schedule_event(event)
            
            # 推进时间到当前所有正在运行事件中最晚完成的时刻
            running_end_times = [e['end_time'] for e in self.scheduled_events if e['end_time'] > self.current_time]
            if running_end_times:
                next_time = max(running_end_times)
                self.current_time = next_time
    
    def _create_microbatch_sequences(self):
        """
        为每个微批次创建操作序列
        
        返回：字典，键为微批次ID，值为该微批次的操作序列
        """
        sequences = {}
        
        for mb_id in range(self.simulator.num_micro_batches):
            sequence = []
            
            # Forward操作序列：从设备0到设备N-1
            for device_id in range(self.simulator.num_devices):
                forward_event = self._find_event(OperationType.FORWARD, mb_id, device_id)
                if forward_event:
                    sequence.append(forward_event)
            
            # Backward操作序列：从设备N-1到设备0
            for device_id in range(self.simulator.num_devices - 1, -1, -1):
                backward_event = self._find_event(OperationType.BACKWARD, mb_id, device_id)
                if backward_event:
                    sequence.append(backward_event)
            
            # Weight Update操作序列：从设备0到设备N-1
            for device_id in range(self.simulator.num_devices):
                weight_event = self._find_event(OperationType.WEIGHT_UPDATE, mb_id, device_id)
                if weight_event:
                    sequence.append(weight_event)
            
            sequences[mb_id] = sequence
        
        return sequences
    
    def _select_event_for_device_zbpp(self, device_id: int, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        为设备选择事件 - ZBPP优化版本
        
        ZBPP调度策略：
        1. B > F > W 基本优先级
        2. 但是：如果执行B会导致下游stage空闲，则优先执行F
        3. 确保微批次内的F、B、W按顺序执行
        """
        # 分离不同类型的候选事件
        forward_candidates = [e for e in candidates if e['type'] == OperationType.FORWARD]
        backward_candidates = [e for e in candidates if e['type'] == OperationType.BACKWARD]
        weight_candidates = [e for e in candidates if e['type'] == OperationType.WEIGHT_UPDATE]
        
        # 过滤出可行的候选事件
        feasible_forward = [e for e in forward_candidates 
                           if self._check_memory_feasible(e) and self._check_microbatch_sequence(e)]
        feasible_backward = [e for e in backward_candidates 
                            if self._check_memory_feasible(e) and self._check_microbatch_sequence(e)]
        feasible_weight = [e for e in weight_candidates 
                          if self._check_memory_feasible(e) and self._check_microbatch_sequence(e)]
        
        # ZBPP核心逻辑：B > F > W，但需要避免下游空闲
        # 1. 如果有B且下游有ready任务，优先执行B
        if feasible_backward and self._downstream_has_ready_task(device_id):
            return feasible_backward[0]
        
        # 2. 如果有F且下游没有ready任务，优先执行F（避免下游空闲）
        if feasible_forward and not self._downstream_has_ready_task(device_id):
            return feasible_forward[0]
        
        # 3. 否则按照 B > F > W 的优先级执行
        if feasible_backward:
            return feasible_backward[0]
        if feasible_forward:
            return feasible_forward[0]
        if feasible_weight:
            return feasible_weight[0]
        
        return None
    
    def _downstream_has_ready_task(self, device_id: int) -> bool:
        """
        检查下游stage是否有ready任务
        
        ZBPP核心：如果执行B会导致下游stage空闲，则优先执行F
        
        Args:
            device_id: 当前设备ID
            
        Returns:
            bool: 下游是否有ready任务
        """
        # 检查下游设备(device_id + 1)是否有可执行的任务
        if device_id + 1 >= self.simulator.num_devices:
            return True  # 最后一个设备，没有下游
        
        downstream_device = device_id + 1
        
        # 获取下游设备的所有可执行事件
        downstream_candidates = []
        for event in self.event_list:
            if (event['device_id'] == downstream_device and 
                event not in self.scheduled_events and
                self._check_dependencies(event)):
                downstream_candidates.append(event)
        
        # 检查下游是否有Forward任务ready（这是关键）
        # Forward任务会解锁更下游的stage，避免空闲
        for candidate in downstream_candidates:
            if candidate['type'] == OperationType.FORWARD:
                if self._check_memory_feasible(candidate) and self._check_microbatch_sequence(candidate):
                    return True
        
        # 也检查Backward任务，但Forward更重要
        for candidate in downstream_candidates:
            if candidate['type'] == OperationType.BACKWARD:
                if self._check_memory_feasible(candidate) and self._check_microbatch_sequence(candidate):
                    return True
        
        return False
    
    def _check_microbatch_sequence(self, event: Dict[str, Any]) -> bool:
        """
        检查微批次内的操作是否按正确顺序执行 - ZBPP版本
        
        ZBPP核心思想：B和W是独立的操作，不需要在B完成后立即执行W
        关键：确保微批次内的F、B、W操作按顺序执行，但B和W可以独立调度
        """
        mb_id = event['micro_batch_id']
        device_id = event['device_id']
        op_type = event['type']
        
        # 获取该微批次在该设备上已调度的操作，按时间排序
        device_events = []
        for scheduled_event in self.scheduled_events:
            if (scheduled_event['micro_batch_id'] == mb_id and 
                scheduled_event['device_id'] == device_id):
                device_events.append(scheduled_event)
        
        # 按开始时间排序
        device_events.sort(key=lambda x: x['start_time'])
        
        if op_type == OperationType.FORWARD:
            # F操作：该微批次在该设备上还没有任何操作
            return len(device_events) == 0
            
        elif op_type == OperationType.BACKWARD:
            # B操作：该微批次在该设备上必须有F操作，且该微批次在所有设备上的F操作都已完成
            if len(device_events) == 0 or device_events[0]['type'] != OperationType.FORWARD:
                return False
            
            # 检查该微批次在所有设备上的F操作是否都已完成
            for dev_id in range(self.simulator.num_devices):
                f_event = self._find_event(OperationType.FORWARD, mb_id, dev_id)
                if f_event and f_event not in self.scheduled_events:
                    return False
            
            return True
            
        elif op_type == OperationType.WEIGHT_UPDATE:
            # W操作：该微批次在该设备上必须有B操作，但不需要立即执行
            # 检查是否有B操作
            has_backward = any(e['type'] == OperationType.BACKWARD for e in device_events)
            return has_backward
        
        return False
    
    def _select_event_for_device(self, device_id: int, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        为设备选择事件 - 原版本（保留兼容性）
        """
        # 按微批次ID和操作类型排序，确保微批次内操作按顺序执行
        def get_priority(event):
            mb_id = event['micro_batch_id']
            op_type = event['type']
            
            # 操作类型优先级：F=0, B=1, W=2
            type_priority = {
                OperationType.FORWARD: 0,
                OperationType.BACKWARD: 1,
                OperationType.WEIGHT_UPDATE: 2
            }
            
            return (mb_id, type_priority[op_type])
        
        # 按优先级排序
        candidates.sort(key=get_priority)
        
        # 选择第一个可行的候选
        for candidate in candidates:
            if self._check_memory_feasible(candidate):
                return candidate
        
        return None
    
    def _get_feasible_events(self) -> List[Dict[str, Any]]:
        """
        获取可执行事件（依赖已满足且未调度）
        """
        feasible_events = []
        
        # 创建已调度事件的集合，使用事件标识符而不是对象引用
        scheduled_event_keys = set()
        for scheduled_event in self.scheduled_events:
            key = (scheduled_event['type'], scheduled_event['micro_batch_id'], scheduled_event['device_id'])
            scheduled_event_keys.add(key)
        
        for event in self.event_list:
            # 使用事件标识符检查是否已调度
            event_key = (event['type'], event['micro_batch_id'], event['device_id'])
            if event_key in scheduled_event_keys:
                continue
            
            if self._check_dependencies(event):
                feasible_events.append(event)
        
        return feasible_events
    
    def _check_dependencies(self, event: Dict[str, Any]) -> bool:
        """
        检查事件依赖是否满足
        """
        op_type = event['type']
        mb_id = event['micro_batch_id']
        device_id = event['device_id']

        def finished(op: OperationType, micro_batch_id: int, dep_device_id: int) -> bool:
            dep_event = self._find_event(op, micro_batch_id, dep_device_id)
            if not dep_event or dep_event not in self.scheduled_events:
                return False
            end_time = dep_event.get('end_time')
            return end_time is not None and end_time <= self.current_time
        
        if op_type == OperationType.FORWARD:
            # Forward: F(m,n) 依赖 F(m, n-1) 和 F(m-1, n)
            # 1. 依赖前一个设备上的Forward
            if device_id > 0:
                if not finished(OperationType.FORWARD, mb_id, device_id - 1):
                    return False
            
            # 2. 依赖前一个微批次在同一个设备上的Forward
            if mb_id > 0:
                if not finished(OperationType.FORWARD, mb_id - 1, device_id):
                    return False
            
        elif op_type == OperationType.BACKWARD:
            # Backward: B(m,n) 依赖 B(m, n+1) 和 F(m,N)
            if device_id < self.simulator.num_devices - 1:
                if not finished(OperationType.BACKWARD, mb_id, device_id + 1):
                    return False
            
            # 依赖最后一个设备的Forward
            if not finished(OperationType.FORWARD, mb_id, self.simulator.num_devices - 1):
                return False
            
        elif op_type == OperationType.WEIGHT_UPDATE:
            # Weight Update: W(m,n) 依赖 B(m,n)
            if not finished(OperationType.BACKWARD, mb_id, device_id):
                return False
        
        return True
    
    def _check_memory_feasible(self, event: Dict[str, Any]) -> bool:
        """
        检查事件是否显存可行
        """
        device_id = event['device_id']
        current_memory = self._get_current_memory_usage(device_id)
        memory_change = self._get_memory_change(event)
        
        
        return current_memory + memory_change <= self.simulator.max_memory
    
    def _check_total_memory_feasible(self, event: Dict[str, Any]) -> bool:
        """
        检查设备内存限制是否可行（每个设备独立的内存限制）
        """
        # 计算目标设备的当前内存使用
        device_id = event['device_id']
        current_memory = self._get_current_memory_usage(device_id)
        
        # 计算新事件的内存变化
        memory_change = self._get_memory_change(event)
        
        # 检查是否超过该设备的内存限制
        new_memory = current_memory + memory_change
        is_feasible = new_memory <= self.simulator.max_memory
        
        # 调试输出
        print(f"    内存检查: 当前{current_memory:.1f} + 变化{memory_change:+.1f} = {new_memory:.1f} <= {self.simulator.max_memory} = {is_feasible}")
        
        return is_feasible
    
    def _get_current_memory_usage(self, device_id: int) -> float:
        """
        获取设备当前显存使用量
        F操作触发时+1，B操作触发时-0.5，W操作触发时-0.5
        与结束无关，与其他操作无关
        """
        memory_usage = 0.0
        
        for event in self.scheduled_events:
            if event['device_id'] == device_id and event['start_time'] <= self.current_time:
                # 每个操作触发时改变显存，与结束无关
                memory_usage += self._get_memory_change(event)
        
        return memory_usage
    
    def _get_memory_change(self, event: Dict[str, Any]) -> float:
        """
        获取事件的显存变化
        """
        if event['type'] == OperationType.FORWARD:
            return 1.0
        elif event['type'] == OperationType.BACKWARD:
            return -0.5
        elif event['type'] == OperationType.WEIGHT_UPDATE:
            return -0.5
        else:
            return 0.0
    
    def _get_available_devices(self) -> List[int]:
        """
        获取可用设备列表
        """
        available_devices = []
        
        for device_id in range(self.simulator.num_devices):
            if self.device_available_time[device_id] <= self.current_time:
                available_devices.append(device_id)
        
        return available_devices
    
    def _find_event(self, op_type: OperationType, micro_batch_id: int, device_id: int) -> Optional[Dict[str, Any]]:
        """
        查找事件
        """
        for event in self.event_list:
            if (event['type'] == op_type and 
                event['micro_batch_id'] == micro_batch_id and 
                event['device_id'] == device_id):
                return event
        return None
    
    def _schedule_event(self, event: Dict[str, Any]):
        """
        调度事件 - 确保同一个微批次的不同操作不会重叠
        """
        device_id = event['device_id']
        mb_id = event['micro_batch_id']
        
        # 检查该微批次在该设备上的最后一个事件
        last_event_time = 0.0
        for scheduled_event in self.scheduled_events:
            if (scheduled_event['device_id'] == device_id and 
                scheduled_event['micro_batch_id'] == mb_id):
                last_event_time = max(last_event_time, scheduled_event['end_time'])
        
        # 设置开始和结束时间
        # 关键：确保同一个微批次的不同操作不会重叠
        start_time = max(self.current_time, self.device_available_time[device_id], last_event_time)
        event['start_time'] = start_time
        event['end_time'] = start_time + event['duration']
        
        # 添加到调度列表
        self.scheduled_events.append(event)
        
        # 更新设备可用时间
        self.device_available_time[device_id] = event['end_time']
        
