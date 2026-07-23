#!/usr/bin/env python3
"""Event-driven pipeline simulator used by the Conductor RL scheduler."""

import random
from enum import Enum
from typing import List, Dict, Any, Optional

class OperationType(Enum):
    """操作类型枚举"""
    FORWARD = "F"
    BACKWARD = "B"
    WEIGHT_UPDATE = "W"

class Operation:
    """操作类"""
    def __init__(self, op_type: OperationType, micro_batch_id: int, device_id: int):
        self.op_type = op_type
        self.micro_batch_id = micro_batch_id
        self.device_id = device_id
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.end_time: Optional[float] = None
        self.dependencies: List['Operation'] = []

class TimeModel:
    """时间模型 - 模拟计算时间和通信时间"""
    
    def __init__(self, micro_batch_size: int, noise_level: float = 0.1):
        self.micro_batch_size = micro_batch_size
        self.noise_level = noise_level
        
        # 基础计算时间（根据设备性能不同）
        self.base_compute_times = {
            OperationType.FORWARD: 10.0,      # Forward基础时间
            OperationType.BACKWARD: 10.0,     # Backward基础时间
            OperationType.WEIGHT_UPDATE: 7.0  # Weight Update基础时间
        }
        
        # 通信时间
        self.communication_time = 0.4
    
    def get_compute_time(self, op_type: OperationType, device_id: int) -> float:
        """获取计算时间（包含波动）"""
        base_time = self.base_compute_times[op_type]
        
        # 设备性能差异
        device_factor = 1.0 + (device_id * 0.1)  # 设备ID越大，性能越好
        
        # 添加波动
        noise = random.uniform(-self.noise_level, self.noise_level)
        
        return base_time * device_factor * (1 + noise)
    
    def get_communication_time(self, device_id: int) -> float:
        """获取通信时间"""
        return self.communication_time

class PipelineSimulator:
    """Pipeline模拟器 - 重写版本"""
    
    def __init__(self, num_devices: int, num_micro_batches: int, 
                 micro_batch_size: int, max_memory: float, noise_level: float = 0.1):
        self.num_devices = num_devices  # P
        self.num_micro_batches = num_micro_batches  # M
        self.micro_batch_size = micro_batch_size
        self.max_memory = max_memory  # N
        self.noise_level = noise_level  # X
        
        # 时间模型
        self.time_model = TimeModel(micro_batch_size, noise_level)
        
        # 事件表
        self.event_list: List[Dict[str, Any]] = []
        
        # 显存管理
        self.current_memory = 0.0
        self.memory_history = []
        
        # 统计信息
        self.stats = {
            'max_memory_used': 0.0,
            'memory_violations': 0,
            'total_operations': 0
        }
    
    def check_operation_feasible(self, op_type: OperationType, micro_batch_id: int, device_id: int) -> bool:
        """检查事件是否可行"""
        
        if op_type == OperationType.FORWARD:
            # Forward操作：F(m,n) 依赖 F(m, n-1)，显存可行
            # 检查依赖
            if device_id > 0:
                # 需要检查前一个设备的Forward操作是否存在
                prev_forward = self._find_operation(OperationType.FORWARD, micro_batch_id, device_id - 1)
                if prev_forward is None:
                    return False
            
            # 检查显存
            return self._check_memory_feasible(op_type)
            
        elif op_type == OperationType.BACKWARD:
            # Backward操作：B(m,n) 依赖 B(m, n+1) 和 F(m,N)
            # 检查依赖
            if device_id < self.num_devices - 1:
                # 需要检查后一个设备的Backward操作是否存在
                next_backward = self._find_operation(OperationType.BACKWARD, micro_batch_id, device_id + 1)
                if next_backward is None:
                    return False
            
            # 需要检查最后一个设备的Forward操作是否存在
            last_forward = self._find_operation(OperationType.FORWARD, micro_batch_id, self.num_devices - 1)
            if last_forward is None:
                return False
            
            # 检查显存
            return self._check_memory_feasible(op_type)
            
        elif op_type == OperationType.WEIGHT_UPDATE:
            # Weight Update操作：W(m,n) 依赖 B(m,n)
            # 检查依赖
            backward_op = self._find_operation(OperationType.BACKWARD, micro_batch_id, device_id)
            if backward_op is None:
                return False
            
            # 检查显存
            return self._check_memory_feasible(op_type)
        
        return False
    
    def _find_operation(self, op_type: OperationType, micro_batch_id: int, device_id: int) -> Optional[Operation]:
        """查找操作"""
        for event in self.event_list:
            if (event['type'] == op_type and 
                event['micro_batch_id'] == micro_batch_id and 
                event['device_id'] == device_id):
                return event.get('operation')
        return None
    
    def _check_memory_feasible(self, op_type: OperationType) -> bool:
        """检查显存是否可行"""
        memory_impact = self._get_memory_impact(op_type)
        return self.current_memory + memory_impact <= self.max_memory
    
    def _get_memory_impact(self, op_type: OperationType) -> float:
        """获取显存影响"""
        if op_type == OperationType.FORWARD:
            return 1.0  # Forward操作 +1 显存
        elif op_type == OperationType.BACKWARD:
            return -0.5  # Backward操作 -0.5 显存
        elif op_type == OperationType.WEIGHT_UPDATE:
            return -0.5  # Weight Update操作 -0.5 显存
        else:
            return 0.0
    
    def simulate_time_fluctuation(self, op_type: OperationType, device_id: int) -> float:
        """模拟时间波动"""
        # 基础计算时间
        compute_time = self.time_model.get_compute_time(op_type, device_id)
        
        # 通信时间（除了Weight Update）
        if op_type != OperationType.WEIGHT_UPDATE:
            comm_time = self.time_model.get_communication_time(device_id)
            return compute_time + comm_time
        else:
            return compute_time
    
    def generate_events(self) -> List[Dict[str, Any]]:
        """生成P*M个事件"""
        # 清空现有事件列表，避免重复
        self.event_list = []
        
        # 生成所有Forward操作
        for mb in range(self.num_micro_batches):
            for device in range(self.num_devices):
                if self.check_operation_feasible(OperationType.FORWARD, mb, device):
                    event = self._create_event(OperationType.FORWARD, mb, device)
                    self.event_list.append(event)
        
        # 生成所有Backward操作
        for mb in range(self.num_micro_batches):
            for device in reversed(range(self.num_devices)):  # 反向顺序
                if self.check_operation_feasible(OperationType.BACKWARD, mb, device):
                    event = self._create_event(OperationType.BACKWARD, mb, device)
                    self.event_list.append(event)
        
        # 生成所有Weight Update操作
        for mb in range(self.num_micro_batches):
            for device in range(self.num_devices):
                if self.check_operation_feasible(OperationType.WEIGHT_UPDATE, mb, device):
                    event = self._create_event(OperationType.WEIGHT_UPDATE, mb, device)
                    self.event_list.append(event)
        
        return self.event_list
    
    def _create_event(self, op_type: OperationType, micro_batch_id: int, device_id: int) -> Dict[str, Any]:
        """创建事件"""
        # 模拟时间波动
        duration = self.simulate_time_fluctuation(op_type, device_id)
        
        # 创建操作对象
        operation = Operation(op_type, micro_batch_id, device_id)
        operation.duration = duration
        
        # 判断是否可行：只有F(0,0)一开始就可行，其余均不可行
        is_feasible = (op_type == OperationType.FORWARD and micro_batch_id == 0 and device_id == 0)
        
        # 创建事件
        event = {
            'type': op_type,
            'micro_batch_id': micro_batch_id,
            'device_id': device_id,
            'start_time': None,  # 开始时间为None
            'duration': duration,
            'end_time': None,    # 结束时间待计算
            'is_feasible': is_feasible,  # 是否可行
            'operation': operation
        }
        
        return event
    
    def get_event_list(self) -> List[Dict[str, Any]]:
        """获取事件表"""
        return self.event_list
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_events': len(self.event_list),
            'forward_events': sum(1 for event in self.event_list if event['type'] == OperationType.FORWARD),
            'backward_events': sum(1 for event in self.event_list if event['type'] == OperationType.BACKWARD),
            'weight_events': sum(1 for event in self.event_list if event['type'] == OperationType.WEIGHT_UPDATE),
            'max_memory_used': self.stats['max_memory_used'],
            'memory_violations': self.stats['memory_violations']
        }
