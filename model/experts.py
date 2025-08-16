import logging
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType

class BaseExpertNetwork(nn.Module):
    """
    基础专家网络（原始模型）
    这是一个简单的前馈神经网络，将作为LoRA的基础模型
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LoRAExpert(nn.Module):
    """
    使用PEFT库的LoRA专家模块
    每个专家都是基于PEFT库实现的LoRA结构
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout: float = 0.1,
                 expert_id: int = 0,
                 lora_r: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        """
        初始化PEFT LoRA专家

        Args:
            input_dim: 输入维度（来自Llama的hidden_size）
            hidden_dim: 专家隐藏层维度
            output_dim: 输出维度
            dropout: 基础网络dropout概率
            expert_id: 专家ID，用于标识和日志
            lora_r: LoRA的秩
            lora_alpha: LoRA的缩放参数
            lora_dropout: LoRA dropout概率
        """
        super().__init__()
        self.expert_id = expert_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # 1. 创建基础专家网络
        self.base_model = BaseExpertNetwork(input_dim, hidden_dim, output_dim, dropout)

        # 2. 使用PEFT库应用LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # 任务类型
            inference_mode=False,  # 训练模式
            r=lora_r,  # LoRA的秩
            lora_alpha=lora_alpha,  # LoRA的缩放参数
            lora_dropout=lora_dropout,  # LoRA的dropout
            target_modules=["network.0", "network.3", "network.6"],  # 要应用LoRA的线性层
        )

        # 3. 将基础模型转换为LoRA模型
        self.model = get_peft_model(self.base_model, lora_config)

        logging.info(f"PEFT LoRA Expert {expert_id} initialized with r={lora_r}, alpha={lora_alpha}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA专家前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            output: 专家输出 [batch_size, output_dim]
        """
        return self.model(x)

    def freeze_base_layers(self):
        """冻结所有基础层，只训练LoRA适配器"""
        # PEFT模型默认已经冻结了基础层参数
        # 这里可以显式确保基础层被冻结
        for name, param in self.model.named_parameters():
            if 'lora_' not in name.lower():
                param.requires_grad = False

    def unfreeze_base_layers(self):
        """解冻所有基础层"""
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def get_lora_parameters(self):
        """获取LoRA参数（用于优化器）"""
        lora_params = []
        for name, param in self.model.named_parameters():
            if 'lora_' in name.lower():
                lora_params.append(param)
        return lora_params

    def get_trainable_parameters(self):
        """获取可训练参数统计"""
        # 使用PEFT内置的参数统计功能
        if hasattr(self.model, 'print_trainable_parameters'):
            # 这会打印详细的参数统计信息
            self.model.print_trainable_parameters()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # 计算LoRA参数数量
        lora_params_count = sum(p.numel() for p in self.get_lora_parameters())

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lora_params": lora_params_count,
            "trainable_percentage": trainable_params / total_params * 100 if total_params > 0 else 0
        }

    def enable_adapters(self):
        """启用LoRA适配器"""
        if hasattr(self.model, 'enable_adapters'):
            self.model.enable_adapters()

    def disable_adapters(self):
        """禁用LoRA适配器（使用原始模型）"""
        if hasattr(self.model, 'disable_adapters'):
            self.model.disable_adapters()

class ExpertLayer(nn.Module):
    """
    专家层：包含多个LoRA专家的集合
    所有专家都会参与计算，通过路由器权重进行加权求和
    不使用top-k稀疏激活，每个专家都有贡献
    """

    def __init__(self,
                 input_dim: int,
                 expert_hidden_dim: int,
                 output_dim: int,
                 num_experts: int = 6,
                 dropout: float = 0.1,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0):
        """
        初始化专家层

        Args:
            input_dim: 输入维度
            expert_hidden_dim: 每个专家的隐藏层维度
            output_dim: 输出维度
            num_experts: 专家数量
            dropout: dropout概率
            lora_rank: LoRA的秩
            lora_alpha: LoRA的缩放参数
        """
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.output_dim = output_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # 创建多个LoRA专家
        self.experts = nn.ModuleList([
            LoRAExpert(
                input_dim=input_dim,
                hidden_dim=expert_hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                expert_id=i,
                lora_r=lora_rank,
                lora_alpha=int(lora_alpha),
                lora_dropout=dropout
            ) for i in range(num_experts)
        ])

        logging.info(f"Full Expert layer initialized with {num_experts} experts (all experts participate)")

    def forward(self, features: torch.Tensor, expert_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        专家层前向传播 - 所有专家都参与计算

        计算流程：
        1. 所有6个专家都会对输入特征进行计算
        2. 使用路由器提供的权重对所有专家输出进行加权求和
        3. 不进行top-k选择，每个专家都有贡献（权重可能为0但会参与计算）

        Args:
            features: 输入特征 [batch_size, input_dim]
            expert_weights: 专家权重分布 [batch_size, num_experts] (softmax输出，和为1)

        Returns:
            weighted_output: 加权聚合的最终输出 [batch_size, output_dim]
            expert_outputs: 包含各专家输出的字典
        """
        batch_size = features.size(0)

        # 存储每个专家的输出
        expert_outputs = {}
        all_expert_outputs = []

        # 计算所有专家的输出（全专家参与）
        for i, expert in enumerate(self.experts):
            expert_output = expert(features)  # [batch_size, output_dim]
            expert_outputs[f"expert_{i}"] = expert_output
            all_expert_outputs.append(expert_output)

        # 堆叠所有专家输出 [batch_size, num_experts, output_dim]
        stacked_outputs = torch.stack(all_expert_outputs, dim=1)

        # 使用路由权重对所有专家输出进行加权求和
        # expert_weights: [batch_size, num_experts] -> [batch_size, num_experts, 1]
        expert_weights_expanded = expert_weights.unsqueeze(-1)

        # 最终输出 = Σ(expert_weight_i * expert_output_i) [batch_size, output_dim]
        weighted_output = torch.sum(stacked_outputs * expert_weights_expanded, dim=1)

        # 添加统计信息
        expert_outputs["stacked_outputs"] = stacked_outputs
        expert_outputs["expert_weights"] = expert_weights
        expert_outputs["weighted_output"] = weighted_output

        return weighted_output, expert_outputs

    def forward_single_expert(self, features: torch.Tensor, expert_id: int) -> torch.Tensor:
        """
        使用单个专家进行前向传播（用于分析和调试）

        Args:
            features: 输入特征 [batch_size, input_dim]
            expert_id: 专家ID

        Returns:
            output: 单个专家的输出 [batch_size, output_dim]
        """
        if expert_id >= self.num_experts:
            raise ValueError(f"Expert ID {expert_id} exceeds number of experts {self.num_experts}")

        return self.experts[expert_id](features)

    def get_expert_activations(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取所有专家的激活值（用于分析专家特化程度）

        Args:
            features: 输入特征 [batch_size, input_dim]

        Returns:
            activations: 包含所有专家激活值的字典
        """
        activations = {}

        for i, expert in enumerate(self.experts):
            with torch.no_grad():
                # 获取PEFT LoRA专家的激活值
                # 由于使用了PEFT库，我们只能获取最终输出
                expert_output = expert(features)
                activations[f"expert_{i}_output"] = expert_output.clone()

                # 如果需要中间激活，可以通过hook机制获取
                # 但这需要更复杂的实现，这里简化处理

        return activations

    def compute_expert_diversity(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算专家间的多样性（输出差异程度）

        Args:
            features: 输入特征 [batch_size, input_dim]

        Returns:
            diversity_score: 专家多样性分数
        """
        # 获取所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            output = expert(features)
            expert_outputs.append(output)

        # 堆叠输出 [batch_size, num_experts, output_dim]
        stacked_outputs = torch.stack(expert_outputs, dim=1)

        # 计算专家间的余弦相似度矩阵
        normalized_outputs = F.normalize(stacked_outputs, p=2, dim=-1)

        # 计算所有专家对之间的相似度
        similarity_matrix = torch.matmul(normalized_outputs, normalized_outputs.transpose(-2, -1))

        # 计算平均相似度（除对角线）
        mask = torch.eye(self.num_experts, device=similarity_matrix.device).bool()
        off_diagonal = similarity_matrix.masked_fill(mask.unsqueeze(0), 0)

        # 多样性 = 1 - 平均相似度
        avg_similarity = off_diagonal.sum() / (self.num_experts * (self.num_experts - 1) * features.size(0))
        diversity_score = 1 - avg_similarity

        return diversity_score

